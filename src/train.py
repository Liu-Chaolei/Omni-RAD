import argparse
import math
import os
import random
import shutil
import time
from datetime import datetime

import pyiqa
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from diffusers.utils.import_utils import is_xformers_available
from my_utils.datasets import ImageFolder
from my_utils.utils import RateDistortionLoss, logger_setup, get_raw_state_dict
from StableCodec import StableCodec
from my_utils import wandb_utils

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch_ema import ExponentialMovingAverage
from PIL import ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.cuda.amp import autocast
from contextlib import nullcontext

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def piecewise_lambda(current_step):
    if current_step < 5000:
        return 1.0
    elif current_step < 10000:
        return 0.4
    elif current_step < 15000:
        return 0.2
    else:
        return 0.02


def configure_optimizers(net, net_disc, train_config, world_size, stage, rank, save_path):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters

    assert len(inter_params) == 0

    if rank == 0:
        structure_logger = logger_setup(log_file_name='structure.log', log_file_folder_name=os.path.join(save_path, 'logs'))
        structure_logger.info(f"Number of parameters for Main Optimizer: {len(parameters)}")
        structure_logger.info("List of Main parameters:")
        for n in sorted(parameters):
            structure_logger.info(f"  - {n}: {params_dict[n].shape}")

        structure_logger.info(f"Number of parameters for Aux Optimizer: {len(aux_parameters)}")
        structure_logger.info("List of Aux parameters:")
        for n in sorted(aux_parameters):
            structure_logger.info(f"  - {n}: {params_dict[n].shape}")

    if train_config['scale_lr']:
        train_config['learning_rate'] = (
                train_config['learning_rate'] * train_config['gradient_accumulation_steps'] * train_config['batch_size'] * world_size
        )
    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=train_config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=train_config['adam_weight_decay'],
        eps=train_config['adam_epsilon'],
    )
    aux_optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=train_config['aux_learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=train_config['adam_weight_decay'],
        eps=train_config['adam_epsilon'],
    )
    if stage == 2:
        disc_optimizer = optim.AdamW(
            net_disc.parameters(),
            lr=train_config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=train_config['adam_weight_decay'],
            eps=train_config['adam_epsilon'],
        )
    else:
        disc_optimizer = None
    return optimizer, aux_optimizer, disc_optimizer


def log_validation(step, val_dataloader, model, criterion, iqa_psnr, save_val, save_steps, save_path, train_config):
    model.eval()
    device = next(model.parameters()).device
    rank = dist.get_rank()

    loss = AverageMeter()
    bpp = AverageMeter()
    mse_loss = AverageMeter()
    lpips = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        validx = 1
        for d in val_dataloader:
            d = d.to(device)
            output = model(d, freeze_aux=True, training=False)

            out_criterion = criterion(output, d)

            bpp.update(out_criterion["bpp"].item())
            loss.update(out_criterion["compression_loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            lpips.update(out_criterion["lpips"].item())
            x_hat_01 = ((output["x_hat"] + 1) / 2).clamp(0, 1)
            d_01 = ((d + 1) / 2).clamp(0, 1)
            psnr.update(iqa_psnr(x_hat_01, d_01).mean().item())
            if save_val and rank == 0 and step % save_steps == 0:
                out_img = output["x_hat"]
                out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
                output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))
                os.makedirs(save_path+'/valpics', exist_ok=True)
                outf = os.path.join(save_path, 'valpics', f'step{step}_preview_{validx}.png')
                output_pil.save(outf)
                validx += 1

    metrics = torch.tensor([loss.sum, bpp.sum, mse_loss.sum, lpips.sum, psnr.sum, loss.count], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_count = metrics[-1]
    if total_count > 0:
        avg_loss = (metrics[0] / total_count).item()
        avg_bpp = (metrics[1] / total_count).item()
        avg_mse = (metrics[2] / total_count).item()
        avg_lpips = (metrics[3] / total_count).item()
        avg_psnr = (metrics[4] / total_count).item()
    else:
        avg_loss = avg_bpp = avg_mse = avg_lpips = avg_psnr = 0.0
    if rank == 0:
        print(
            f"step {step}: Average losses:"
            f"\tLoss: {avg_loss:.3f} |"
            f"\tBpp: {avg_bpp:.4f} |"
            f"\tMSE loss: {avg_mse:.3f} |"
            f"\tLPIPS: {avg_lpips:.3f} |"
            f"\tPSNR: {avg_psnr:.3f} |"
        )

    return avg_loss, avg_bpp, avg_mse, avg_lpips, avg_psnr


def save_checkpoint(state, checkpoint_dir="./checkpoint", checkpoint_name="checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(state, file_path)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


class ConditionalPadding:
    def __init__(self, base_patch_size, padding_mode='reflect'):
        self.padding_mode = padding_mode
        self.base_patch_size = base_patch_size
    
    def __call__(self, img):
        ori_h, ori_w = img.shape[-2:]
        pad_h = (math.ceil(ori_h / self.base_patch_size[0])) * self.base_patch_size[0] - ori_h
        pad_w = (math.ceil(ori_w / self.base_patch_size[1])) * self.base_patch_size[1] - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode=self.padding_mode)
        return img_padded


def ddp_setup(rank: int, world_size: int, MASTER_PORT: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def reduce_tensor(tensor):
    if dist.is_available() and dist.is_initialized():
        n = dist.get_world_size()
    else:
        n = 1
    if n == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def load_train_objs(train_dataloader, stage, train_config, val_config, world_size, device, rank, save_path):
    net = StableCodec(config=train_config['model']).to(device)

    # Discriminator
    if stage == 2:
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=train_config['gan_loss_type'], device=device)
        net_disc = net_disc.to(device)
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)
    else:
        net_disc = None

    if train_config['enable_xformers_memory_efficient_attention']:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    optimizer, aux_optimizer, disc_optimizer = configure_optimizers(net, net_disc, train_config, world_size, stage, rank, save_path)
    num_warmup_steps_for_scheduler = train_config['lr_warmup_steps'] * world_size
    if train_config['max_train_steps'] is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / world_size)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / train_config['gradient_accumulation_steps'])
        num_training_steps_for_scheduler = (
                train_config['num_train_epochs'] * num_update_steps_per_epoch * world_size
        )
    else:
        num_training_steps_for_scheduler = train_config['max_train_steps'] * world_size
    if stage == 2:
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=piecewise_lambda
        )
        disc_lr_scheduler = get_scheduler(
            train_config['lr_scheduler'],
            optimizer=disc_optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=train_config['lr_num_cycles'],
            power=train_config['lr_power'],
        )
    else:
        lr_scheduler = get_scheduler(
            train_config['lr_scheduler'],
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
            num_cycles=train_config['lr_num_cycles'],
            power=train_config['lr_power'],
        )
        disc_lr_scheduler = None
    return net, net_disc, optimizer, aux_optimizer, disc_optimizer, lr_scheduler, disc_lr_scheduler


def prepare_dataloader(base_config, train_config, val_config, micro_batch_size):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.RandomCrop(
            train_config['patch_size'], 
            pad_if_needed=True,
            padding_mode='reflect'
        ),
    ])

    if val_config['crop_val']:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomCrop(
                train_config['patch_size'], 
                pad_if_needed=True,
                padding_mode='reflect'
            ),
            ])
    else:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConditionalPadding(base_config['model_stride']),
            ])

    train_dataset = ImageFolder(train_config['dataset'], split="train", transform=train_transforms)
    val_dataset = ImageFolder(train_config['dataset'], split="valid", transform=val_transforms)

    train_sampler=DistributedSampler(train_dataset)
    val_sampler=DistributedSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        num_workers=base_config['num_workers'],
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_config['val_batch_size'],
        num_workers=base_config['num_workers'],
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
    )
    return train_dataloader, val_dataloader


def main(
    rank: int,
    world_size: int,
    stage: bool,
    base_config: dict,
    train_config: dict,
    val_config: dict,
    MASTER_PORT: int,
    experiment_name: str,
):
    if base_config['global_seed'] is not None:
        seed = base_config['global_seed'] * world_size + rank
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(train_config['savepath'], f'stage{stage}', experiment_name)
    if rank==0:
        os.makedirs(save_path, exist_ok=True)
        train_logger = logger_setup(log_file_name='train.log', log_file_folder_name=os.path.join(save_path, 'logs'))
        val_logger = logger_setup(log_file_name='val.log', log_file_folder_name=os.path.join(save_path, 'logs'))
        train_logger.info("Create experiment save folder")
        val_logger.info("Create experiment save folder")
        shutil.copytree("configs", os.path.join(save_path, 'configs'))
        if train_config['wandb']:
            entity = os.environ.get("ENTITY")
            project = os.environ.get("PROJECT", "mysc")
            wandb_utils.initialize(entity, experiment_name, project)

    ddp_setup(rank, world_size, MASTER_PORT)
    device = torch.device(f"cuda:{rank}")

    grad_accum_steps = train_config['grad_accum_steps']
    global_batch_size = train_config['global_batch_size']
    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)

    use_bf16 = train_config["precision"] == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    train_dataloader, val_dataloader = prepare_dataloader(base_config, train_config, val_config, micro_batch_size)
    net, net_disc, optimizer, aux_optimizer, disc_optimizer, lr_scheduler, disc_lr_scheduler = load_train_objs(train_dataloader, stage, train_config, val_config, world_size, device, rank, save_path)

    if train_config['compile']:
        net.unet = torch.compile(net.unet)
        net.vae = torch.compile(net.vae)

    if train_config['save_ema']:
        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        if train_config['ema_position'] == 'cpu':
            ema_net.to('cpu')
        elif train_config['ema_position'] == 'gpu':
            ema_net.to(device)

    last_epoch = 0
    global_step = 0
    log_steps = 0
    if train_config['model']["resume_train"]:
        sd = torch.load(train_config['model']['codec_path'], map_location="cpu")
        if train_config['save_ema']:
            ema_net.load_state_dict(sd["ema_state_dict"])
        last_epoch = sd["epoch"]
        global_step = sd["global_step"]
        optimizer.load_state_dict(sd["optimizer"])
        aux_optimizer.load_state_dict(sd["aux_optimizer"])
        lr_scheduler.load_state_dict(sd["lr_scheduler"])
        del sd

    net = DDP(net, device_ids=[rank])
    if stage == 2:
        net_disc = DDP(net_disc, device_ids=[rank])
        # from train_s3diff
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

    criterion = RateDistortionLoss(
        lmbda=train_config['lambda'],
        clip_model_path=train_config['clip_model_path'],
        mse_coefficient=train_config['mse_coefficient'],
        lpips_coefficient=train_config['lpips_coefficient'],
        clip_coefficient=train_config['clip_coefficient'],
        )
    iqa_psnr = pyiqa.create_metric('psnr', device=device)

    start_time = time.time()
    epoch = last_epoch

    best_loss = float('inf')
    metric_names = ['Train/loss', 'Train/mse_loss', 'Train/lpips', 'Train/bpp', 'Train/y_bpp', 'Train/z_bpp', 'Train/loss_G', 'Train/loss_D']
    while global_step < args.max_train_steps:
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if rank==0:
            train_logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_dataloader.sampler.set_epoch(epoch)

        device = next(net.parameters()).device
        accum_counter = 0
        step_accum = {k: 0.0 for k in metric_names}
        running_metrics = {k: 0.0 for k in metric_names}
        for i, d in enumerate(train_dataloader):
            if global_step >= train_config['max_train_steps']:
                break
            net.train()
            if stage == 2:
                net_disc.train()
            d = d.to(device)

            is_accumulation_step = (accum_counter + 1) < grad_accum_steps
            net_context = net.no_sync() if is_accumulation_step else nullcontext()
            with net_context:
                with autocast(**autocast_kwargs):
                    output = net(d, freeze_aux=train_config['model']['freeze_aux_encoder'])
                    out_criterion = criterion(output, d)
                    generator_loss = out_criterion["compression_loss"]

                    if stage == 2:
                        """
                        Generator loss: fool the discriminator
                        """
                        requires_grad(net_disc, False)
                        net_disc.eval()
                        out_criterion["lossG"] = net_disc(output['x_hat'], for_G=True).mean() * train_config['gan_coefficient']
                        generator_loss += out_criterion["lossG"]
                
                    aux_loss = net.module.codec.aux_loss()

                (generator_loss / grad_accum_steps).backward()
                (aux_loss / grad_accum_steps).backward()

            accum_counter += 1

            if accum_counter == grad_accum_steps:
                if train_config['clip_max_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), train_config['clip_max_norm'])
                optimizer.step()
                lr_scheduler.step()

                if world_size > 1:
                    for param in aux_optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size
                aux_optimizer.step()

                if global_step % train_config['ema_update_every'] == 0 and train_config['save_ema']:
                    if train_config['ema_position'] == 'cpu':
                        ema_net.update([p.cpu() for p in net.module.parameters()])
                    elif train_config['ema_position'] == 'gpu':
                        ema_net.update(net.module.parameters())

                optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])
                aux_optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])

            if stage == 2:
                """
                Discriminator loss: fake image vs real image
                """
                disc_context = net_disc.no_sync() if is_accumulation_step else nullcontext()
                with disc_context:
                    with autocast(**autocast_kwargs): 
                        requires_grad(net_disc, True)
                        net_disc.train()
                        # real image
                        out_criterion["lossD_real"] = net_disc(d.detach(), for_real=True).mean()
                        # fake image
                        out_criterion["lossD_fake"] = net_disc(output['x_hat'].detach(), for_real=False).mean()

                    out_criterion["lossD"] = out_criterion["lossD_real"] + out_criterion["lossD_fake"]
                    (out_criterion["lossD"] / grad_accum_steps).backward()

                if accum_counter == grad_accum_steps:
                    if train_config['clip_max_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(net_disc.parameters(), train_config['clip_max_norm'])
                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])

            current_metrics = {
                'Train/loss': generator_loss.item(),
                'Train/mse_loss': out_criterion["mse_loss"].item(),
                'Train/lpips': out_criterion["lpips"].item(),
                'Train/bpp': out_criterion["bpp"].item(),
                'Train/y_bpp': out_criterion["y_bpp"].item(),
                'Train/z_bpp': out_criterion["z_bpp"].item(),
                'Train/loss_G': 0.0,
                'Train/loss_D': 0.0,
            }
            if stage == 2:
                current_metrics['Train/loss_G'] = out_criterion["lossG"].item()
                current_metrics['Train/loss_D'] = out_criterion["lossD"].item()
            for k in metric_names:
                step_accum[k] += current_metrics[k]

            if accum_counter < grad_accum_steps:
                continue

            for k in metric_names:
                running_metrics[k] += step_accum[k] / grad_accum_steps
                step_accum[k] = 0.0
            log_steps += 1
            global_step += 1
            accum_counter = 0

            if global_step < 20000 or global_step % train_config['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                metrics_tensor = torch.tensor(
                    [running_metrics[k] / log_steps for k in metric_names], 
                    device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= world_size
                avg_metrics = {k: v.item() for k, v in zip(metric_names, metrics_tensor)}

                if rank == 0:
                    log_msg = f"step: {global_step} "
                    log_msg += ", ".join([f"Train {k.upper()}: {avg_metrics[k]:.4f}" for k in metric_names])
                    log_msg += f", Steps/Sec: {steps_per_sec:.2f}"
                    train_logger.info(log_msg)
                if train_config['wandb']:
                    wandb_log_dict = {f"train {k}": v for k, v in avg_metrics.items()}
                    wandb_log_dict["train steps/sec"] = steps_per_sec
                    wandb_utils.log(wandb_log_dict, step=global_step)

                running_metrics = {k: 0.0 for k in metric_names}
                log_steps = 0
                start_time = time.time()

            if global_step % train_config['checkpointing_steps'] == 0 and global_step > 0:
                if rank == 0:
                    vae_trainable_keys = {n for n, p in net.module.vae.named_parameters() if p.requires_grad}
                    unet_trainable_keys = {n for n, p in net.module.unet.named_parameters() if p.requires_grad}
                    ckpt_dict = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict_codec": net.module.codec.state_dict(),
                            "state_dict_aux_codec": net.module.aux_codec.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                    }
                    if train_config['save_ema']:
                        ckpt_dict["ema_state_dict"] = ema_net.state_dict()
                    ckpt_dict["state_dict_vae"] = {
                        k: v for k, v in get_raw_state_dict(net.module.vae).items() 
                        if k in vae_trainable_keys
                    }
                    ckpt_dict["state_dict_unet"] = {
                        k: v for k, v in get_raw_state_dict(net.module.unet).items() 
                        if k in unet_trainable_keys
                    }
                    save_checkpoint(
                        ckpt_dict,
                        checkpoint_dir = os.path.join(save_path, 'checkpoints'),
                        checkpoint_name = "checkpoint-{}.pth.tar".format(global_step)
                    )
                    del ckpt_dict
                dist.barrier()
                start_time = time.time()

            if global_step % val_config['validation_steps'] == 0 or (val_config['val_first'] and global_step == 1):
                optimizer.zero_grad(set_to_none=True)
                aux_optimizer.zero_grad(set_to_none=True)
                if disc_optimizer is not None:
                    disc_optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache() 
                if train_config['save_ema']:
                    ema_net.store(net.parameters())
                    ema_net.copy_to(net.parameters())
                val_loss, val_bpp, val_mse, val_lpips, val_psnr = log_validation(global_step, val_dataloader, net, criterion, iqa_psnr, val_config['save_val'], val_config['save_steps'], save_path, train_config)
                if train_config['save_ema']:
                    ema_net.restore(net.parameters())
                torch.cuda.empty_cache()
                if rank==0:
                    val_logger.info(f'step: {global_step}, val/loss: {val_loss:.4f}, val/lpips: {val_lpips:.3f}, val/psnr: {val_psnr:.2f}, val/bpp: {val_bpp:.4f}')
                    if train_config['save_best'] and val_loss <= best_loss and global_step > train_config['min_steps_for_best']:
                        vae_trainable_keys = {n for n, p in net.module.vae.named_parameters() if p.requires_grad}
                        unet_trainable_keys = {n for n, p in net.module.unet.named_parameters() if p.requires_grad}
                        best_loss = val_loss
                        ckpt_dict = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "state_dict_codec": net.module.codec.state_dict(),
                                "state_dict_aux_codec": net.module.aux_codec.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "aux_optimizer": aux_optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                        }
                        if train_config['save_ema']:
                            ckpt_dict["ema_state_dict"] = ema_net.state_dict()
                        ckpt_dict["state_dict_vae"] = {
                            k: v for k, v in get_raw_state_dict(net.module.vae).items() 
                            if k in vae_trainable_keys
                        }
                        ckpt_dict["state_dict_unet"] = {
                            k: v for k, v in get_raw_state_dict(net.module.unet).items() 
                            if k in unet_trainable_keys
                        }
                        save_checkpoint(
                            ckpt_dict,
                            checkpoint_dir = os.path.join(save_path, 'checkpoints'),
                            checkpoint_name = "best_checkpoint.pth.tar"
                        )
                        del ckpt_dict
                        val_logger.info(f'New best model saved at step {global_step} with LOSS: {best_loss:.3f}')
                start_time = time.time()
        epoch += 1

    dist.destroy_process_group()


# CUDA_VISIBLE_DEVICES=5,6 python src/train.py --stage 2 --experiment_name sc-finetune
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--config_dir", type=str, default="./configs/")
    parser.add_argument("--MASTER_PORT", type=int, default=12355)
    parser.add_argument("--experiment_name", type=str, default=None)

    args = parser.parse_args()
    base_config = load_yaml(args.config_dir+'base.yaml')
    train_config = load_yaml(args.config_dir+f"stage{args.stage}.yaml")
    val_config = load_yaml(args.config_dir+'val.yaml')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.stage, base_config, train_config, val_config, args.MASTER_PORT, args.experiment_name), nprocs=world_size)
