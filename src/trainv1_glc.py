"""
trainv1_glc.py — Training script for GLC variable-rate StableCodec.

Key changes from trainv1_variable3_debug_step.py:
  1. Discrete quality_index sampling: torch.randint(0, NUM_QUALITY, (B,))
  2. Loss: L = mean(D_i / λ_i) + mean(bpp_i), λ_i = LAMBDA_TABLE[quality_index_i]
  3. Imports StableCodec_glc (GLC global/local scalers, no λ-FiLM)
  4. Forward: net(x, pos_prompt, H, W, quality_index=qi) — no lmbda, no rho
  5. Diagnostics adapted for discrete quality levels
  6. Removed: lambda_min/lambda_max/lambda_sample_max, rho schedule
"""

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
from my_utils.utils import RateDistortionLoss_v7, logger_setup, get_raw_state_dict
from StableCodec_glc import StableCodec
from latent_codec_glc import NUM_QUALITY, LAMBDA_TABLE
from my_utils import wandb_utils
from my_utils.training_utils import parse_args_training, H5Dataset

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch_ema import ExponentialMovingAverage
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
import torch._dynamo
from peft.tuners.lora.layer import LoraLayer

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def unwrap_model(accelerator, model):
    """unwrap_model that works around accelerate bug with partially compiled models."""
    options = [torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel]
    while isinstance(model, tuple(options)):
        model = model.module
    return model


# ---------------------------------------------------------------------------
# Quality level sampling helper
# ---------------------------------------------------------------------------

def sample_quality_index(B: int, device) -> torch.Tensor:
    """Sample B discrete quality indices uniformly in [0, NUM_QUALITY)."""
    return torch.randint(0, NUM_QUALITY, (B,), device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def piecewise_lambda(current_step):
    if   current_step < 5000:
        return 1.0
    elif current_step < 10000:
        return 0.4
    elif current_step < 15000:
        return 0.2
    else:
        return 0.02


def configure_optimizers(net, net_disc, train_config, world_size, stage, rank, save_path, accelerator, structure_logger=None):
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    assert len(parameters & aux_parameters) == 0

    if accelerator.is_main_process:
        if structure_logger is None:
            structure_logger = logger_setup(log_file_name='structure.log', log_file_folder_name=os.path.join(save_path, 'logs'))
        structure_logger.info(f"Main Parameters ({len(parameters)}): {sorted(list(parameters))}")
        structure_logger.info(f"Aux Parameters ({len(aux_parameters)}): {sorted(list(aux_parameters))}")

    if train_config['scale_lr']:
        train_config['learning_rate'] = (
            train_config['learning_rate']
            * train_config['gradient_accumulation_steps']
            * train_config['batch_size'] * world_size
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
    disc_optimizer = None
    if stage == 2:
        disc_optimizer = optim.AdamW(
            net_disc.parameters(),
            lr=train_config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=train_config['adam_weight_decay'],
            eps=train_config['adam_epsilon'],
        )
    return optimizer, aux_optimizer, disc_optimizer


def log_validation(step, val_dataloader, model, criterion, iqa_psnr,
                   save_val, save_steps, save_path, train_config, accelerator, stage):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp = AverageMeter()
    mse_loss = AverageMeter()
    lpips = AverageMeter()
    psnr = AverageMeter()
    t_star_mean_meter = AverageMeter()

    # Validate at a reference quality level (default: qi=0, λ=2)
    val_qi = int(train_config['model'].get('val_quality_index', 0))

    with torch.no_grad():
        validx = 1
        for d in val_dataloader:
            d = d.to(device)
            B, C, H, W = d.shape
            pos_tag_prompt = [1 for _ in range(B)]
            qi_val = torch.full((B,), val_qi, dtype=torch.long, device=device)
            x_hat, RateLossOutput, T_star = model(d, pos_tag_prompt, H, W, quality_index=qi_val)

            out_criterion = criterion(x_hat, d)
            out_criterion["bpp"] = RateLossOutput.quantized_total_bpp.detach()

            bpp.update(out_criterion["bpp"].item())
            loss.update(out_criterion["compression_loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            lpips.update(out_criterion["lpips"].item())
            x_hat_01 = ((x_hat + 1) / 2).clamp(0, 1)
            d_01     = ((d     + 1) / 2).clamp(0, 1)
            psnr.update(iqa_psnr(x_hat_01, d_01).mean().item())
            t_star_mean_meter.update(T_star.mean().item())

            if save_val and accelerator.is_main_process and step % save_steps == 0 and validx <= 5:
                out_img = (x_hat * 0.5 + 0.5).float().cpu().detach()
                out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
                os.makedirs(save_path + '/valpics', exist_ok=True)
                out_pil.save(os.path.join(save_path, 'valpics', f'step{step}_preview_{validx}.png'))
                validx += 1

    metrics = torch.tensor(
        [loss.sum, bpp.sum, mse_loss.sum, lpips.sum, psnr.sum, t_star_mean_meter.sum, loss.count],
        device=device)
    accelerator.reduce(metrics, reduction='sum')
    total_count = metrics[-1]
    if total_count > 0:
        avg_loss  = (metrics[0] / total_count).item()
        avg_bpp   = (metrics[1] / total_count).item()
        avg_mse   = (metrics[2] / total_count).item()
        avg_lpips = (metrics[3] / total_count).item()
        avg_psnr  = (metrics[4] / total_count).item()
    else:
        avg_loss = avg_bpp = avg_mse = avg_lpips = avg_psnr = 0.0

    if accelerator.is_main_process:
        print(
            f"step {step}: Loss:{avg_loss:.3f} | Bpp:{avg_bpp:.4f} | "
            f"MSE:{avg_mse:.3f} | LPIPS:{avg_lpips:.3f} | PSNR:{avg_psnr:.3f} | "
            f"T*:{(metrics[5] / total_count).item():.1f}")
    return avg_loss, avg_bpp, avg_mse, avg_lpips, avg_psnr


def save_checkpoint(state, checkpoint_dir="./checkpoint", checkpoint_name="checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, checkpoint_name))


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


class ConditionalPadding:
    def __init__(self, base_patch_size, padding_mode='reflect'):
        self.padding_mode  = padding_mode
        self.base_patch_size = base_patch_size

    def __call__(self, img):
        ori_h, ori_w = img.shape[-2:]
        pad_h = (math.ceil(ori_h / self.base_patch_size[0])) * self.base_patch_size[0] - ori_h
        pad_w = (math.ceil(ori_w / self.base_patch_size[1])) * self.base_patch_size[1] - ori_w
        return F.pad(img, pad=(0, pad_w, 0, pad_h), mode=self.padding_mode)


def load_train_objs(train_dataloader, stage, train_config, val_config,
                    world_size, device, rank, save_path, accelerator):
    if accelerator.is_main_process:
        structure_logger = logger_setup(log_file_name='structure.log', log_file_folder_name=os.path.join(save_path, 'logs'))
    else:
        structure_logger = None
    net = StableCodec(
        sd_path=train_config['model']['sd_path'],
        config=train_config['model'],
        logger=structure_logger,
    ).to(device)

    net_disc = None
    if stage == 2:
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type='dino', output_type='conv_multi_level',
            loss_type=train_config['gan_loss_type'], device=device)
        net_disc = net_disc.to(device)
        net_disc.requires_grad_(True)
        net_disc.cv_ensemble.requires_grad_(False)

    if train_config['enable_xformers_memory_efficient_attention']:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available.")
    if train_config['gradient_checkpointing']:
        net.unet.enable_gradient_checkpointing()
    if train_config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer, aux_optimizer, disc_optimizer = configure_optimizers(
        net, net_disc, train_config, world_size, stage, rank, save_path, accelerator, structure_logger=structure_logger)

    num_warmup = train_config['lr_warmup_steps'] * world_size
    if train_config['max_train_steps'] is None:
        num_steps = (
            math.ceil(len(train_dataloader) / world_size)
            // train_config['gradient_accumulation_steps']
        ) * train_config['num_train_epochs'] * world_size
    else:
        num_steps = train_config['max_train_steps'] * world_size

    if stage == 2:
        lr_scheduler      = LambdaLR(optimizer, lr_lambda=piecewise_lambda)
        disc_lr_scheduler = get_scheduler(
            train_config['lr_scheduler'], optimizer=disc_optimizer,
            num_warmup_steps=num_warmup, num_training_steps=num_steps,
            num_cycles=train_config['lr_num_cycles'], power=train_config['lr_power'])
    else:
        lr_scheduler = get_scheduler(
            train_config['lr_scheduler'], optimizer=optimizer,
            num_warmup_steps=num_warmup, num_training_steps=num_steps,
            num_cycles=train_config['lr_num_cycles'], power=train_config['lr_power'])
        disc_lr_scheduler = None

    return net, net_disc, optimizer, aux_optimizer, disc_optimizer, lr_scheduler, disc_lr_scheduler


def prepare_dataloader(base_config, train_config, val_config, micro_batch_size):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(train_config['patch_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    if val_config['crop_val']:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomCrop(
                train_config['patch_size'], pad_if_needed=True, padding_mode='reflect'),
        ])
    else:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConditionalPadding(base_config['model_stride']),
        ])
    train_dataset = H5Dataset(base_config['hdf5_dataset'],   transform=train_transforms)
    val_dataset   = ImageFolder(base_config['train_dataset'], split="valid", transform=val_transforms)
    train_dataloader = DataLoader(
        train_dataset, batch_size=micro_batch_size,
        num_workers=base_config['num_workers'], shuffle=True, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_config['val_batch_size'],
        num_workers=base_config['num_workers'], shuffle=True, pin_memory=True)
    return train_dataloader, val_dataloader


def main(stage, base_config, train_config, val_config, MASTER_PORT, experiment_name):

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        mixed_precision=train_config['mixed_precision'],
        log_with=train_config['report_to'],
    )
    world_size = accelerator.num_processes
    rank       = accelerator.process_index

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
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        train_logger = logger_setup('train.log', os.path.join(save_path, 'logs'))
        val_logger   = logger_setup('val.log',   os.path.join(save_path, 'logs'))
        train_logger.info("Created experiment folder")
        shutil.copytree("configs", os.path.join(save_path, 'configs'))
        if train_config['wandb']:
            entity  = os.environ.get("ENTITY")
            project = os.environ.get("PROJECT", "sc")
            wandb_utils.initialize(entity, experiment_name, project)

    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    global_batch_size = train_config['global_batch_size']
    if global_batch_size % (world_size * gradient_accumulation_steps) != 0:
        raise ValueError("global_batch_size must be divisible by world_size * gradient_accumulation_steps.")
    micro_batch_size = global_batch_size // (world_size * gradient_accumulation_steps)

    train_dataloader, val_dataloader = prepare_dataloader(base_config, train_config, val_config, micro_batch_size)
    net, net_disc, optimizer, aux_optimizer, disc_optimizer, lr_scheduler, disc_lr_scheduler = \
        load_train_objs(train_dataloader, stage, train_config, val_config,
                        world_size, device, rank, save_path, accelerator)

    if train_config['compile']:
        torch._dynamo.config.optimize_ddp = False
        LoraLayer.unscale_layer = torch._dynamo.disable(LoraLayer.unscale_layer)
        unwrap_net = unwrap_model(accelerator, net)
        unwrap_net.unet = torch.compile(unwrap_net.unet)

    ema_net = None
    if train_config['save_ema']:
        ema_net = ExponentialMovingAverage(net.parameters(), decay=0.999)
        ema_net.to('cpu' if train_config['ema_position'] == 'cpu' else device)

    last_epoch = 0
    global_step = 0
    log_steps = 0

    if train_config['model'].get("resume_train"):
        sd = torch.load(train_config['model']['codec_path'], map_location="cpu")
        if train_config['save_ema']:
            ema_net.load_state_dict(sd["ema_state_dict"])
        last_epoch  = sd["epoch"]
        global_step = sd["global_step"]
        try:
            optimizer.load_state_dict(sd["optimizer"])
            aux_optimizer.load_state_dict(sd["aux_optimizer"])
            lr_scheduler.load_state_dict(sd["lr_scheduler"])
        except ValueError as e:
            print(f"[WARNING] Optimizer state dict mismatch, skipping optimizer resume: {e}")
            print("[WARNING] Training will resume from step/epoch only, LR scheduler reset.")
            if stage == 2 and "state_dict_disc" in sd and "disc_optimizer" in sd and "disc_lr_scheduler" in sd:
                net_disc.load_state_dict(sd["state_dict_disc"])
                disc_optimizer.load_state_dict(sd["disc_optimizer"])
                disc_lr_scheduler.load_state_dict(sd["disc_lr_scheduler"])
        del sd

    if stage == 2:
        net, optimizer, aux_optimizer, train_dataloader, val_dataloader, net_disc, disc_optimizer = \
            accelerator.prepare(net, optimizer, aux_optimizer, train_dataloader, val_dataloader, net_disc, disc_optimizer)
    else:
        net, optimizer, aux_optimizer, train_dataloader, val_dataloader = \
            accelerator.prepare(net, optimizer, aux_optimizer, train_dataloader, val_dataloader)

    net.to(accelerator.device, dtype=weight_dtype)
    if stage == 2:
        net_disc.to(accelerator.device, dtype=weight_dtype)

    # Lambda table tensor for per-image loss weighting
    lambda_table_t = torch.tensor(LAMBDA_TABLE, dtype=torch.float32, device=device)

    # Reference λ for criterion (used internally by RateDistortionLoss_v7)
    lambda_ref = float(train_config['model'].get('lambda_ref', train_config.get('lmbda', 2.0)))

    criterion = RateDistortionLoss_v7(
        lmbda           = lambda_ref,
        clip_model_path = train_config['clip_model_path'],
        mse_coefficient = train_config['mse_coefficient'],
        lpips_coefficient = train_config['lpips_coefficient'],
        clip_coefficient  = train_config['clip_coefficient'],
    )
    iqa_psnr = pyiqa.create_metric('psnr', device=device)

    start_time = time.time()
    epoch      = last_epoch
    best_loss  = float('inf')

    metric_names = ['Train/loss', 'Train/mse_loss', 'Train/lpips', 'Train/bpp',
                    'Train/y_bpp', 'Train/z_bpp', 'Train/loss_G', 'Train/loss_D', 'Train/aux_loss',
                    'Train/T_star_mean', 'Train/T_star_std']

    # -------------------- Quality-BPP diagnostics --------------------
    diag_every    = int(train_config.get('diag_every', 500))
    diag_buffer   = {"qi": [], "bpp": [], "dist": [], "t_star": []}

    def _log_diag_and_clear(logger, step):
        qi_arr   = np.asarray(diag_buffer["qi"],    dtype=np.int64)
        bpp_arr  = np.asarray(diag_buffer["bpp"],   dtype=np.float64)
        dist_arr = np.asarray(diag_buffer["dist"],  dtype=np.float64)
        tstar_arr = np.asarray(diag_buffer["t_star"], dtype=np.float64)
        if qi_arr.size == 0:
            return
        n = min(qi_arr.size, bpp_arr.size, dist_arr.size, tstar_arr.size)
        qi_arr, bpp_arr, dist_arr, tstar_arr = qi_arr[:n], bpp_arr[:n], dist_arr[:n], tstar_arr[:n]
        msg = (f"[DIAG step={step}] n={n} "
               f"mean_bpp={bpp_arr.mean():.6f} "
               f"range_bpp=({bpp_arr.min():.6f},{bpp_arr.max():.6f}) "
               f"mean_T*={tstar_arr.mean():.1f}")
        print(msg)
        if logger is not None:
            logger.info(msg)
        # Per quality-level breakdown
        for qi_val in range(NUM_QUALITY):
            m = (qi_arr == qi_val)
            if not np.any(m):
                continue
            bmsg = (f"[DIAG-QI step={step}] qi={qi_val} λ={LAMBDA_TABLE[qi_val]:.1f} "
                    f"n={m.sum()} "
                    f"mean_bpp={bpp_arr[m].mean():.6f} "
                    f"mean_dist={dist_arr[m].mean():.6f} "
                    f"mean_T*={tstar_arr[m].mean():.1f}")
            print(bmsg)
            if logger is not None:
                logger.info(bmsg)
        for k in diag_buffer:
            diag_buffer[k].clear()
    # -----------------------------------------------------------

    # Explicitly zero gradients before entering the loop
    optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    if disc_optimizer:
        disc_optimizer.zero_grad(set_to_none=True)

    while global_step < train_config['max_train_steps']:
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if accelerator.is_main_process:
            train_logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        device = next(net.parameters()).device
        step_accum    = {k: 0.0 for k in metric_names}
        running_metrics = {k: 0.0 for k in metric_names}

        for i, d in enumerate(train_dataloader):
            if global_step >= train_config['max_train_steps']:
                break

            net.train()
            if stage == 2:
                net_disc.train()

            with accelerator.accumulate(net):
                d = d.to(device)
                B, C, H, W = d.shape
                pos_tag_prompt = [1 for _ in range(B)]

                # --------------------------------------------------------
                # Sample per-image discrete quality index
                # --------------------------------------------------------
                qi = sample_quality_index(B, device)

                x_hat, RateLossOutput, T_star = net(d, pos_tag_prompt,
                                                    train_config['patch_size'][0],
                                                    train_config['patch_size'][1],
                                                    quality_index=qi)
                d = d.detach().float()

                # Distortion loss from criterion
                out_criterion = criterion(x_hat, d)

                # --------------------------------------------------------
                # Variable-rate loss:  L = mean(D_i / λ_i) + mean(bpp_i)
                #
                #   - High λ → D/λ small → bpp dominates → lower bpp
                #   - Low  λ → D/λ large → distortion dominates → higher bpp
                # --------------------------------------------------------
                per_image_dist = out_criterion["per_image_distortion"]  # [B]
                lmbda_per_image = lambda_table_t[qi]                    # [B]
                inv_lambda_distortion = (per_image_dist / lmbda_per_image).mean()

                # ---- diagnostics collect (detached, per-image) ----
                with torch.no_grad():
                    diag_buffer["qi"].extend(qi.detach().cpu().numpy().tolist())
                    diag_buffer["bpp"].extend(RateLossOutput.per_image_bpp.detach().cpu().numpy().tolist())
                    diag_buffer["dist"].extend(per_image_dist.detach().cpu().numpy().tolist())
                    t_star_val = T_star.detach().cpu().reshape(-1)
                    if t_star_val.numel() == 1:
                        t_star_val = t_star_val.expand(B)
                    diag_buffer["t_star"].extend(t_star_val.numpy().tolist())

                out_criterion['bpp_loss'] = RateLossOutput.rate_loss
                out_criterion['bpp']      = RateLossOutput.quantized_total_bpp.detach()
                out_criterion['y_bpp']    = RateLossOutput.quantized_latent_bpp.detach()
                out_criterion['z_bpp']    = RateLossOutput.quantized_hyper_bpp.detach()

                # L = mean(D_i / λ_i) + mean(bpp_i)
                generator_loss  = inv_lambda_distortion + out_criterion['bpp_loss']
                out_criterion["compression_loss"] = generator_loss

                current_metrics = {
                    'Train/loss':        generator_loss.item(),
                    'Train/mse_loss':    out_criterion["mse_loss"].item(),
                    'Train/lpips':       out_criterion["lpips"].item(),
                    'Train/bpp':         out_criterion["bpp"].item(),
                    'Train/y_bpp':       out_criterion["y_bpp"].item(),
                    'Train/z_bpp':       out_criterion["z_bpp"].item(),
                    'Train/loss_G':      0.0,
                    'Train/loss_D':      0.0,
                    'Train/aux_loss':    0.0,
                    # Dynamic timestep monitoring
                    'Train/T_star_mean': T_star.mean().item(),
                    'Train/T_star_std':  T_star.std().item() if T_star.numel() > 1 else 0.0,
                }

                if stage == 2:
                    requires_grad(net_disc, False)
                    net_disc.eval()
                    out_criterion["lossG"] = net_disc(x_hat, for_G=True).mean() * train_config['gan_coefficient']
                    generator_loss         = generator_loss + out_criterion["lossG"]
                    current_metrics['Train/loss_G'] = out_criterion["lossG"].item()

                aux_loss = unwrap_model(accelerator, net).codec.aux_loss()

                accelerator.backward(generator_loss)
                accelerator.backward(aux_loss)

                # Record aux_loss for logging
                current_metrics['Train/aux_loss'] = aux_loss.item()

                if accelerator.sync_gradients and train_config['clip_max_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), train_config['clip_max_norm'])
                optimizer.step()
                lr_scheduler.step()

                if world_size > 1:
                    for param in aux_optimizer.param_groups[0]['params']:
                        if param.grad is not None:
                            accelerator.reduce(param.grad.data, reduction='sum')
                            param.grad.data /= world_size
                aux_optimizer.step()

                if global_step % train_config['ema_update_every'] == 0 and train_config['save_ema']:
                    if train_config['ema_position'] == 'cpu':
                        ema_net.update([p.cpu() for p in unwrap_model(accelerator, net).parameters()])
                    else:
                        ema_net.update(unwrap_model(accelerator, net).parameters())

                optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])
                aux_optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])

            if stage == 2:
                with accelerator.accumulate(net_disc):
                    requires_grad(net_disc, True)
                    unwrap_disc = accelerator.unwrap_model(net_disc)
                    if hasattr(unwrap_disc, 'cv_ensemble'):
                        unwrap_disc.cv_ensemble.requires_grad_(False)
                    net_disc.train()
                    out_criterion["lossD_real"] = net_disc(d.detach(), for_real=True).mean()
                    accelerator.backward(out_criterion["lossD_real"])
                    out_criterion["lossD_fake"] = net_disc(x_hat.detach(), for_real=False).mean()
                    accelerator.backward(out_criterion["lossD_fake"])
                    out_criterion["lossD"] = out_criterion["lossD_real"] + out_criterion["lossD_fake"]
                    current_metrics['Train/loss_D'] = out_criterion["lossD"].item()
                    if accelerator.sync_gradients and train_config['clip_max_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(net_disc.parameters(), train_config['clip_max_norm'])
                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=train_config['set_grads_to_none'])

            if accelerator.sync_gradients:
                log_steps   += 1
                global_step += 1
                for k in metric_names:
                    step_accum[k] += current_metrics[k]

                # ---- Quality-BPP diagnostics ----
                if accelerator.is_main_process and (
                    global_step <= 20 or (global_step > 0 and global_step % diag_every == 0)
                ):
                    _log_diag_and_clear(train_logger, global_step)

                if accelerator.is_main_process:
                    for k in metric_names:
                        running_metrics[k] += step_accum[k] / gradient_accumulation_steps
                        step_accum[k] = 0.0

                if global_step < 20000 or global_step % train_config['log_every'] == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    metrics_tensor = torch.tensor(
                        [running_metrics[k] / log_steps for k in metric_names], device=device)
                    accelerator.reduce(metrics_tensor, reduction='sum')
                    metrics_tensor /= world_size
                    avg_metrics = {k: v.item() for k, v in zip(metric_names, metrics_tensor)}

                    if accelerator.is_main_process:
                        log_msg = f"step:{global_step} " + \
                            ", ".join([f"{k}:{avg_metrics[k]:.4f}" for k in metric_names]) + \
                            f", Steps/Sec:{steps_per_sec:.2f}"
                        train_logger.info(log_msg)
                    if train_config['wandb']:
                        wandb_utils.log({f"train {k}": v for k, v in avg_metrics.items()}, step=global_step)

                    running_metrics = {k: 0.0 for k in metric_names}
                    log_steps = 0
                    start_time = time.time()

                if global_step % train_config['checkpointing_steps'] == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        _save_ckpt(net, ema_net, optimizer, aux_optimizer, lr_scheduler,
                                   epoch, global_step, save_path, train_config, accelerator,
                                   f"checkpoint-{global_step}.pth.tar",
                                   net_disc=net_disc, disc_optimizer=disc_optimizer,
                                   disc_lr_scheduler=disc_lr_scheduler)

                if (global_step % val_config['validation_steps'] == 0
                        or (val_config['val_first'] and global_step == 1)):
                    optimizer.zero_grad(set_to_none=True)
                    aux_optimizer.zero_grad(set_to_none=True)
                    if disc_optimizer:
                        disc_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if train_config['save_ema']:
                        ema_net.store(net.parameters())
                        ema_net.copy_to(net.parameters())
                    val_loss, val_bpp, val_mse, val_lpips, val_psnr = log_validation(
                        global_step, val_dataloader, net, criterion, iqa_psnr,
                        val_config['save_val'], val_config['save_steps'],
                        save_path, train_config, accelerator, stage)
                    if train_config['save_ema']:
                        ema_net.restore(net.parameters())
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        val_logger.info(
                            f'step: {global_step}, '
                            f'val/loss: {val_loss:.4f}, '
                            f'val/mse: {val_mse:.4f}, '
                            f'val/lpips: {val_lpips:.3f}, '
                            f'val/psnr: {val_psnr:.2f}, '
                            f'val/bpp: {val_bpp:.4f}'
                        )
                        if train_config['wandb']:
                            wandb_utils.log({
                                'val/loss':  val_loss,
                                'val/mse':   val_mse,
                                'val/lpips': val_lpips,
                                'val/psnr':  val_psnr,
                                'val/bpp':   val_bpp,
                            }, step=global_step)
                        if (train_config['save_best'] and val_loss <= best_loss
                                and global_step > train_config['min_steps_for_best']):
                            best_loss = val_loss
                            _save_ckpt(net, ema_net, optimizer, aux_optimizer, lr_scheduler,
                                       epoch, global_step, save_path, train_config, accelerator,
                                       "best_checkpoint.pth.tar",
                                       net_disc=net_disc, disc_optimizer=disc_optimizer,
                                       disc_lr_scheduler=disc_lr_scheduler)
                            val_logger.info(f"New best at step {global_step}: {best_loss:.3f}")
                    start_time = time.time()
        epoch += 1
    accelerator.end_training()



def _save_ckpt(net, ema_net, optimizer, aux_optimizer, lr_scheduler,
               epoch, global_step, save_path, train_config, accelerator, filename,
               net_disc=None, disc_optimizer=None, disc_lr_scheduler=None):
    """Build and save a checkpoint dict including GLC variable-rate module weights."""
    unwrapped = unwrap_model(accelerator, net)
    vae_trainable_keys  = {n for n, p in unwrapped.vae.named_parameters() if p.requires_grad}
    raw_unet = unwrapped.unet._orig_mod if hasattr(unwrapped.unet, '_orig_mod') else unwrapped.unet
    unet_trainable_keys = {n for n, p in raw_unet.named_parameters() if p.requires_grad}

    def _extract_mismatched_keys(loading_info, raw_sd):
        raw_mismatched = loading_info.get("mismatched_keys", [])
        key_set = set()
        for item in raw_mismatched:
            key = item[0] if isinstance(item, (list, tuple)) else item
            key_set.add(key)
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                key_set.add(f"{parts[0]}.base_layer.{parts[1]}")
        return {k for k in key_set if k in raw_sd}

    vae_raw_sd  = get_raw_state_dict(unwrapped.vae)
    unet_raw_sd = get_raw_state_dict(unwrapped.unet)

    vae_mismatched_keys  = _extract_mismatched_keys(unwrapped.vae_loading_info,  vae_raw_sd)
    unet_mismatched_keys = _extract_mismatched_keys(unwrapped.unet_loading_info, unet_raw_sd)

    ckpt_dict = {
        "epoch":            epoch,
        "global_step":      global_step,
        "state_dict_codec": unwrapped.codec.state_dict(),
        "state_dict_aux_codec": unwrapped.aux_codec.state_dict(),
        # GLC variable-rate: UNet LoRA projection (Injection Point 3)
        "state_dict_lora_proj": unwrapped.unet_lora_proj.state_dict(),
        # GLC variable-rate: Quality embedding module
        "state_dict_quality_embed": unwrapped.quality_embed.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "aux_optimizer":    aux_optimizer.state_dict(),
        "lr_scheduler":     lr_scheduler.state_dict(),
    }
    if train_config['save_ema']:
        ckpt_dict["ema_state_dict"] = ema_net.state_dict()
    if net_disc is not None:
        ckpt_dict["state_dict_disc"] = accelerator.unwrap_model(net_disc).state_dict()
        ckpt_dict["disc_optimizer"] = disc_optimizer.state_dict()
        ckpt_dict["disc_lr_scheduler"] = disc_lr_scheduler.state_dict()

    # Save VAE/UNet keys: trainable OR non-base_layer OR mismatched
    vae_save_keys = {k for k in vae_raw_sd if k in vae_trainable_keys
                     or "base_layer" not in k
                     or k in vae_mismatched_keys}
    unet_save_keys = {k for k in unet_raw_sd if k in unet_trainable_keys
                      or "base_layer" not in k
                      or k in unet_mismatched_keys}

    ckpt_dict["state_dict_vae"]  = {k: v for k, v in vae_raw_sd.items()  if k in vae_save_keys}
    ckpt_dict["state_dict_unet"] = {k: v for k, v in unet_raw_sd.items() if k in unet_save_keys}

    save_checkpoint(ckpt_dict, checkpoint_dir=os.path.join(save_path, 'checkpoints'), checkpoint_name=filename)
    del ckpt_dict


# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 src/trainv1_glc.py --stage 1 --MASTER_PORT 12355 --experiment_name glc_stage1
if __name__ == "__main__":
    args         = parse_args_training()
    base_config  = load_yaml(os.path.join(args.config_dir, 'base.yaml'))
    train_config = load_yaml(os.path.join(args.config_dir, f"stage{args.stage}.yaml"))
    val_config   = load_yaml(os.path.join(args.config_dir, 'val.yaml'))
    main(args.stage, base_config, train_config, val_config, args.MASTER_PORT, args.experiment_name)
