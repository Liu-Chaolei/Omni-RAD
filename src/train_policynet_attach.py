"""
train_policynet_attach.py

Train only TimestepPolicyNet on top of an existing debug-step StableCodec
checkpoint. The StableCodec body is frozen; gradients update only
codec.timestep_policy.
"""

import math
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image, ImageFile
from torchvision import transforms

from my_utils.training_utils import H5Dataset, parse_args_training
from my_utils.utils import RateDistortionLoss_v7, logger_setup
from my_utils import wandb_utils
from StableCodec_variable2_step_policy_attach import StableCodec

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def unwrap_model(accelerator, model):
    options = [torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel]
    while isinstance(model, tuple(options)):
        model = model.module
    return model


def sample_lambda(B: int, lambda_min: float, lambda_max: float, device) -> torch.Tensor:
    log_min = math.log(lambda_min)
    log_max = math.log(lambda_max)
    return torch.exp(
        torch.empty(B, dtype=torch.float32, device=device).uniform_(log_min, log_max)
    )


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_checkpoint(state, checkpoint_dir="./checkpoint", checkpoint_name="checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, checkpoint_name))


class ConditionalPadding:
    def __init__(self, base_patch_size, padding_mode="reflect"):
        self.padding_mode = padding_mode
        self.base_patch_size = base_patch_size

    def __call__(self, img):
        ori_h, ori_w = img.shape[-2:]
        pad_h = (math.ceil(ori_h / self.base_patch_size[0])) * self.base_patch_size[0] - ori_h
        pad_w = (math.ceil(ori_w / self.base_patch_size[1])) * self.base_patch_size[1] - ori_w
        return F.pad(img, pad=(0, pad_w, 0, pad_h), mode=self.padding_mode)


class FlexibleImageFolder(torch.utils.data.Dataset):
    """Image folder that accepts either root/train/*.png or flat root/*.png."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root, transform=None, split="train"):
        root_path = os.path.abspath(root)
        split_path = os.path.join(root_path, split)
        if os.path.isdir(split_path):
            image_dir = split_path
        elif os.path.isdir(root_path):
            image_dir = root_path
        else:
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
            and os.path.splitext(name)[1].lower() in self.VALID_EXTENSIONS
        )
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {image_dir}")
        self.transform = transform
        print(f"[{split.upper()}] Loaded {len(self.samples)} images from {image_dir}")

    def __getitem__(self, index):
        image = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(image)
        return image

    def __len__(self):
        return len(self.samples)


def prepare_dataloader(base_config, train_config, val_config, micro_batch_size):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(train_config["patch_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    if val_config["crop_val"]:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomCrop(
                train_config["patch_size"], pad_if_needed=True, padding_mode="reflect"),
        ])
    else:
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            ConditionalPadding(base_config["base_patch_size"]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    train_dataset = H5Dataset(base_config["hdf5_dataset"], transform=train_transforms)
    val_dataset = FlexibleImageFolder(base_config["test_dataset"], transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=base_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_config["val_batch_size"],
        shuffle=False,
        num_workers=base_config["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(train_config, device, save_path, accelerator):
    if accelerator.is_main_process:
        structure_logger = logger_setup(
            log_file_name="structure.log",
            log_file_folder_name=os.path.join(save_path, "logs"),
        )
    else:
        structure_logger = None

    model_config = dict(train_config["model"])
    model_config["use_policy_delta"] = True
    policy_config = dict(model_config.get("policy_net", {}))
    policy_config.update(train_config.get("policy_net", {}))
    policy_config.setdefault("t_min", 800.0)
    policy_config.setdefault("t_max", 999.0)
    model_config["policy_net"] = policy_config
    net = StableCodec(
        sd_path=model_config["sd_path"],
        config=model_config,
        logger=structure_logger,
    ).to(device)
    net.set_policy_train_only()

    if train_config["enable_xformers_memory_efficient_attention"]:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available.")
    if train_config["gradient_checkpointing"]:
        net.unet.enable_gradient_checkpointing()
    if train_config["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    if accelerator.is_main_process:
        names = [n for n, _ in trainable]
        structure_logger.info(f"Trainable PolicyNet parameters ({len(names)}): {names}")
        bad = [n for n in names if "timestep_policy" not in n]
        if bad:
            raise RuntimeError(f"Non-PolicyNet parameters are trainable: {bad}")

    policy_config = train_config.get("policy_net", {})
    policy_lr = float(policy_config.get("lr", 1e-4))
    optimizer = optim.AdamW(
        [p for _, p in trainable],
        lr=policy_lr,
        betas=(0.9, 0.999),
        weight_decay=train_config["adam_weight_decay"],
        eps=train_config["adam_epsilon"],
    )
    num_steps = train_config["max_train_steps"] * accelerator.num_processes
    lr_scheduler = get_scheduler(
        train_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_config["lr_warmup_steps"] * accelerator.num_processes,
        num_training_steps=num_steps,
        num_cycles=train_config["lr_num_cycles"],
        power=train_config["lr_power"],
    )
    return net, optimizer, lr_scheduler


def log_validation(step, val_loader, model, criterion, iqa_psnr,
                   save_val, save_steps, save_path, train_config, accelerator):
    model.eval()
    unwrap_model(accelerator, model).set_policy_train_only()
    unwrap_model(accelerator, model).codec.timestep_policy.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp = AverageMeter()
    psnr = AverageMeter()
    lpips = AverageMeter()
    t_snr = AverageMeter()
    t_pred = AverageMeter()
    delta_t = AverageMeter()
    delta_t_reg = AverageMeter()
    lambda_ref = float(train_config["model"].get("lambda_ref", train_config.get("lmbda", 1.0)))
    delta_reg_weight = float(train_config.get("policy_delta_reg_weight", 0.01))

    with torch.no_grad():
        validx = 1
        for d in val_loader:
            d = d.to(device)
            B, _, H, W = d.shape
            lmbda = torch.full((B,), lambda_ref, dtype=torch.float32, device=device)
            x_hat, rate_out, _, policy_info = model(
                d, [1 for _ in range(B)], H, W, lmbda=lmbda, use_policy_delta=True)
            out = criterion(x_hat, d)
            per_image_dist = out["per_image_distortion"]
            cur_distortion_loss = (per_image_dist / lmbda).mean()
            policy_module = unwrap_model(accelerator, model).codec.timestep_policy
            delta = policy_info["T_pred"] - policy_info["T_snr"]
            delta_max = float(getattr(policy_module, "delta_max", 30.0))
            cur_delta_reg = (delta / max(delta_max, 1e-6)).pow(2).mean()
            cur_loss = cur_distortion_loss + delta_reg_weight * cur_delta_reg

            loss.update(cur_loss.item())
            bpp.update(rate_out.quantized_total_bpp.item())
            lpips.update(out["lpips"].item())
            x_hat_01 = ((x_hat + 1) / 2).clamp(0, 1)
            d_01 = ((d + 1) / 2).clamp(0, 1)
            psnr.update(iqa_psnr(x_hat_01, d_01).mean().item())
            t_snr.update(policy_info["T_snr"].mean().item())
            t_pred.update(policy_info["T_pred"].mean().item())
            delta_t.update(delta.mean().item())
            delta_t_reg.update(cur_delta_reg.item())

            if save_val and accelerator.is_main_process and step % save_steps == 0 and validx <= 5:
                out_img = (x_hat * 0.5 + 0.5).float().cpu().detach()
                out_pil = transforms.ToPILImage()(out_img[0].clamp(0, 1))
                os.makedirs(os.path.join(save_path, "valpics"), exist_ok=True)
                out_pil.save(os.path.join(save_path, "valpics", f"step{step}_preview_{validx}.png"))
                validx += 1

    metrics = torch.tensor(
        [loss.sum, bpp.sum, psnr.sum, lpips.sum, t_snr.sum, t_pred.sum,
         delta_t.sum, delta_t_reg.sum, loss.count],
        device=device,
    )
    accelerator.reduce(metrics, reduction="sum")
    count = metrics[-1].item()
    keys = ["loss", "bpp", "psnr", "lpips", "t_snr", "t_pred", "delta_t", "delta_t_reg"]
    if count <= 0:
        return {k: 0.0 for k in keys}
    return {k: (metrics[i] / metrics[-1]).item() for i, k in enumerate(keys)}


def main(stage, base_config, train_config, val_config, MASTER_PORT, experiment_name):
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    accelerator = Accelerator(
        mixed_precision=train_config["mixed_precision"],
        log_with=train_config["report_to"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
    )
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    if base_config["global_seed"] is not None:
        seed = base_config["global_seed"] * world_size + rank
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(train_config["savepath"], f"stage{stage}", experiment_name)
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        train_logger = logger_setup("train.log", os.path.join(save_path, "logs"))
        val_logger = logger_setup("val.log", os.path.join(save_path, "logs"))
        shutil.copytree("configs", os.path.join(save_path, "configs"))
        if train_config["wandb"]:
            entity = os.environ.get("ENTITY")
            project = os.environ.get("PROJECT", "sc")
            wandb_utils.initialize(entity, experiment_name, project)

    gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
    global_batch_size = train_config["global_batch_size"]
    if global_batch_size % (world_size * gradient_accumulation_steps) != 0:
        raise ValueError("global_batch_size must be divisible by world_size * gradient_accumulation_steps.")
    micro_batch_size = global_batch_size // (world_size * gradient_accumulation_steps)

    train_loader, val_loader = prepare_dataloader(base_config, train_config, val_config, micro_batch_size)
    net, optimizer, lr_scheduler = build_model(train_config, accelerator.device, save_path, accelerator)
    net, optimizer, train_loader, val_loader = accelerator.prepare(net, optimizer, train_loader, val_loader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    net.to(accelerator.device, dtype=weight_dtype)

    lambda_ref = float(train_config["model"].get("lambda_ref", train_config.get("lmbda", 1.0)))
    lambda_min = float(train_config["model"].get("lambda_min", 0.1))
    lambda_max = float(train_config["model"].get("lambda_max", 128.0))
    lambda_sample_max = float(train_config["model"].get("lambda_sample_max", lambda_max))

    criterion = RateDistortionLoss_v7(
        lmbda=lambda_ref,
        clip_model_path=train_config["clip_model_path"],
        mse_coefficient=train_config["mse_coefficient"],
        lpips_coefficient=train_config["lpips_coefficient"],
        clip_coefficient=train_config["clip_coefficient"],
    )
    iqa_psnr = pyiqa.create_metric("psnr", device=accelerator.device)

    metric_names = [
        "Train/loss", "Train/mse_loss", "Train/lpips", "Train/bpp",
        "Train/y_bpp", "Train/z_bpp", "Train/T_snr_mean",
        "Train/T_pred_mean", "Train/delta_T_mean", "Train/delta_T_std",
        "Train/delta_T_reg",
    ]
    running = {k: 0.0 for k in metric_names}
    log_steps = 0
    global_step = 0
    epoch = 0
    start_time = time.time()
    best_loss = float("inf")

    while global_step < train_config["max_train_steps"]:
        unwrap_model(accelerator, net).set_policy_train_only()
        for d in train_loader:
            if global_step >= train_config["max_train_steps"]:
                break
            with accelerator.accumulate(net):
                unwrap_model(accelerator, net).set_policy_train_only()
                d = d.to(accelerator.device)
                B, _, H, W = d.shape
                lmbda = sample_lambda(B, lambda_min, lambda_sample_max, d.device)
                x_hat, rate_out, _, policy_info = net(
                    d, [1 for _ in range(B)],
                    train_config["patch_size"][0],
                    train_config["patch_size"][1],
                    lmbda=lmbda,
                    use_policy_delta=True,
                )
                d = d.detach().float()
                out = criterion(x_hat, d)
                per_image_dist = out["per_image_distortion"]
                distortion_loss = (per_image_dist / lmbda).mean()
                policy_module = unwrap_model(accelerator, net).codec.timestep_policy
                delta_t = policy_info["T_pred"] - policy_info["T_snr"]
                delta_max = float(getattr(policy_module, "delta_max", 30.0))
                delta_reg_weight = float(train_config.get("policy_delta_reg_weight", 0.01))
                delta_t_reg = (delta_t / max(delta_max, 1e-6)).pow(2).mean()
                policy_loss = distortion_loss + delta_reg_weight * delta_t_reg

                accelerator.backward(policy_loss)
                if accelerator.sync_gradients and train_config["clip_max_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        unwrap_model(accelerator, net).codec.timestep_policy.parameters(),
                        train_config["clip_max_norm"],
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=train_config["set_grads_to_none"])

                current = {
                    "Train/loss": policy_loss.item(),
                    "Train/mse_loss": out["mse_loss"].item(),
                    "Train/lpips": out["lpips"].item(),
                    "Train/bpp": rate_out.quantized_total_bpp.detach().item(),
                    "Train/y_bpp": rate_out.quantized_latent_bpp.detach().item(),
                    "Train/z_bpp": rate_out.quantized_hyper_bpp.detach().item(),
                    "Train/T_snr_mean": policy_info["T_snr"].mean().item(),
                    "Train/T_pred_mean": policy_info["T_pred"].mean().item(),
                    "Train/delta_T_mean": delta_t.mean().item(),
                    "Train/delta_T_std": delta_t.std().item() if delta_t.numel() > 1 else 0.0,
                    "Train/delta_T_reg": delta_t_reg.item(),
                }

            if accelerator.sync_gradients:
                global_step += 1
                log_steps += 1
                for k in metric_names:
                    running[k] += current[k]

                if global_step < 20000 or global_step % train_config["log_every"] == 0:
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    avg = {k: running[k] / max(log_steps, 1) for k in metric_names}
                    metrics_tensor = torch.tensor([avg[k] for k in metric_names], device=d.device)
                    accelerator.reduce(metrics_tensor, reduction="sum")
                    metrics_tensor /= world_size
                    avg = {k: v.item() for k, v in zip(metric_names, metrics_tensor)}
                    if accelerator.is_main_process:
                        msg = "step:{} ".format(global_step) + \
                            ", ".join([f"{k}:{avg[k]:.4f}" for k in metric_names]) + \
                            f", Steps/Sec:{log_steps / max(elapsed, 1e-6):.2f}"
                        train_logger.info(msg)
                    if train_config["wandb"]:
                        wandb_utils.log({f"train {k}": v for k, v in avg.items()}, step=global_step)
                    running = {k: 0.0 for k in metric_names}
                    log_steps = 0
                    start_time = time.time()

                if global_step % train_config["checkpointing_steps"] == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        _save_policy(net, optimizer, lr_scheduler, epoch, global_step, save_path, train_config, accelerator,
                                     f"policynet-attach-{global_step}.pth")

                if (global_step % val_config["validation_steps"] == 0
                        or (val_config["val_first"] and global_step == 1)):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    val_metrics = log_validation(
                        global_step, val_loader, net, criterion, iqa_psnr,
                        val_config["save_val"], val_config["save_steps"],
                        save_path, train_config, accelerator,
                    )
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        val_logger.info(
                            f"step: {global_step}, "
                            f"val/loss: {val_metrics['loss']:.4f}, "
                            f"val/lpips: {val_metrics['lpips']:.3f}, "
                            f"val/psnr: {val_metrics['psnr']:.2f}, "
                            f"val/bpp: {val_metrics['bpp']:.4f}, "
                            f"val/T_snr: {val_metrics['t_snr']:.1f}, "
                            f"val/T_pred: {val_metrics['t_pred']:.1f}, "
                            f"val/delta_T: {val_metrics['delta_t']:.1f}, "
                            f"val/delta_T_reg: {val_metrics['delta_t_reg']:.4f}"
                        )
                        if train_config["wandb"]:
                            wandb_utils.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                        if val_metrics["loss"] <= best_loss:
                            best_loss = val_metrics["loss"]
                            _save_policy(net, optimizer, lr_scheduler, epoch, global_step, save_path, train_config,
                                         accelerator, "policynet_attach_best.pth")
                    start_time = time.time()
        epoch += 1

    if accelerator.is_main_process:
        _save_policy(net, optimizer, lr_scheduler, epoch, global_step, save_path, train_config,
                     accelerator, "policynet_attach.pth")
    accelerator.end_training()


def _save_policy(net, optimizer, lr_scheduler, epoch, global_step, save_path, train_config, accelerator, filename):
    unwrapped = unwrap_model(accelerator, net)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict_policy_net": unwrapped.codec.timestep_policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "source_codec_path": train_config["model"].get("codec_path"),
        "policy_config": train_config.get("policy_net", {}),
        "policy_delta_reg_weight": train_config.get("policy_delta_reg_weight", 0.01),
    }
    save_checkpoint(payload, checkpoint_dir=os.path.join(save_path, "checkpoints"), checkpoint_name=filename)


# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 src/train_policynet_attach.py --stage 2 --MASTER_PORT 12356 --experiment_name attach_policy
if __name__ == "__main__":
    args = parse_args_training()
    base_config = load_yaml(os.path.join(args.config_dir, "base.yaml"))
    train_config = load_yaml(os.path.join(args.config_dir, f"stage{args.stage}.yaml"))
    val_config = load_yaml(os.path.join(args.config_dir, "val.yaml"))
    main(args.stage, base_config, train_config, val_config, args.MASTER_PORT, args.experiment_name)
