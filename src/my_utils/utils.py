"""Loss and logging helpers used by Omni-RAD training."""
import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import lpips

class CLIPLoss(torch.nn.Module):

    def __init__(self, clip_model_name = "openai/clip-vit-base-patch32"):
        super().__init__()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder.requires_grad_(False)

        self.transform_for_clip = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2.0),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def forward(self, rec, gt):

        rec_inputs = self.transform_for_clip(rec)
        gt_inputs = self.transform_for_clip(gt)

        rec_features = self.image_encoder(rec_inputs).image_embeds
        gt_features = self.image_encoder(gt_inputs).image_embeds

        rec_features = rec_features / rec_features.norm(p=2, dim=-1, keepdim=True)
        gt_features = gt_features / gt_features.norm(p=2, dim=-1, keepdim=True)

        loss = torch.norm(gt_features - rec_features, p=2, dim=-1).mean()
        return loss


class RateDistortionLoss_v7(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1, stage=1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.requires_grad_(False)
        self.alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.alex_lpips.requires_grad_(False)
        self.net_clip = CLIPLoss(clip_model_name=clip_model_path) if clip_coefficient > 0 else None

        self.k_C = clip_coefficient

        if stage == 3:
            self.k_P = 0.0
            self.k_C = 0.0

    def forward(self, x_hat, target, mode='train'):
        out = {}

        out["mse_loss"] = self.k_M * self.mse(x_hat, target)

        out["compression_loss"] = out["mse_loss"].clone()

        # Per-image MSE: [B] — for variable-rate per-image λ weighting
        B = x_hat.shape[0]
        per_image_mse = self.k_M * (x_hat - target).pow(2).reshape(B, -1).mean(dim=1)
        per_image_distortion = per_image_mse  # [B]

        if self.k_C > 0.0:
            self.net_clip = self.net_clip.to(target.device)
            out["clip_loss"] = self.k_C * self.net_clip(x_hat, target)
            out["compression_loss"] +=  out["clip_loss"]
        if self.k_P > 0.0 :
            if mode == 'train':
                self.net_lpips = self.net_lpips.to(target.device)
                lpips_raw = self.net_lpips(x_hat, target)  # [B, 1, 1, 1]
                out["lpips"] = lpips_raw.mean()
                per_image_distortion = per_image_distortion + self.k_P * lpips_raw.reshape(B)
            elif mode == 'val':
                x_hat_01 = ((x_hat + 1) / 2).clamp(0, 1)
                target_01 = ((target + 1) / 2).clamp(0, 1)
                self.alex_lpips = self.alex_lpips.to(target.device)
                out["lpips"] = self.alex_lpips(x_hat_01, target_01).mean()
            out["lpips_loss"] = self.k_P * out["lpips"]
            out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"]
        out["per_image_distortion"] = per_image_distortion  # [B]

        return out


def logger_setup(log_file_name=None, log_file_folder_name = None, filepath=os.path.abspath(__file__), package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(log_file_name)
    logger.setLevel('INFO'.upper())

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setLevel('INFO'.upper())
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        os.makedirs(log_file_folder_name, exist_ok=True)
        info_file_handler = logging.FileHandler(log_file_folder_name + '/' + log_file_name , mode="a")
        info_file_handler.setLevel('INFO'.upper())
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger


def get_raw_state_dict(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    else:
        return model.state_dict()
