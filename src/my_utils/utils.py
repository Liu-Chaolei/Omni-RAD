import importlib

import torch
import torch.nn as nn
import numpy as np
from collections import abc

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
from typing import Dict
import os

import math
import logging
from torchvision import transforms
import lpips
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_from_config_sr(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print("Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


def DelfileList(path, filestarts='checkpoint_last'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(filestarts):
                os.remove(os.path.join(root, file))


def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.loss_fn_alex = lpips.LPIPS(net='vgg')
        self.loss_fn_alex.net.requires_grad_(False)

        self.k_C = clip_coefficient

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path).eval()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path).eval()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # mimic the image transform function of CLIP
        self.transform_for_clip = transforms.Compose([
            lambda x: x*255.0, 
            transforms.Resize(224), # do_resize
            transforms.CenterCrop(224), # do_center_crop
            lambda x: x*0.00392156862745098, # do_rescale
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # do_normalize
            ])

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def clip_loss(self, x_hat_01, target_01) :
        
        self.image_encoder = self.image_encoder.to(x_hat_01.device)
        self.text_encoder = self.text_encoder.to(x_hat_01.device)

        compressed_image = self.transform_for_clip(x_hat_01)
        target_image = self.transform_for_clip(target_01)

        compressed_image_embeddings = self.image_encoder(compressed_image).image_embeds
        target_image_embeddings = self.image_encoder(target_image).image_embeds

        compressed_image_embeddings = compressed_image_embeddings / compressed_image_embeddings.norm(p=2, dim=-1, keepdim=True)
        target_image_embeddings = target_image_embeddings / target_image_embeddings.norm(p=2, dim=-1, keepdim=True)

        clip_loss = torch.norm(compressed_image_embeddings - target_image_embeddings, p=2)

        return clip_loss

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = self.lmbda * sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["bpp"] = out["bpp_loss"] / self.lmbda
        out["y_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        x_hat_01 = ((output["x_hat"] + 1) / 2).clamp(0, 1)
        target_01 = ((target + 1) / 2).clamp(0, 1)
        out["mse_loss"] = self.k_M * self.mse(x_hat_01, target_01) * 255 ** 2

        out["compression_loss"] = out["bpp_loss"] + out["mse_loss"]

        if self.k_C > 0.0:
            out["clip_loss"] = self.k_C * self.clip_loss(x_hat_01, target_01)
            out["compression_loss"] +=  out["clip_loss"]
        if self.k_P > 0.0 :
            self.loss_fn_alex = self.loss_fn_alex.to(target.device)
            out["lpips"] = self.loss_fn_alex(output["x_hat"], target).mean()
            out["lpips_loss"] = self.k_P * out["lpips"]
            out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"] - out["bpp_loss"]

        return out


class RateDistortionLossv4(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.loss_fn_alex = lpips.LPIPS(net='vgg')
        self.loss_fn_alex.net.requires_grad_(False)

        self.k_C = clip_coefficient

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path).eval()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path).eval()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # mimic the image transform function of CLIP
        self.transform_for_clip = transforms.Compose([
            lambda x: x*255.0, 
            transforms.Resize(224), # do_resize
            transforms.CenterCrop(224), # do_center_crop
            lambda x: x*0.00392156862745098, # do_rescale
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # do_normalize
            ])

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def clip_loss(self, x_hat_01, target_01) :
        
        self.image_encoder = self.image_encoder.to(x_hat_01.device)
        self.text_encoder = self.text_encoder.to(x_hat_01.device)

        compressed_image = self.transform_for_clip(x_hat_01)
        target_image = self.transform_for_clip(target_01)

        compressed_image_embeddings = self.image_encoder(compressed_image).image_embeds
        target_image_embeddings = self.image_encoder(target_image).image_embeds

        compressed_image_embeddings = compressed_image_embeddings / compressed_image_embeddings.norm(p=2, dim=-1, keepdim=True)
        target_image_embeddings = target_image_embeddings / target_image_embeddings.norm(p=2, dim=-1, keepdim=True)

        clip_loss = torch.norm(compressed_image_embeddings - target_image_embeddings, p=2)

        return clip_loss

    def forward(self, x_hat, target):
        out = {}

        x_hat_01 = ((x_hat + 1) / 2).clamp(0, 1)
        target_01 = ((target + 1) / 2).clamp(0, 1)
        out["mse_loss"] = self.k_M * self.mse(x_hat_01, target_01) * 255 ** 2

        out["compression_loss"] = out["mse_loss"].clone()

        if self.k_C > 0.0:
            out["clip_loss"] = self.k_C * self.clip_loss(x_hat_01, target_01)
            out["compression_loss"] +=  out["clip_loss"]
        if self.k_P > 0.0 :
            self.loss_fn_alex = self.loss_fn_alex.to(target.device)
            out["lpips"] = self.loss_fn_alex(x_hat, target).mean()
            out["lpips_loss"] = self.k_P * out["lpips"]
            out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"]

        return out


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

class RateDistortionLoss_v3(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.requires_grad_(False)
        self.alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.alex_lpips.requires_grad_(False)
        self.net_clip = CLIPLoss(clip_model_name=clip_model_path)

        self.k_C = clip_coefficient

    def forward(self, output, target, mode='train'):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = self.lmbda * sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["bpp"] = out["bpp_loss"] / self.lmbda
        out["y_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )

        out["mse_loss"] = self.k_M * self.mse(output["x_hat"], target)

        out["compression_loss"] = out["bpp_loss"] + out["mse_loss"]

        if self.k_C > 0.0:
            self.net_clip = self.net_clip.to(target.device)
            out["clip_loss"] = self.k_C * self.net_clip(output["x_hat"], target)
            out["compression_loss"] +=  out["clip_loss"]
        if self.k_P > 0.0 :
            if mode == 'train':
                self.net_lpips = self.net_lpips.to(target.device)
                out["lpips"] = self.net_lpips(output["x_hat"], target).mean()
            elif mode == 'val':
                x_hat_01 = ((output["x_hat"] + 1) / 2).clamp(0, 1)
                target_01 = ((target + 1) / 2).clamp(0, 1)
                self.alex_lpips = self.alex_lpips.to(target.device)
                out["lpips"] = self.alex_lpips(x_hat_01, target_01).mean()
            out["lpips_loss"] = self.k_P * out["lpips"]
            out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"] - out["bpp_loss"]

        return out

class RateDistortionLoss_v6(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.loss_fn_alex = lpips.LPIPS(net='vgg')
        self.loss_fn_alex.net.requires_grad_(False)

        self.k_C = clip_coefficient

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path).eval()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_path).eval()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # mimic the image transform function of CLIP
        self.transform_for_clip = transforms.Compose([
            lambda x: x*255.0, 
            transforms.Resize(224), # do_resize
            transforms.CenterCrop(224), # do_center_crop
            lambda x: x*0.00392156862745098, # do_rescale
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # do_normalize
            ])

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def clip_loss(self, x_hat_01, target_01) :
        
        self.image_encoder = self.image_encoder.to(x_hat_01.device)
        self.text_encoder = self.text_encoder.to(x_hat_01.device)

        compressed_image = self.transform_for_clip(x_hat_01)
        target_image = self.transform_for_clip(target_01)

        compressed_image_embeddings = self.image_encoder(compressed_image).image_embeds
        target_image_embeddings = self.image_encoder(target_image).image_embeds

        compressed_image_embeddings = compressed_image_embeddings / compressed_image_embeddings.norm(p=2, dim=-1, keepdim=True)
        target_image_embeddings = target_image_embeddings / target_image_embeddings.norm(p=2, dim=-1, keepdim=True)

        clip_loss = torch.norm(compressed_image_embeddings - target_image_embeddings, p=2)

        return clip_loss

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = self.lmbda * sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["bpp"] = out["bpp_loss"] / self.lmbda
        out["y_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )

        x_hat_01 = ((output["x_hat"] + 1) / 2).clamp(0, 1)
        target_01 = ((target + 1) / 2).clamp(0, 1)
        out["mse_loss"] = self.k_M * self.mse(output["x_hat"], target)

        out["compression_loss"] = out["bpp_loss"] + out["mse_loss"]

        if self.k_C > 0.0:
            out["clip_loss"] = self.k_C * self.clip_loss(x_hat_01, target_01)
            out["compression_loss"] +=  out["clip_loss"]
        if self.k_P > 0.0 :
            self.loss_fn_alex = self.loss_fn_alex.to(target.device)
            out["lpips"] = self.loss_fn_alex(output["x_hat"], target).mean()
            out["lpips_loss"] = self.k_P * out["lpips"]
            out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"] - out["bpp_loss"]

        return out

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
        self.net_clip = CLIPLoss(clip_model_name=clip_model_path)

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

class RateDistortionLoss_fusion(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.5, clip_model_path = "openai/clip-vit-base-patch32", mse_coefficient=2.0, lpips_coefficient = 1.0, clip_coefficient=0.1, only_obj=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.k_M = mse_coefficient

        self.k_P = lpips_coefficient
        self.net_lpips = lpips.LPIPS(net='vgg')
        self.net_lpips.requires_grad_(False)
        self.alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.alex_lpips.requires_grad_(False)
        self.net_clip = CLIPLoss(clip_model_name=clip_model_path)

        self.k_C = clip_coefficient

        if only_obj:
            self.k_P = 0.0
            self.k_C = 0.0

    def forward(self, x_hat, target, mode='train'):
        out = {}

        out["mse_loss"] = self.k_M * self.mse(x_hat, target)

        out["compression_loss"] = out["mse_loss"].clone()

        # if self.k_C > 0.0:
        self.net_clip = self.net_clip.to(target.device)
        out["clip_loss"] = self.k_C * self.net_clip(x_hat, target)
        out["compression_loss"] +=  out["clip_loss"]
        # if self.k_P > 0.0 :
        if mode == 'train':
            self.net_lpips = self.net_lpips.to(target.device)
            out["lpips"] = self.net_lpips(x_hat, target).mean()
        elif mode == 'val':
            x_hat_01 = ((x_hat + 1) / 2).clamp(0, 1)
            target_01 = ((target + 1) / 2).clamp(0, 1)
            self.alex_lpips = self.alex_lpips.to(target.device)
            out["lpips"] = self.alex_lpips(x_hat_01, target_01).mean()
        out["lpips_loss"] = self.k_P * out["lpips"]
        out["compression_loss"] += out["lpips_loss"]

        out["distortion_loss"] = out["compression_loss"]

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


def configure_module_gradients(module, module_name, skipped_keys, stage):
    frozen_count = 0
    active_count = 0
    
    for name, param in module.named_parameters():
        param.requires_grad = False
        condition = False
        if stage == 1:
            condition = (name in skipped_keys) or (name.endswith(".quantiles"))
        elif stage == 2:
            condition = (name in skipped_keys) or (name.endswith(".quantiles")) or ("lora" in name)
        if condition:
            param.requires_grad = True
            active_count += 1
            print(f" -> Unlocked {module_name} layer: {name}")
        else:
            frozen_count += 1

    print(f"[{module_name}] Frozen layers: {frozen_count}, Active (Trainable) layers: {active_count}")
