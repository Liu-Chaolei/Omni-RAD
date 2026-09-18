import h5py
import torch
import argparse
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPVisionModelWithProjection

def parse_args_training(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--config_dir", type=str, default="./configs")
    parser.add_argument("--MASTER_PORT", type=int, default=12355)
    parser.add_argument("--experiment_name", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.transform = transform
        self.dataset = None
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f)
            self.keys = [
                key for key in f.keys() 
                if f[key].shape[0] >= 512 and f[key].shape[1] >= 512
            ]

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        key = self.keys[index]
        image = self.dataset[key][:]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.keys)
    

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