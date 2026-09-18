"""DINO backbone adapted from vision-aided-gan; see THIRD_PARTY_NOTICES.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_aided_loss.DiffAugment_pytorch import DiffAugment

class DINO(torch.nn.Module):

    def __init__(self, cv_type='adv'):
        super().__init__(
        )

        self.cv_type = cv_type
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.model.eval()
        self.model.requires_grad = False
        self.input_resolution = 224
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])

    def __call__(self, x):
        x = F.interpolate(x*0.5+0.5, size=(224, 224), mode='area')
        x = x - self.image_mean[:, None, None].to(x.device)
        x /= self.image_std[:, None, None].to(x.device)

        if 'conv_multi_level' in self.cv_type:
            x = self.model.get_intermediate_layers(x, n=8)
            x = [x[i] for i in [0, 4, -1]]
            x[0] = x[0][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[1] = x[1][:, 1:, :].permute(0, 2, 1).reshape(-1, 768, 14, 14)
            x[2] = x[2][:, 0, :]
        else:
            x = self.model(x)

        return x



class CVBackbone(nn.Module):
    def __init__(self, cv_type, output_type, diffaug=False, device='cpu'):
        super().__init__()
        if cv_type != 'dino' or output_type != 'conv_multi_level':
            raise ValueError('The paper release supports dino / conv_multi_level only')
        self.policy = 'color,translation,cutout' if diffaug else ''
        self.models = [
            DINO(cv_type='dino_conv_multi_level').requires_grad_(False).to(device)
        ]

    def forward(self, images):
        return [model(DiffAugment(images, policy=self.policy)) for model in self.models]
