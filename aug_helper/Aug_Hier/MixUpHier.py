from typing import Any
import torch
from torchvision.transforms.v2 import Transform, query_size
import torch.nn.functional as F
import math

class MixUpHier(Transform):
    def __init__(self, number_of_class, alpha) -> None:
        super().__init__()
        self.number_of_class = number_of_class
        self.alpha = alpha
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")
        params = self.make_params()
        lam = params["lam"]
        mixed_image = self.transform_images(images=images, lam=lam)
        mixed_labels = self.transform_labels(labels=labels, lam=lam)
    

        return mixed_image, mixed_labels
    
    # Dùng để tạo ra tọa độ và lam 
    def make_params(self):
        lam = float(torch.distributions.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha])).sample())  # type: ignore[arg-type]

        return {
            "lam": lam
        }

    def apply_mix(self, inpt, lam):
        output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))
        return output  
    
    def transform_images(self, images, lam):
        return self.apply_mix(images, lam)
    
    def transform_labels(self, labels, lam):
        new_labels = dict()
        if labels.ndim == 2:
            for idx, key in enumerate(self.number_of_class.keys()):
                one_hot_labels = F.one_hot(labels[:, idx], num_classes=self.number_of_class[key])
                one_hot_labels = one_hot_labels.roll(1, 0).mul_(1.0 - lam).add_(one_hot_labels.mul(lam))
                new_labels[key] = one_hot_labels
        return new_labels
        # if labels.ndim == 1:
        #     labels = F.one_hot(labels, num_classes=self.number_of_class)  # type: ignore[arg-type]
        # if not labels.dtype.is_floating_point:
        #     labels = labels.float()
        # return labels.roll(1, 0).mul_(1.0 - lam).add_(labels.mul(lam))