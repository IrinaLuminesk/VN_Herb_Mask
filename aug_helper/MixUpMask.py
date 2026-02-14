from typing import Any
import torch
from torchvision.transforms.v2 import Transform, query_size
import torch.nn.functional as F
import math

class MixUpMask(Transform):
    def __init__(self, number_of_class, alpha) -> None:
        super().__init__()
        self.number_of_class = number_of_class
        self.apply = alpha
    def forward(self, images: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")
        params = self.make_params()
        lam = params["lam"]
        mixed_image = self.transform_images(images=images, lam=lam)
        mixed_mask = self.transform_masks(masks=masks)
        mixed_labels = self.transform_labels(labels=labels, lam=lam)
        
        has_masks = mixed_mask.any(dim=(1, 2, 3))

        return mixed_image, mixed_mask, mixed_labels, has_masks
    
    # Dùng để tạo ra tọa độ và lam 
    def make_params(self):
        lam = float(torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample())  # type: ignore[arg-type]

        return {
            "lam": lam
        }

    def apply_mix(self, inpt, lam):
        output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))
        return output  
    
    def transform_images(self, images, lam):
        return self.apply_mix(images, lam)
    
    def transform_masks(self, masks):
        original_mask = masks
        donor_mask = masks.roll(1, 0)
        union_masks = torch.maximum(original_mask, donor_mask)
        return union_masks
    
    def transform_labels(self, labels, lam):
        if labels.ndim == 1:
            labels = F.one_hot(labels, num_classes=self.number_of_class)  # type: ignore[arg-type]
        if not labels.dtype.is_floating_point:
            labels = labels.float()
        return labels.roll(1, 0).mul_(1.0 - lam).add_(labels.mul(lam))