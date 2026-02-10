from typing import Any
import torch
from torchvision.transforms.v2 import Transform, query_size
import torch.nn.functional as F
import math

class CutMixMask(Transform):
    def __init__(self, number_of_class, alpha) -> None:
        super().__init__()
        self.number_of_class = number_of_class
        self.apply = alpha
    def forward(self, images: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")
        params = self.make_params(images, masks)
        box = params["box"] #Tọa độ dùng để cắt ảnh
        lam = params["lam_adjusted"]
        mixed_image = self.transform_images(images=images, box=box)
        mixed_mask = self.transform_masks(masks=masks, box=box)
        mixed_labels = self.transform_labels(labels=labels, lam=lam)
        return mixed_image, mixed_mask, mixed_labels
    
    # Dùng để tạo ra tọa độ và lam 
    def make_params(self, images, masks):
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H_images, W_images = query_size(images)
        H_masks, W_masks = query_size(masks)

        if H_images != H_masks or W_images != W_masks:
            raise ValueError("Images and Masks must have the same dimension to be able to CutMix")

        r_x = torch.randint(W_images, size=(1,))
        r_y = torch.randint(H_images, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W_images)
        r_h_half = int(r * H_images)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        return {
            "box": box,
            "lam_adjusted": lam_adjusted
        }

    def apply_box(self, inpt, box):
        x1, y1, x2, y2 = box
        rolled = inpt.roll(1, 0)
        output = inpt.clone()
        output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

        return output  
    
    def transform_images(self, images, box):
        return self.apply_box(images, box)
    
    def transform_masks(self, masks, box):
        return self.apply_box(masks, box)
    
    def transform_labels(self, labels, lam):
        if labels.ndim == 1:
            label = F.one_hot(label, num_classes=self.number_of_class)  # type: ignore[arg-type]
        if not labels.dtype.is_floating_point:
            label = labels.float()
        return labels.roll(1, 0).mul_(1.0 - lam).add_(labels.mul(lam))