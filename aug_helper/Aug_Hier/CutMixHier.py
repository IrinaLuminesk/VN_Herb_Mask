from typing import Any
import torch
from torchvision.transforms.v2 import Transform, query_size
import torch.nn.functional as F
import math

class CutMixHier(Transform):
    def __init__(self, number_of_class, alpha) -> None:
        super().__init__()
        self.number_of_class = number_of_class
        self.alpha = alpha
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")
        params = self.make_params(images)
        box = params["box"] #Tọa độ dùng để cắt ảnh
        lam = params["lam_adjusted"]
        mixed_image = self.transform_images(images=images, box=box)
        mixed_labels = self.transform_labels(labels=labels, lam=lam)

        return mixed_image, mixed_labels
    
    # Dùng để tạo ra tọa độ và lam 
    def make_params(self, images):
        lam = float(torch.distributions.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha])).sample())  # type: ignore[arg-type]

        H_images, W_images = query_size(images)

        r_x = torch.randint(W_images, size=(1,))
        r_y = torch.randint(H_images, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W_images)
        r_h_half = int(r * H_images)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W_images))
        y2 = int(torch.clamp(r_y + r_h_half, max=H_images))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W_images * H_images))
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

    
    def transform_labels(self, labels, lam):
        new_labels = dict()
        if labels.ndim == 2:
            for idx, key in enumerate(self.number_of_class.keys()):
                one_hot_labels = F.one_hot(labels[:, idx], num_classes=self.number_of_class[key])
                one_hot_labels = one_hot_labels.roll(1, 0).mul_(1.0 - lam).add_(one_hot_labels.mul(lam))
                new_labels[key] = one_hot_labels
        return new_labels