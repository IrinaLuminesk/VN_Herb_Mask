import torch
from torchvision.transforms.v2 import Transform
import torch.nn.functional as F

class FMix(Transform):
    def __init__(self, alpha, decay_power, number_of_class, max_soft=0.0, reformulate=False):
        super().__init__()
        self.decay_power = decay_power
        self.reformulate = reformulate
        self.alpha = alpha
        self.max_soft = max_soft
        self.number_of_class = number_of_class
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")
        params = self.make_params(images)
        mask = params["mask"]
        lam = params["lam"]
        mixed_image = self.transform_images(images=images, mask=mask)
        mixed_labels = self.transform_labels(labels=labels, lam=lam)
        return mixed_image, mixed_labels
    def fftfreqnd(self, device, h, w=None, z=None):
        fy = torch.fft.fftfreq(h, device=device)
        fx = fz = torch.tensor(0., device=device)

        if w is not None:
           
            fy = fy.unsqueeze(-1)
            fx = torch.fft.rfftfreq(w, device=device)

        if z is not None:
        
            fy = fy.unsqueeze(-1)
            fz = torch.fft.fftfreq(z, device=device).unsqueeze(-1)

        freqs = torch.sqrt(fx * fx + fy * fy + fz * fz)
        return freqs

    def get_spectrum(self, device, freqs, decay_power, ch, h, w=0, z=0):
    
        eps = 1. / max(h, w or 1, z or 1)
        scale = torch.clamp(freqs, min=eps) ** (-decay_power)
        scale = scale.to(device=device, dtype=torch.float32)

        # random complex params
        param = torch.randn((ch, *freqs.shape, 2), device=device, dtype=torch.float32)
        scale = scale.unsqueeze(0).unsqueeze(-1)  # (1, H, W[, Z], 1)
        return scale * param
    def make_low_freq_image(self, device, decay, shape, ch=1):
    
        h, w = shape
        freqs = self.fftfreqnd(device=device, h=h, w=w)
        spectrum = self.get_spectrum(device, freqs, decay, ch=ch, h=h, w=w)

        spectrum = spectrum[..., 0] + 1j * spectrum[..., 1]
        mask = torch.fft.irfftn(spectrum, s=shape, dim=tuple(range(1, spectrum.ndim))).real

        mask = mask - mask.min()
        mask = mask / mask.max()
        binary_mask = (mask > mask.mean()).float()
        return binary_mask
    def make_params(self, images: torch.Tensor):
        B, C, H, W = images.shape
        masks = []
        lams = []
        for _ in range(B):
            mask = self.make_low_freq_image(images.device, self.decay_power, (H, W))
            masks.append(mask)
            lam = mask.mean()
            lams.append(lam)
        masks = torch.stack(masks, dim=0)   # [B, 1, H, W]
        lams = torch.stack(lams, dim=0)
        return {
            "mask": masks,
            "lam": lams
        }
    def transform_images(self, images: torch.Tensor, mask: torch.Tensor):
        original_img = images
        donor_img = images.roll(1,0)
        mixed_img = original_img * mask + donor_img * (1 - mask)
        return mixed_img
    def transform_labels(self, labels: torch.Tensor, lam: torch.Tensor):
        donor_labels = labels.roll(1, 0)
        lam = lam.view(-1, 1)
        # 1D classification labels
        if labels.ndim == 1:
            if self.number_of_class is None:
                raise ValueError("num_classes must be provided for 1D labels")
            labels_onehot = F.one_hot(labels, num_classes=self.number_of_class).float()
            donor_labels_onehot = F.one_hot(donor_labels, num_classes=self.number_of_class).float()
            mixed_labels = lam * labels_onehot + (1 - lam) * donor_labels_onehot
            return mixed_labels
        return lam * labels + (1 - lam) * donor_labels