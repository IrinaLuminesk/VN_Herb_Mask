from cv2 import transform
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch

from utils.Utilities import get_num_workers

from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import tv_tensors

#Cái mới nhất
class ImageMaskFolder(Dataset):
    def __init__(self, img_root, mask_root, std, mean, img_size, data_type, transform = True):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.data_type = data_type
        self.transform = transform

        # Mimic ImageFolder indexing
        self.class_to_idx = self.Get_Class_idx()
        self.samples = self.Get_Imgs_Masks_sample()
    def align_img_mask_arrays(self, imgs, masks, fill=-1):
        lookup = {guid: val for val, guid in masks}
        return [lookup.get(guid, fill) for _, guid in imgs]

    def Get_Class_idx(self):
        classes = sorted([d.name for d in self.img_root.iterdir() if d.is_dir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}

        return class_to_idx
    def Get_Imgs_Masks_sample(self):
        samples = []
        imgs = [(img, img.stem) for img in self.img_root.rglob("*") if img.is_file()] #(Đường dẫn, Guid)
        masks = [(mask, mask.stem) for mask in self.mask_root.rglob("*") if mask.is_file] #(Đường dẫn, Guid)

        masks = self.align_img_mask_arrays(imgs, masks, -1)
        imgs = [i for i, _ in imgs]
        print("Found {0} images, {1} masks belong to {2} classes".format(len(imgs),
                                                                        len(masks),
                                                                        len(self.class_to_idx)))
        for img, mask in zip(imgs, masks):
            #Đường dẫn của ảnh và label
            class_name = img.parent.name #Tên label
            samples.append((img, mask, self.class_to_idx[class_name]))
            #phần này đang tạo ra một list chứa tuples có 3 value là đường dẫn, mask và label. Nếu mask = -1 nghĩa là ảnh đó không có mask
        return samples

    def train_transform(self):
        if self.transform:
            v2.Compose([
                v2.Resize(self.img_size),
                v2.RandomChoice([
                    v2.RandomResizedCrop(size=self.img_size),
                    v2.RandomHorizontalFlip(p=1),
                    v2.RandomVerticalFlip(p=1),
                    v2.Compose([
                        v2.Pad((10, 20)),
                        v2.Resize(self.img_size)
                    ]),
                    v2.Compose([
                        v2.RandomZoomOut(p=1, side_range=(1, 1.5)),
                        v2.Resize(self.img_size)
                    ]),
                    v2.RandomRotation(degrees=(-180, 180)),
                    v2.RandomAffine(degrees=(-180, 180), translate=(0.1, 0.3), scale=(0.5, 1.75)),
                    v2.RandomPerspective(p=1),
                    v2.ElasticTransform(alpha=120),
                    v2.ColorJitter(brightness=(1,2), contrast=(1,2)),
                    v2.RandomPhotometricDistort(brightness=(1,2), contrast=(1,2), p=1),
                    v2.RandomChannelPermutation(),
                    v2.RandomGrayscale(p=1),
                    v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 4.75)),
                    v2.RandomInvert(p=1),
                    v2.Lambda(lambda x: x),
                    ]),
                    v2.ToImage(), 
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=self.mean,
                        std=self.std
                    )
                ])
        return v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def test_transform(self):
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        # to_Tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        # img = to_Tensor(img)
        if mask_path != -1:
            mask = Image.open(mask_path).convert("L")  # binary
            mask = torch.from_numpy(np.array(mask))    # uint8 {0,255}
            mask = (mask > 0).float()
            has_mask = True
        else:
            width, height = img.size #Đảo ngược lại do Pil trả về W, H không phải H, W như cv2
            mask = torch.zeros((height, width), dtype=torch.float32)
            has_mask = False
        img  = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask)
        
        data_transform = self.train_transform() if self.data_type == "train" else self.test_transform()
        
        img, mask = data_transform(img, mask)

        return img, mask, label, has_mask


class DatasetLoader():
    def __init__(self, img_path, mask_path, std, mean, img_size, batch_size, transform = True) -> None:
        self.img_path = img_path
        self.mask_path = mask_path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform

    def dataset_loader(self, type):
        dataset = ImageMaskFolder(
            img_root=self.img_path,
            mask_root=self.mask_path,
            data_type=type,
            std=self.std,
            mean=self.mean,
            img_size=self.img_size,
            transform=self.transform
        )
        print("Total {0} image: {1}".format(type, len(dataset)))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,          # START HERE
            pin_memory=True,
            persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
            prefetch_factor=2
        )
        return loader
        