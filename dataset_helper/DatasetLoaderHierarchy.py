from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import torch

from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd


class HierarchialDataloader(Dataset):
    def __init__(self, img_root, hierarchy_label_root, hierarchy_columns, std, mean, img_size, data_type, transform = True):
        self.img_root = Path(img_root)
        self.hierarchy_label_root = hierarchy_label_root
        self.hierarchy_columns = hierarchy_columns 
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.data_type = data_type
        self.transform = transform
        self.data_transform = self.train_transform() if self.data_type == "train" else self.test_transform()

        self.class_to_idx = self.Get_Class_idx()
        self.samples = self.Get_Samples()
    def Get_Class_idx(self):
        data = pd.read_csv(self.hierarchy_label_root)
        new_col_ids = []
        for col in self.hierarchy_columns:
            col_id = "{0}_id".format(col)
            new_col_ids.append(col_id)
            data[col_id] = data[col].astype("category").cat.codes #Dùng để đổi các field từ text sang integer
        class_to_idx = defaultdict(list)
        
        for _, row in data.iterrows():
            for col_id in new_col_ids:
                class_to_idx[row["Original_Class"]].append(row[col_id])
        return class_to_idx
    
    def Get_Samples(self):
        #Tạo ra samples bao gồm đường dẫn tới ảnh và label, 
        # Mỗi label là 1 array chứa các label của mỗi cấp từ thấp (Species) tới cao (Phylum)
        image_paths = []
        labels = []
        for img in self.img_root.rglob("*"):
            if img.is_file():
                image_paths.append(img)
                labels.append(self.class_to_idx[img.parent.name])
        samples = list(zip(image_paths, labels)) #Biến 2 arrays thành tuple dạng pairing element-wise
        arr = np.array(labels)
        num_of_class_sample_per_hierarchy = [len(np.unique(arr[:, i])) for i in range(arr.shape[1])] 
        text = []
        for hier, num in zip(self.hierarchy_columns, num_of_class_sample_per_hierarchy):
            text.append("{0} {1}".format(str(num), hier))
        print("Found {0} images belong to {1}".format(len(samples),", ".join(text)))
        return samples
    
    def train_transform(self):
        if self.transform:
            return v2.Compose([
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
            v2.Resize(self.img_size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def test_transform(self):
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        image_path, label = self.samples[index]
        
        img = Image.open(image_path).convert("RGB")
        img  = tv_tensors.Image(img)
        img = self.data_transform(img)
        return img, label


class DatasetLoader():
    def __init__(self, img_path, hierarchy_label_root, hierarchy_columns, std, mean, img_size, batch_size, transform = True) -> None:
        self.img_path = img_path
        self.hierarchy_label_root = hierarchy_label_root
        self.hierarchy_columns = hierarchy_columns
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform

    def dataset_loader(self, type):
        if type == "train":
            train_dataset = HierarchialDataloader(
                img_root=self.img_path,
                hierarchy_label_root=self.hierarchy_label_root,
                hierarchy_columns=self.hierarchy_columns,
                data_type=type,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                transform=self.transform
            )
            # print("Total train image: {0}, train mask: {1}".format(len(train_dataset), len(train_dataset)))
            loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
                prefetch_factor=2
            )
        else:
            test_dataset = HierarchialDataloader(
                img_root=self.img_path,
                hierarchy_label_root=self.hierarchy_label_root,
                hierarchy_columns=self.hierarchy_columns,
                data_type=type,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                transform=self.transform
            )
            # print("Total test image: {0}, train mask: {1}".format(len(test_dataset), len(test_dataset)))
            loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
            )
        return loader