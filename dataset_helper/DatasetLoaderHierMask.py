from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import torch

from Utilities_class import ApplyToBoth, ApplyToImageOnly

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

class HierarchialMaskDataloader(Dataset):
    def __init__(self, img_root, mask_root, hierarchy_label_root, hierarchy_columns, std, mean, img_size, data_type, transform = True):
        super().__init__()
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.hierarchy_label_root = hierarchy_label_root
        self.hierarchy_columns = hierarchy_columns 
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.data_type = data_type
        self.transform = transform
        self.data_transform = self.train_transform() if self.data_type == "train" else self.test_transform()

        self.class_to_idx = self.Get_Class_idx()
        self.samples, self.num_classes = self.Get_Samples()
    def Get_Class_idx(self):
        data = pd.read_csv(self.hierarchy_label_root)
        new_col_ids = []
        for col in self.hierarchy_columns:
            col_id = "{0}_id".format(col)
            new_col_ids.append(col_id)
            data[col_id] = data[col].astype("category").cat.codes #Dùng để đổi các field từ text sang integer
        
        class_to_idx = (
            data.assign(Original_Class=data["Original_Class"].astype(str)) #Để không bị lỗi type
            .set_index("Original_Class")[new_col_ids] #Lấy Original Class làm index 
            .apply(list, axis=1) #Biến từng dòng một thành list
            .to_dict() #Chuyển thành dict
        ) #Cách này nhanh hơn
        return class_to_idx
    
    def Get_Samples(self):
        #Tạo ra samples bao gồm đường dẫn tới ảnh và label, 
        # Mỗi label là 1 array chứa các label của mỗi cấp từ thấp (Species) tới cao (Phylum)
        # hoặc từ cao (Phylum) tới thấp (Species) tùy theo array hierarchy_columns
        image_paths = []
        mask_paths = []
        labels = []
        for img in self.img_root.rglob("*"):
            if img.is_file():
                image_paths.append(img)
                labels.append(self.class_to_idx[img.parent.name])

                relative_path = img.relative_to(self.img_root) 

                mask_path = (self.mask_root / relative_path).with_suffix(".png")

                mask_paths.append(mask_path if mask_path.exists() else -1)
                #Lấy đường dẫn tương đối đến file đó, bỏ qua folder lớn nhất
        samples = list(zip(image_paths, mask_paths, labels)) #Biến 2 arrays thành tuple dạng pairing element-wise
        arr = np.array(labels)
        num_of_class_sample_per_hierarchy = [len(np.unique(arr[:, i])) for i in range(arr.shape[1])] 
        text = []
        num_classes = dict()
        for hier, num in zip(self.hierarchy_columns, num_of_class_sample_per_hierarchy):
            num_classes[hier] = num #Tạo num_classes để về sau không cần tạo thủ công
            text.append("{0} {1}".format(str(num), hier))
        num_masks = sum(path != -1 for path in mask_paths)
        print("Found {0} images and {1} masks belong to {2}".format(len(samples), num_masks, ", ".join(text)))
        return samples, num_classes
    
    def create_zeros_mask(self, height, width):
        return torch.zeros((height, width), dtype=torch.float32)
    
    def train_transform(self):
        if self.transform:
            return v2.Compose([
                v2.Resize(self.img_size),
                v2.RandomChoice([
                    ApplyToBoth(v2.RandomResizedCrop(size=self.img_size)),
                    ApplyToBoth(v2.RandomHorizontalFlip(p=1)),
                    ApplyToBoth(v2.RandomVerticalFlip(p=1)),
                    ApplyToBoth(v2.Compose([
                        v2.Pad((10, 20)),
                        v2.Resize(self.img_size)
                    ])),
                    ApplyToBoth(v2.Compose([
                        v2.RandomZoomOut(p=1, side_range=(1, 1.5)),
                        v2.Resize(self.img_size)
                    ])),
                    ApplyToBoth(v2.RandomRotation(degrees=(-180, 180))),
                    ApplyToBoth(v2.RandomAffine(degrees=(-180, 180), translate=(0.1, 0.3), scale=(0.5, 1.75))),
                    ApplyToBoth(v2.RandomPerspective(p=1)),
                    ApplyToBoth(v2.ElasticTransform(alpha=120)),
                    ApplyToImageOnly(v2.ColorJitter(brightness=(1,2), contrast=(1,2))),
                    ApplyToImageOnly(v2.RandomPhotometricDistort(brightness=(1,2), contrast=(1,2), p=1)),
                    ApplyToImageOnly(v2.RandomChannelPermutation()),
                    ApplyToImageOnly(v2.RandomGrayscale(p=1)),
                    ApplyToImageOnly(v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 4.75))),
                    ApplyToImageOnly(v2.RandomInvert(p=1)),
                    v2.Lambda(lambda x: x),
                    ]),
                    v2.ToImage(), 
                    v2.ToDtype(torch.float32, scale=True),
                    ApplyToImageOnly(v2.Normalize(
                        mean=self.mean,
                        std=self.std
                    ))
                ])
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            ApplyToImageOnly(v2.Normalize(
                mean=self.mean,
                std=self.std
            ))
        ])
    def test_transform(self):
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            ApplyToImageOnly(v2.Normalize(
                mean=self.mean,
                std=self.std
            ))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        label = torch.tensor(label, dtype=torch.long)
        
        img = Image.open(img_path).convert("RGB")
        width, height = img.size #Đảo ngược lại do Pil trả về W, H không phải H, W như cv2
        if mask_path != -1:
            mask = Image.open(mask_path).convert("L")  # binary
            mask = torch.from_numpy(np.array(mask))    # uint8 {0,255}
            mask = (mask > 0).float()
            mask = F.max_pool2d(
                    mask.unsqueeze(0).unsqueeze(0),
                    kernel_size=15,
                    stride=1,
                    padding=7
            ).squeeze(0).squeeze(0)
            has_mask = True 
            if not mask.any(): #Trường hợp có mask nhưng mask không có gì
                has_mask = False
        else:
            mask = self.create_zeros_mask(height, width)
            has_mask = False
        img  = tv_tensors.Image(img)
        mask = mask.unsqueeze(0)
        mask = tv_tensors.Mask(mask)
        
        # data_transform = self.train_transform() if self.data_type == "train" else self.test_transform()
        
        img, mask = self.data_transform(img, mask)

        return img, mask, label, has_mask
    
class DatasetLoader():
    def __init__(self, img_path, mask_path, hierarchy_label_root, hierarchy_columns, std, mean, img_size, batch_size, transform = True) -> None:
        self.img_path = img_path
        self.mask_path = mask_path
        self.hierarchy_label_root = hierarchy_label_root
        self.hierarchy_columns = hierarchy_columns
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform

    def dataset_loader(self, type):
        if type == "train":
            train_dataset = HierarchialMaskDataloader(
                img_root=self.img_path,
                mask_root=self.mask_path,
                hierarchy_label_root=self.hierarchy_label_root,
                hierarchy_columns=self.hierarchy_columns,
                data_type=type,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                transform=self.transform
            )
            self.num_classes = train_dataset.num_classes
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
            test_dataset = HierarchialMaskDataloader(
                img_root=self.img_path,
                mask_root=self.mask_path,
                hierarchy_label_root=self.hierarchy_label_root,
                hierarchy_columns=self.hierarchy_columns,
                data_type=type,
                std=self.std,
                mean=self.mean,
                img_size=self.img_size,
                transform=self.transform
            )
            self.num_classes = test_dataset.num_classes
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

    def Create_Consistent_Matrix(self, device):
        keys = list(self.num_classes.keys())
        keys.reverse()
        matrix_names = []
        hier_matrixs = dict()
        for i in range(len(keys) - 1):
            matrix_name = "{0}2{1}".format(keys[i], keys[i + 1])
            matrix_names.append(matrix_name)
            hier_matrixs[matrix_name] = self.Create_Matrix(keys[i], keys[i + 1]).to(device)
        return matrix_names, hier_matrixs


    #Tạo ma trận để tính loss, hiện tại chưa tìm ra cách tạo ma trận động nên đang làm thủ công
    def Create_Matrix(self, x, y):
        data = pd.read_csv(self.hierarchy_label_root)
        x_cat = data[x].astype("category")
        y_cat = data[y].astype("category")

        x2idx = {
            cat: idx
            for idx, cat in enumerate(x_cat.cat.categories)
        }
        y2idx = {
            cat: idx
            for idx, cat in enumerate(y_cat.cat.categories)
        }

        H_xy = torch.zeros(len(x2idx),len(y2idx))

        xy_pairs = data[[x, y]].drop_duplicates()
        
        for _, row in xy_pairs.iterrows():
            x_idx = x2idx[row[x]]
            y_idx = y2idx[row[y]]
            H_xy[x_idx, y_idx] = 1
        return H_xy


        # x_to_y_dict = self.Create_x_to_y_dict(data, x, y)
        # x_idx = {g:i for i, g in enumerate(data[x].unique())}
        # y_idx = {g:i for i, g in enumerate(data[y].unique())}
        # x_to_y_idx = {
        #     x_idx[x]: y_idx[y]
        #     for x, y in x_to_y_dict.items()
        # }
        # H_yx = torch.zeros(len(y_idx), len(x_idx))

        # # fill y → x
        # for x, y in x_to_y_idx.items():
        #     H_yx[y, x] = 1
        # return H_yx
    def Create_x_to_y_dict(self, data, x, y):
        return (
            data[[x, y]]
            .drop_duplicates()
            .set_index(x)[y]
            .to_dict()
        )