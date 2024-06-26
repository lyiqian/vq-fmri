import torch 
from torch.utils.data import Dataset, Sampler
import pandas as pd 
from pathlib import Path
import numpy as np 
import bdpy
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

class GODDataset(Dataset):
    def __init__(self, data_dir, image_transforms, split='training'):
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.split = split
        if split == 'training':
            df_fname = 'image_training_id.csv'
            data_type = 1
        else:
            self.split = 'test'
            df_fname = 'image_test_id.csv'
            data_type = 2

        df_path = data_dir / 'images' / df_fname
        self.img_ids = pd.read_csv(df_path, header=None)

        # subject_files = [p for p in data_dir.glob("*.h5")]
        subject_files = [p for p in data_dir.glob("Subject1.h5")]
        fmri_data_stack = []
        image_id_stack = []
        for sf in subject_files:
            data = bdpy.BData(sf)
            fmri_data = np.array(data.select('ROI*'))
            dt = np.array(data.select('DataType'))
            # in the GOD dataset, data_type denotes
            # either test set or train set
            split_indexes = dt == data_type
            fmri_data = fmri_data[split_indexes[:, 0], :]
            fmri_data_stack.append(fmri_data)
            image_ids = np.array(data.select('Label'))[:, 0]
            split_image_ids = image_ids[split_indexes[:, 0]]
            image_id_stack.append(split_image_ids)

        # Determine the maximum number of rows and columns
        max_rows = max(array.shape[0] for array in fmri_data_stack)
        max_cols = max(array.shape[1] for array in fmri_data_stack)
        # print(max_cols)
        # Pad arrays with zeros to have the same shape
        padded_fmri_data = []
        for fmri_data in fmri_data_stack:
            padding = ((0, 0), (0, max_cols - fmri_data.shape[1]))  # Define the padding
            padded_fmri_subject = np.pad(fmri_data, padding, 'constant', constant_values=0)
            padded_fmri_data.append(padded_fmri_subject)  # Pad with zeros

        # Now you can concatenate the padded data
        self.fmri_data_all = np.float32(np.concatenate(padded_fmri_data, axis=0))
        # print(self.fmri_data_all.shape)

        # self.fmri_data_all = np.float32(np.concatenate(fmri_data_stack))
        self.image_ids_all = np.concatenate(image_id_stack)
        self.image_paths = []
        self.img_ids = self.img_ids.set_index(0, drop=True)
        for img_id in self.image_ids_all:
            self.image_paths.append(self.img_ids.iloc[self.img_ids.index.get_loc(img_id)][1])
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = str(self.data_dir / 'images' / self.split / self.image_paths[idx])
        # image = read_image(img_path)
        image = Image.open(img_path)

        num_channels = len(image.getbands())
        if num_channels == 1:
            # image = torch.concat([image]*3)
            image = image.convert('RGB')

        fmri = self.fmri_data_all[idx]
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, fmri.astype(np.float32)


class GODLoader():
    def __init__(self, data_dir, batch_size=16) -> None:
        image_transforms = transforms.Compose([
            transforms.Resize((128, 128)),      # Resize the image to 64x64 pixels
            transforms.ToTensor(),              # Convert the image to a PyTorch tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ])
        self.batch_size = batch_size
        self.train_loader = DataLoader(GODDataset(data_dir=data_dir, image_transforms=image_transforms, split='training'), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(GODDataset(data_dir=data_dir, image_transforms=image_transforms, split='test'), batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
        

class ImageDataset(Dataset):
    """The 'train_cls' set of ImageNet containing 1.2M images.

    Also supports 'test' and 'val' sets."""
    def __init__(self, data_dir, image_transforms, split='train') -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_transforms = image_transforms

        df_basepath = self.data_dir / 'ImageSets' / 'CLS-LOC'
        df_fname = f'{split}_cls.txt' if split == 'train' else f'{split}.txt'
        self.img_info = (
            pd.read_csv(df_basepath/df_fname, header=None, sep=' ')
                .assign(path=lambda df: df[0].apply(self._format_img_path))
                .set_index(1)
        )

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img_path = self.img_info.at[idx+1, 'path']
        # image = read_image(img_path)
        image = Image.open(img_path)
        num_channels = len(image.getbands())
        if num_channels == 1:
            # image = torch.concat([image]*3)
            image = image.convert('RGB')
        if num_channels > 3:
            # apparently there is a bug with the PIL so we need to convert image to RGBA first 
            # before converting into RGB:
            # https://stackoverflow.com/a/70191990/6411761
            # https://stackoverflow.com/a/1963146/6411761
            image = image.convert('RGBA').convert('RGB')
        if self.image_transforms:
            image = self.image_transforms(image)

        return image

    def _format_img_path(self, img_id):
        return str(self.data_dir/'Data'/'CLS-LOC'/self.split) + f'/{img_id}.JPEG'


# class ImageDataset(Dataset):
#     """Dummy class for testing purposes"""
#     def __init__(self, data_dir, image_transforms, split='train') -> None:
#         super().__init__()

#     def __len__(self):
#         return 10

#     def __getitem__(self, idx):
#         random_tensor = torch.rand(3, 128, 128)
#         return random_tensor

class ImageLoader():
    def __init__(self, data_dir, batch_size=16, image_size=256) -> None:
        self.image_size = image_size
        image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),      # Resize the image to 256x256 pixels
            transforms.ToTensor(),              # Convert the image to a PyTorch tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ])
        self.batch_size = batch_size
        self.train_loader = DataLoader(ImageDataset(data_dir=data_dir, image_transforms=image_transforms, split='train'), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(ImageDataset(data_dir=data_dir, image_transforms=image_transforms, split='test'), batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader

class ImNetImage(torch.Tensor):
    # Class used only for type hinting
    pass 


class FMRI:
    pass
