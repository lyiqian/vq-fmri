import torch 
from torch.utils.data import Dataset
import pandas as pd 
from pathlib import Path
import numpy as np 
import bdpy
from torchvision.io import read_image

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
            df_fname = 'image_testing_id.csv'
            data_type = 2

        df_path = data_dir / 'images' / df_fname
        self.img_ids = pd.read_csv(df_path, header=None)

        subject_files = [p for p in data_dir.glob("*.h5")]
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
        
        self.fmri_data_all = np.concatenate(fmri_data_stack)
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
        image = read_image(img_path)

        if image.shape == (1, 500, 500):
            image = torch.concat([image]*3)

        fmri = self.fmri_data_all[idx]
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, fmri


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
        image = read_image(img_path)

        if self.image_transforms:
            image = self.image_transforms(image)

        return image

    def _format_img_path(self, img_id):
        return str(self.data_dir/'Data'/'CLS-LOC'/self.split) + f'/{img_id}.JPEG'


class Image:
    pass


class FMRI:
    pass

# god = GODDataset('/Users/bahman/Documents/courses/Deep Learning/Project/code/data', None)
# print(god)
# for im, fmri in god:
#     print(im.shape)
#     print(fmri.shape)
#     exit()
