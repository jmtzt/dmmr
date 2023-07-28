import torch
import random
import lightning as pl
import torchio as tio
import numpy as np
import nibabel as nib
import copy

from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm.auto import tqdm

from utils.randconv import randconv


class BasePatchDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = '/home/joao/data/BraTS/',
                 inter_subj: bool = False,
                 split_ratio: float = 0.9,
                 patch_size: int = 17,
                 samples_per_volume: int = 16,
                 patch_sampler: str = 'label',
                 training_batch_size: int = 512,
                 validation_batch_size: int = 1024,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 mask_threshold: float = 0,
                 overfit: bool = False,
                 modality: str = 't1t2',
                 random_convs: bool = False,
                 *args,
                 **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.inter_subj = inter_subj
        self.split_ratio = split_ratio
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.patch_sampler = patch_sampler
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mask_threshold = mask_threshold
        self.overfit = overfit
        self.modality = modality
        self.negative_transforms = None
        self.positive_transforms = None
        self.random_convs = random_convs
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def _setup_transforms(is_positive: bool = True):
        # TODO: add a lot of rotation transforms here in smaller degrees
        if is_positive:
            pos_tfms = {}
            num_tfms = 10
            total_prob = 0.8
            degrees_arr = np.linspace(0, 270, num=num_tfms, endpoint=True)
            for i, degrees in enumerate(degrees_arr):
                degrees_tuple = (degrees,) * 6
                transform_value = tio.RandomAffine(scales=0, degrees=degrees_tuple,
                                                   include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask'])
                probability = total_prob / num_tfms
                pos_tfms[transform_value] = probability

            pos_tfms[tio.RandomFlip(axes=(0, 1, 2), flip_probability=1,
                                    include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask'])] = 0.10
            pos_tfms[tio.RandomBlur(include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask'])] = 0.10

            training_transform = tio.Compose([
                tio.OneOf(
                    pos_tfms, include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask']),
            ])
        else:
            training_transform = tio.Compose([
                tio.OneOf({
                    tio.RandomFlip(axes=0, flip_probability=1, include=['mod1', 'mod1_mask']): 0.15,
                    tio.RandomAffine(scales=0, degrees=(90, 90, 90), include=['mod1', 'mod1_mask']): 0.15,
                    tio.RandomAffine(scales=0, degrees=(180, 180, 180), include=['mod1', 'mod1_mask']): 0.25,
                    tio.RandomAffine(scales=0, degrees=(270, 270, 270), include=['mod1', 'mod1_mask']): 0.25,
                    tio.RandomBlur(include=['mod1', 'mod1_mask',]): 0.20,
                }, include=['mod1', 'mod1_mask']),
                tio.OneOf({
                    tio.RandomFlip(axes=1, flip_probability=1, include=['mod2', 'mod2_mask']): 0.15,
                    tio.RandomAffine(scales=0, degrees=90, include=['mod2', 'mod2_mask']): 0.25,
                    tio.RandomAffine(scales=0, degrees=180, include=['mod2', 'mod2_mask']): 0.15,
                    tio.RandomAffine(scales=0, degrees=270, include=['mod2', 'mod2_mask']): 0.25,
                    tio.RandomBlur(include=['mod2', 'mod2_mask']): 0.20,
                }, include=['mod2', 'mod2_mask']),
            ])

        validation_transform = tio.Compose([
            tio.ToCanonical(),
        ])

        return training_transform, validation_transform

    def _split_subjects(self, subjects):
        num_subjects = len(subjects)
        num_training_subjects = int(self.split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects

        num_split_subjects = num_training_subjects, num_validation_subjects
        training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

        training_set = tio.SubjectsDataset(training_subjects, transform=tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0, 1))]))
        validation_set = tio.SubjectsDataset(validation_subjects, transform=tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0, 1))]))

        return training_set, validation_set

    def _get_sampler(self):
        if self.patch_sampler == 'uniform':
            return tio.data.UniformSampler(self.patch_size)
        elif self.patch_sampler == 'weighted':
            return tio.data.WeightedSampler(self.patch_size)
        elif self.patch_sampler == 'label':
            return tio.data.LabelSampler(self.patch_size, label_name='mod1_mask')
        else:
            raise ValueError(f'Unknown patch sampler: {self.patch_sampler}')

    def _load_patch_subjects(self, subjects, num_patches_per_subject, random_conv=False):
        self.sampler = self._get_sampler()

        patches = []

        for subject in subjects:
            for patch in self.sampler(subject, num_patches=num_patches_per_subject):
                if random_conv:
                    patch = self._random_conv(patch)
                patches.append(patch)

        return patches

    def _load_subjects(self, stage):
        raise NotImplementedError

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.subjects = self._load_subjects(stage)

            self.training_subjects, self.validation_subjects = self._split_subjects(self.subjects)

            training_patches = self._load_patch_subjects(self.training_subjects,
                                                         self.samples_per_volume,
                                                         random_conv=self.random_convs)

            validation_patches = self._load_patch_subjects(self.validation_subjects,
                                                           self.samples_per_volume,
                                                           random_conv=self.random_convs)

            self.positive_transforms, _ = self._setup_transforms(is_positive=True)
            self.negative_transforms, _ = self._setup_transforms(is_positive=False)

            if self.overfit:
                training_patches = training_patches[:2]

            self.patches_training_set = CustomPatchesDataset(training_patches,
                                                             self.positive_transforms,
                                                             self.negative_transforms)

            self.patches_validation_set = CustomPatchesDataset(validation_patches,
                                                               self.positive_transforms,
                                                               self.negative_transforms)
        elif stage == 'test':
            self.subjects = tio.SubjectsDataset(self._load_subjects(stage),
                                                transform=tio.RescaleIntensity(out_min_max=(0, 1)))

            patches = self._load_patch_subjects(self.subjects,
                                                self.samples_per_volume)
            self.positive_transforms, _ = self._setup_transforms(is_positive=True)
            self.negative_transforms, _ = self._setup_transforms(is_positive=False)

            self.patches_test_set = CustomPatchesDataset(patches,
                                                         self.positive_transforms,
                                                         self.negative_transforms)

    def train_dataloader(self):
        return DataLoader(self.patches_training_set,
                          batch_size=self.training_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,)

    def val_dataloader(self):
        return DataLoader(self.patches_validation_set,
                          batch_size=self.validation_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,)


class BraTSDataModule(BasePatchDataModule):

    def _load_subjects(self, stage):

        if stage == 'fit' or stage is None:
            prefix = 'Training'
        elif stage == 'test' or stage == 'val':
            prefix = 'Validation'

        imgs_dir = self.data_dir / f'MICCAI_BraTS2020_{prefix}Data/'

        def glob_nii_fn(folder, pattern):
            nii_files = folder.glob(pattern)
            return sorted(nii_files)

        t1_img_paths = glob_nii_fn(imgs_dir, f'*/BraTS20_{prefix}_*_t1.nii.gz')
        if self.modality == 't1t2':
            t2_img_paths = glob_nii_fn(imgs_dir, f'*/BraTS20_{prefix}_*_t2.nii.gz')
        elif self.modality == 't1t1':
            t2_img_paths = glob_nii_fn(imgs_dir, f'*/BraTS20_{prefix}_*_t1.nii.gz')

        if self.inter_subj or self.modality == 't1t1':
            # shuffle t2 images so that they don't correspond to the same subject!
            random.shuffle(t2_img_paths)

        subjects = []
        for t1_path, t2_path in zip(t1_img_paths, t2_img_paths):
            subject = tio.Subject(
                mod1=tio.ScalarImage(t1_path),
                mod2=tio.ScalarImage(t2_path),
            )
            mod1_mask = torch.zeros_like(subject['mod1'][tio.DATA])
            mod1_mask[subject['mod1'][tio.DATA] > 0] = 1
            mod2_mask = torch.zeros_like(subject['mod2'][tio.DATA])
            mod2_mask[subject['mod2'][tio.DATA] > 0] = 1
            subject['mod1_mask'] = tio.LabelMap(tensor=mod1_mask)
            subject['mod2_mask'] = tio.LabelMap(tensor=mod2_mask)
            subject = tio.CopyAffine('mod1')(subject)
            subjects.append(subject)

        return subjects


class CamCANDataModule(BasePatchDataModule):

    def _load_subjects(self, stage):

        if stage == 'fit' or stage is None:
            imgs_dir = self.data_dir / 'train/'
        elif stage == 'test' or stage == 'val':
            imgs_dir = self.data_dir / 'val/'

        def glob_nii_fn(folder, pattern):
            nii_files = folder.glob(pattern)
            return sorted(nii_files)

        t1_img_paths = glob_nii_fn(imgs_dir, '*/T1_brain.nii.gz')
        if self.modality == 't1t2':
            t2_img_paths = glob_nii_fn(imgs_dir, '*/T2_brain.nii.gz')
        elif self.modality == 't1t1':
            t2_img_paths = glob_nii_fn(imgs_dir, '*/T1_brain.nii.gz')

        if self.inter_subj or self.modality == 't1t1':
            # shuffle t2 images so that they don't correspond to the same subject!
            random.shuffle(t2_img_paths)

        if self.overfit:
            t1_img_paths = t1_img_paths[:2]
            t2_img_paths = t2_img_paths[:2]

        subjects = []

        for t1_path, t2_path in zip(t1_img_paths, t2_img_paths):
            mask1_path = t1_path.parent / 'T1_brain_MALPEM_tissues.nii.gz'
            mask2_path = t2_path.parent / 'T1_brain_MALPEM_tissues.nii.gz'
            subject = tio.Subject(
                mod1=tio.ScalarImage(t1_path),
                mod2=tio.ScalarImage(t2_path),
                mod1_mask=tio.LabelMap(mask1_path),
                mod2_mask=tio.LabelMap(mask2_path),
            )
            subjects.append(subject)

        return subjects


class CustomPatchesDataset(tio.SubjectsDataset):
    def __init__(self, patches, positive_transform, negative_transform, show=False):
        super().__init__(subjects=patches)
        self.positive_transform = positive_transform
        self.negative_transform = negative_transform
        self.show = show

    def __getitem__(self, index):
        try:
            index = int(index)
        except (RuntimeError, TypeError):
            message = (
                f'Index "{index}" must be int or compatible dtype,'
                f' but an object of type "{type(index)}" was passed'
            )
            raise ValueError(message)

        subject = self._subjects[index]
        if self.show:
            print(f'Loading subject from path: {subject["mod1"].path}...')
        subject = copy.deepcopy(subject)  # cheap since images not loaded yet
        if self.load_getitem:
            subject.load()

        # Apply transform (this is usually the bottleneck)
        if self.positive_transform is not None and self.negative_transform is not None:
            if index % 2 == 0:
                subject = self.negative_transform(subject)
                subject['label'] = torch.tensor(1, dtype=torch.float32)
            else:
                subject = self.positive_transform(subject)
                subject['label'] = torch.tensor(0, dtype=torch.float32)
        return subject
