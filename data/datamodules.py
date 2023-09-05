import copy
import random
from pathlib import Path

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.inference import show_image


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
                 online_augmentations: bool = False,
                 random_convs: bool = False,
                 multiclass: bool = False,
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
        self.online_augmentations = online_augmentations
        self.random_convs = random_convs
        self.multiclass = multiclass
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def _setup_transforms(is_positive: bool = True):
        if is_positive:
            pos_tfms = {}
            num_tfms = 3
            total_prob = 0.8
            degrees_arr = np.linspace(0, 270, num=num_tfms, endpoint=True)
            for i, degrees in enumerate(degrees_arr):
                degrees_tuple = (degrees,) * 6
                neg_degrees_tuple = (-degrees,) * 6
                transform_value = tio.RandomAffine(scales=0, degrees=degrees_tuple,
                                                   include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask'])
                neg_transform_value = tio.RandomAffine(scales=0, degrees=neg_degrees_tuple,
                                                       include=['mod1', 'mod2', 'mod1_mask', 'mod2_mask'])
                probability = total_prob / num_tfms
                pos_tfms[transform_value] = probability / 2
                pos_tfms[neg_transform_value] = probability / 2

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
                    tio.RandomBlur(include=['mod1', 'mod1_mask', ]): 0.20,
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

        for subject in tqdm(subjects, desc='Loading patches from subjects'):
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

    @staticmethod
    def multiclass_label_collate_fn(batch):
        flip_params = {'axes': (0, 1, 2)}

        def generate_transforms(modules, flip_params, affine_params):
            transform = tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=0, degrees=affine_params['degrees'],
                                     translation=0, include=modules): 0.5,
                    tio.RandomAffine(scales=0, degrees=tuple(-1 * np.array(affine_params['degrees'])),
                                     translation=0, include=modules): 0.5,
                }, include=modules),
                tio.RandomFlip(**flip_params, include=modules),
            ])
            return transform

        def generate_pos_transforms(modules, flip_params, affine_params):
            transform = tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=0, degrees=affine_params['degrees'], translation=0, include=modules): 0.3,
                    tio.RandomAffine(scales=affine_params['scales'], degrees=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(scales=0, degrees=0, translation=affine_params['translation'],
                                     include=modules): 0.2,
                    tio.RandomAffine(degrees=affine_params['x_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['y_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['z_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['degrees'],
                                     scales=affine_params['scales'],
                                     translation=affine_params['translation'],
                                     include=modules): 0.1,
                }, include=modules),
                tio.RandomFlip(**flip_params, include=modules),
            ])
            return transform

        affine_params = {'degrees': (-25, 25),
                            'scales': (0.98, 1.2),
                            'translation': (0.5, 0.5, 0.5),
                            'x_degrees': (25, 0, 0),
                            'y_degrees': (0, 25, 0),
                            'z_degrees': (0, 0, 25), }
        transform_pos1 = generate_pos_transforms(['mod1', 'mod2'],
                                                 flip_params,
                                                    affine_params)
        transform_neg = tio.Compose([generate_pos_transforms(['mod1'], flip_params, affine_params),
                                     generate_pos_transforms(['mod2'], flip_params, affine_params),
                                     tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])])

        transform_neg1 = tio.Compose([
            generate_transforms(['mod1'], flip_params, {'degrees': (0, 0, 0, 0, 0, 0)}),
            generate_transforms(['mod2'], flip_params, {'degrees': (45, 45, 45, 45, 45, 45)}),
            tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])
        ])
        transform_neg2 = tio.Compose([
            generate_transforms(['mod1'], flip_params, {'degrees': (0, 0, 0, 0, 0, 0)}),
            generate_transforms(['mod2'], flip_params, {'degrees': (90, 90, 90, 90, 90, 90)}),
            tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])
        ])
        transform_neg3 = tio.Compose([
            generate_transforms(['mod1'], flip_params, {'degrees': (0, 0, 0, 0, 0, 0)}),
            generate_transforms(['mod2'], flip_params, {'degrees': (135, 135, 135, 135, 135, 135)}),
            tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])
        ])
        transform_neg4 = tio.Compose([
            generate_transforms(['mod1'], flip_params, {'degrees': (0, 0, 0, 0, 0, 0)}),
            generate_transforms(['mod2'], flip_params, {'degrees': (180, 180, 180, 180, 180, 180)}),
            tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])
        ])

        def process_item(item, transform, label_value):
            tfm_item = transform(item)
            mod1 = tfm_item['mod1'][tio.DATA]
            mod2 = tfm_item['mod2'][tio.DATA]
            label = torch.tensor(label_value)
            history = tfm_item.history
            return mod1, mod2, label, history

        positive_transforms = [transform_pos1]
        negative_transforms = [transform_neg]
        # negative_transforms = [transform_neg1, transform_neg2, transform_neg3, transform_neg4][:1]

        mod1_augmented_list = []
        mod2_augmented_list = []
        labels_augmented_list = []
        for idx, item in enumerate(batch):
            if idx < len(batch) // 2:
                transform_index = random.randint(0, len(positive_transforms) - 1)
                transform = positive_transforms[transform_index]
                mod1, mod2, label, history = process_item(item, transform, transform_index)
            else:  # Second half of items get negative transforms
                transform_index = random.randint(0, len(negative_transforms) - 1)
                transform = negative_transforms[transform_index]
                mod1, mod2, label, history = process_item(item, transform, len(positive_transforms) + transform_index)

            mod1_augmented_list.append(mod1)
            mod2_augmented_list.append(mod2)
            labels_augmented_list.append(label)

        mod1 = torch.stack(mod1_augmented_list).float()
        mod2 = torch.stack(mod2_augmented_list).float()
        labels = torch.stack(labels_augmented_list).float()

        unique_labels, counts = torch.unique(labels, return_counts=True)

        return {'mod1': mod1, 'mod2': mod2, 'label': labels}

    @staticmethod
    def label_collate_fn(batch):
        affine_params = {'degrees': (-25, 25),
                         'scales': (0.98, 1.2),
                         'translation': (0.5, 0.5, 0.5),
                         'x_degrees': (25, 0, 0),
                         'y_degrees': (0, 25, 0),
                         'z_degrees': (0, 0, 25), }
        flip_params = {'axes': (0, 1, 2)}

        def generate_transforms(modules, flip_params, affine_params):
            transform = tio.Compose([
                tio.OneOf({
                    tio.RandomAffine(scales=0, degrees=affine_params['degrees'], translation=0, include=modules): 0.3,
                    tio.RandomAffine(scales=affine_params['scales'], degrees=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(scales=0, degrees=0, translation=affine_params['translation'],
                                     include=modules): 0.2,
                    tio.RandomAffine(degrees=affine_params['x_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['y_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['z_degrees'], scales=0, translation=0, include=modules): 0.1,
                    tio.RandomAffine(degrees=affine_params['degrees'],
                                     scales=affine_params['scales'],
                                     translation=affine_params['translation'],
                                     include=modules): 0.1,
                }, include=modules),
                tio.RandomFlip(**flip_params, include=modules),
            ])
            return transform

        transform_pos = generate_transforms(['mod1', 'mod2'], flip_params, affine_params)
        transform_neg = tio.Compose([generate_transforms(['mod1'], flip_params, affine_params),
                                     generate_transforms(['mod2'], flip_params, affine_params),
                                     tio.RandomFlip(**flip_params, include=['mod1', 'mod2'])])

        def process_item(item, transform, label_value):
            tfm_item = transform(item)
            mod1 = tfm_item['mod1'][tio.DATA]
            mod2 = tfm_item['mod2'][tio.DATA]
            label = torch.tensor(label_value)
            history = tfm_item.history
            return mod1, mod2, label, history

        def process_hist(hist):
            for t in hist:
                if t.name == 'Affine':
                    round_tuple = lambda t: tuple(round(x, 2) for x in t)
                    t.value = (f'deg{str(round_tuple(t.degrees))}-'
                               f'scl{str(round_tuple(t.scales))}-'
                               f'trn{str(round_tuple(t.translation))}')
                else:
                    t.value = ''

            return hist

        show = False
        mod1_augmented_list = []
        mod2_augmented_list = []
        labels_augmented_list = []
        for idx, item in enumerate(batch):

            mod1_og, mod2_og = item['mod1'][tio.DATA], item['mod2'][tio.DATA]

            if idx % 2 == 0:
                mod1, mod2, label, history = process_item(item, transform_pos, 0.0)
            else:
                mod1, mod2, label, history = process_item(item, transform_neg, 1.0)

            if show and idx < 16:
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
                history = process_hist(history)
                tfm_string = '\n'.join(f'{t.name}: {t.value}' for t in history)
                show_image(mod1_og[0, mod1_og.shape[1] // 2],
                           f"mod1_og", ax=axes[0])
                show_image(mod2_og[0, mod2_og.shape[1] // 2],
                           f"mod2_og", ax=axes[1])
                show_image(mod1[0, mod1.shape[1] // 2],
                           f"mod1", ax=axes[2])
                show_image(mod2[0, mod2.shape[1] // 2],
                           f"mod2 - idx {idx} - label {label}", ax=axes[3])
                axes[3].text(1.05, 0.5, tfm_string, fontsize=10, ha='left', va='center',
                             transform=axes[3].transAxes)
                plt.tight_layout()
                plt.show()

            mod1_augmented_list.append(mod1)
            mod2_augmented_list.append(mod2)
            labels_augmented_list.append(label)

        mod1 = torch.stack(mod1_augmented_list).float()
        mod2 = torch.stack(mod2_augmented_list).float()
        labels = torch.stack(labels_augmented_list).float()

        return {'mod1': mod1, 'mod2': mod2, 'label': labels}

    def train_dataloader(self):
        if self.online_augmentations:
            patches_queue = tio.Queue(
                self.training_subjects,
                self.samples_per_volume * 4,
                self.samples_per_volume,
                self._get_sampler(),
                num_workers=self.num_workers,
            )
            return DataLoader(
                patches_queue,
                batch_size=self.training_batch_size,
                num_workers=0,
                collate_fn=self.label_collate_fn if not self.multiclass else self.multiclass_label_collate_fn,
            )
        else:
            return DataLoader(self.patches_training_set,
                              batch_size=self.training_batch_size,
                              num_workers=self.num_workers,
                              shuffle=True,
                              pin_memory=True,)

    def val_dataloader(self):
        if not self.multiclass:
            return DataLoader(self.patches_validation_set,
                              batch_size=self.validation_batch_size,
                              num_workers=self.num_workers,
                              shuffle=False,
                              pin_memory=True, )
        else:
            patches_queue = tio.Queue(
                self.validation_subjects,
                self.samples_per_volume * 4,
                self.samples_per_volume,
                self._get_sampler(),
                num_workers=self.num_workers,
            )
            return DataLoader(
                patches_queue,
                batch_size=self.validation_batch_size,
                num_workers=0,
                collate_fn=self.multiclass_label_collate_fn,
            )


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

        for t1_path, t2_path in tqdm(zip(t1_img_paths, t2_img_paths), desc="Loading subjects"):
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


class IXIDataModule(BasePatchDataModule):
    def _load_subjects(self, stage):
        def glob_nii_fn(folder, pattern):
            nii_files = folder.glob(pattern)
            return sorted(nii_files)

        t1_img_paths = glob_nii_fn(self.data_dir, '*-T1.nii.gz')
        if self.modality == 't1t2':
            t2_img_paths = glob_nii_fn(self.data_dir, '*-T2.nii.gz')
        elif self.modality == 't1t1':
            t2_img_paths = glob_nii_fn(self.data_dir, '*-T1.nii.gz')
        else:
            raise ValueError(f"Modality {self.modality} not supported")

        t1_prefixes = {str(f).split('-T1.nii.gz')[0] for f in t1_img_paths}
        t2_prefixes = {str(f).split('-T2.nii.gz')[0] for f in t2_img_paths}
        common_prefixes = t1_prefixes.intersection(t2_prefixes)
        t1_img_paths = [f for f in t1_img_paths if str(f).split('-T1.nii.gz')[0] in common_prefixes]
        t2_img_paths = [f for f in t2_img_paths if str(f).split('-T2.nii.gz')[0] in common_prefixes]

        if self.inter_subj or self.modality == 't1t1':
            # shuffle t2 images so that they don't correspond to the same subject!
            random.shuffle(t2_img_paths)

        if self.overfit:
            t1_img_paths = t1_img_paths[:2]
            t2_img_paths = t2_img_paths[:2]

        def create_binary_mask(image):
            mask_data = (image.data > 0).int()
            return tio.LabelMap(tensor=mask_data, affine=image.affine)

        tfms = tio.Compose([
            tio.ToCanonical(),
            tio.Resample('mod1')
        ])
        subjects = []
        for t1_path, t2_path in tqdm(zip(t1_img_paths, t2_img_paths), desc="Loading subjects"):
            t1_image = tio.ScalarImage(t1_path)
            t2_image = tio.ScalarImage(t2_path)

            mod1_mask = create_binary_mask(t1_image)
            mod2_mask = create_binary_mask(t2_image)

            subject = tio.Subject(
                mod1=t1_image,
                mod2=t2_image,
                mod1_mask=mod1_mask,
                mod2_mask=mod2_mask,
            )
            subject = tfms(subject)
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
