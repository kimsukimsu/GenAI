import torch
import os
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from functools import reduce
from operator import __or__
import random

class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, org_dataset, aug_dataset):
        """
        org_dataset: 원본 데이터셋 (예: ImageFolder)
        aug_dataset: 증강/혼합된 데이터셋 (예: ImageFolder)
        """
        self.org_dataset = org_dataset
        self.aug_dataset = aug_dataset
        self.blend_ratio = 0.0  # 0.0: Original 100%, 1.0: Augmented 100%
        
        # Sampler 생성을 위해 targets와 classes 정보를 원본에서 가져옴
        self.targets = np.array(org_dataset.targets) 
        self.classes = org_dataset.classes

    def set_blend_ratio(self, ratio):
        """main.py에서 Epoch마다 이 함수를 호출하여 비율을 변경함"""
        self.blend_ratio = ratio

    def __len__(self):
        # 데이터셋의 크기는 원본 데이터셋의 크기를 따름
        return len(self.org_dataset)

    def __getitem__(self, idx):
        # [핵심 로직] blend_ratio 확률보다 작으면 Augmented(Blended) 이미지 반환
        if random.random() < self.blend_ratio:
            # 인덱스 안전장치 (혹시 aug 데이터셋이 더 작을 경우 대비)
            if idx < len(self.aug_dataset):
                return self.aug_dataset[idx]
            else:
                return self.org_dataset[idx]
        else:
            # 그 외의 경우 원본 이미지 반환
            return self.org_dataset[idx]



def load_data_subset(batch_size,
                     workers,
                     dataset,
                     data_train_org_dir,
                     data_train_aug_dir,
                     data_test_dir,
                     labels_per_class=100,
                     valid_labels_per_class=500,
                     mixup_alpha=1):
    '''return datalaoder'''
    ## copied from GibbsNet_pytorch/load.py
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'tiny-imagenet-200':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    else:
        assert False, "Unknow dataset : {}".format(dataset)

    # pre-processing
    if dataset == 'tiny-imagenet-200':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

    # dataset
    if dataset == 'cifar10':
        pass
        train_data = datasets.CIFAR10(data_train_org_dir,
                                      train=True,
                                      transform=train_transform,
                                      download=True)
        test_data = datasets.CIFAR10(data_test_dir,
                                     train=False,
                                     transform=test_transform,
                                     download=True)
        num_classes = 10

    elif dataset == 'cifar100':
        #from torch.utils.data import ConcatDataset
        
        train_root_1 = data_train_org_dir
        train_root_2 = data_train_aug_dir
        test_root = data_test_dir 

        if not os.path.exists(train_root_1) or not os.path.exists(train_root_2):
            raise FileNotFoundError(f"One of the train directories not found. Check paths.")
        if not os.path.exists(test_root):
            raise FileNotFoundError(f"Test directory not found: {test_root}")

        print(f"Loading train data from: {train_root_1}")
        train_data_1 = datasets.ImageFolder(train_root_1, transform=train_transform)
        
        print(f"Loading train data from: {train_root_2}")
        train_data_2 = datasets.ImageFolder(train_root_2, transform=train_transform)

        if train_data_1.classes != train_data_2.classes:
            raise ValueError("Class list/order mismatch between the two train directories. "
                             "This will cause incorrect labels.")
            
        train_data = CurriculumDataset(train_data_1, train_data_2) # 기존: train_data = ConcatDataset([train_data_1, train_data_2])
        print(f"Created CurriculumDataset. Total size (per epoch): {len(train_data)}")
        
        test_data = datasets.ImageFolder(test_root, transform=test_transform)
        
        num_classes = len(train_data.classes)
        print(f"Found {num_classes} classes total.")

    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    n_labels = num_classes

    # random sampler
    def get_sampler(labels, n=None, n_valid=None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        (indices, ) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        np.random.shuffle(indices)

        indices_valid = np.hstack([
            list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)
        ])
        indices_train = np.hstack([
            list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid + n]
            for i in range(n_labels)
        ])
        indices_unlabelled = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])

        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    if dataset == 'tiny-imagenet-200':
        pass
    else:
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(
            train_data.targets, labels_per_class, valid_labels_per_class)

    # dataloader
    if dataset == 'tiny-imagenet-200':
        labelled = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)
        validation = None
        unlabelled = None
        test = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=workers,
                                           pin_memory=True)
    else:
        labelled = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               shuffle=False,
                                               num_workers=workers,
                                               pin_memory=True)
        validation = torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 sampler=valid_sampler,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True)
        unlabelled = torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 sampler=unlabelled_sampler,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True)
        test = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=workers,
                                           pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes


def create_val_folder(data_set_path):
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the pytorch dataloaders
    """
    path = os.path.join(data_set_path, 'val/images')  # path where validation data is present now
    filename = os.path.join(data_set_path,
                            'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")
    data = fp.readlines()

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


if __name__ == "__main__":
    create_val_folder('data/tiny-imagenet-200')
