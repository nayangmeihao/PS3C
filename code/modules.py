import datetime

from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import matplotlib.pyplot as plt
import torch
import logging

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = datetime.datetime.fromtimestamp(record.created).strftime(datefmt)
        else:
            s = datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return s
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('training.log')

        formatter = CustomFormatter(
            '[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'[:-3]
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


#
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            if folder_name == '__MACOSX':
                continue

            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name == 'bothcells' or folder_name == 'unhealthy':
                    label = 'unhealthy'
                else:
                    label = folder_name
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        self.label_map = {'unhealthy': 0, 'healthy': 1, 'rubbish': 2}
        self.labels = [self.label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def reinhard_standardization_fun(source_image, target_mean=[172.60249999,133.93109593,125.11238891], target_std=[41.87876043,10.9258465,10.77863437]):
    # Reshape target_mean and target_std
    target_mean = np.array(target_mean).reshape((1, 1, 3))
    target_std = np.array(target_std).reshape((1, 1, 3))

    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calculate mean and std deviation for the source patch
    source_mean, source_std = cv2.meanStdDev(source_lab)
    source_mean = source_mean.reshape((1, 1, 3))
    source_std = source_std.reshape((1, 1, 3))

    # Apply Reinhard standardization
    standardized_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
    standardized_lab = np.clip(standardized_lab, 0, 255).astype(np.uint8)

    # Convert back to BGR color space
    standardized_image = cv2.cvtColor(standardized_lab, cv2.COLOR_LAB2BGR)
    return standardized_image

class CustomDataset_v2(Dataset):
    def __init__(self, root_dir, transform=None, apply_reinhard=True):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_reinhard = apply_reinhard
        self.image_paths = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            if folder_name == '__MACOSX':
                continue

            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name == 'bothcells' or folder_name == 'unhealthy':
                    label = 'unhealthy'
                else:
                    label = folder_name  #
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

        self.label_map = {'unhealthy': 0, 'healthy': 1, 'rubbish': 2}
        self.labels = [self.label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        original_image = self._load_image(image_path)
        image = cv2.imread(image_path)

        if self.apply_reinhard:
            image = reinhard_standardization_fun(image)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def _load_image(self, image_path):
        """
        :param image_path: image path
        :return: image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or corrupted: {image_path}")
        # RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

def visualize_processed_data(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        original_image, processed_image, label = dataset[i]

        original_image_np = np.array(original_image)
        processed_image_np = np.transpose(np.array(processed_image), (1, 2, 0))
        axes[i, 0].imshow(original_image_np)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Original Image\nLabel: {label}')

        axes[i, 1].imshow(processed_image_np)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Processed Image\nLabel: {label}')

    plt.tight_layout()
    plt.show()


class ExternalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for image_name in os.listdir(root_dir):
            image_path = os.path.join(root_dir, image_name)
            if os.path.isfile(image_path):
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, image_path


def calculate_mean_std(root_dir):
    means = []
    stds = []
    for folder_name in os.listdir(root_dir):
        print(folder_name)
        if folder_name == '__MACOSX':
            continue
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            if folder_name == 'bothcells' or folder_name == 'unhealthy':
                label = 'unhealthy'
            else:
                label = folder_name  #
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path)
                image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
                mean, std = cv2.meanStdDev(image_lab)
                means.append(mean)
                stds.append(std)
        print("finished!")
    #
    mean_of_means = np.mean(means, axis=0)
    std_of_stds = np.mean(stds, axis=0)
    # target_mean = [172.60249999, 133.93109593, 125.11238891], target_std = [41.87876043, 10.9258465, 10.77863437]
    return mean_of_means, std_of_stds

class EarlyStopping:
    def __init__(self, patience=10, delta=0, check_path='../result/resnet50_checkpoint.pt', model_path="../result/resnet50_mdel.pth", verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
            verbose (bool): Whether to print messages about early stopping.
        """
        self.logger = get_logger('MyClassLogger')
        self.patience = patience
        self.delta = delta
        self.path = check_path
        self.model_path = model_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf  # Initialize to negative infinity

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score > self.val_acc_max:  # Compare with max validation accuracy
            self.val_acc_max = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0  # Reset counter since we have a new best model
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_acc, model):
        """Save model when validation accuracy improves."""
        if self.verbose:
            # print(f"Validation accuracy improved ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model ...")
            self.logger.info(f"Validation F1-score improved ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        # model
        torch.save(model, self.model_path)
