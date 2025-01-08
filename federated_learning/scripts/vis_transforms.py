import random
import matplotlib.pyplot as plt
import cv2
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

MAX_BOXES = 10
BATCH_SIZE = 8

class BoneFractureDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)

        # Convert image to RGB (OpenCV loads images as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)
            image = image['image']
        else:
            # Default transformation to tensor and normalization
            image = transforms.ToTensor()(image)

        label_path = os.path.join(self.label_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            labels = f.readlines()

        boxes = []
        for label in labels:
            parts = label.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            if len(boxes) < MAX_BOXES:
                boxes.append([bbox[0]-bbox[2],bbox[1]-bbox[3],bbox[0]+bbox[2],bbox[1]+bbox[3]])

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor([1 for _ in range(len(boxes))], dtype=torch.int64)}

        return image, target

# Augmentation transforms
rand_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.5),
])

# Create datasets for train, validation, and test sets
train_dataset = BoneFractureDataset(image_dir='do1/train/images', label_dir='do1/train/labels')
valid_dataset = BoneFractureDataset(image_dir='do1/valid/images', label_dir='do1/valid/labels')
test_dataset = BoneFractureDataset(image_dir='do1/test/images', label_dir='do1/test/labels')

def collate_fn(batch):
    return tuple(zip(*batch))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def denormalize_image(image, mean, std):
    """Denormalize image given mean and std"""
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image * std) + mean
    return np.clip(image, 0, 1)


def resize_image(image, target_size):
    return cv2.resize(image, (target_size[1], target_size[0]))

def save_transformations(dataset, transform_list, save_dir='img'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Select a random image from the dataset
    random_idx = random.randint(0, len(dataset) - 1)
    image, _ = dataset[random_idx]

    # Convert tensor to NumPy array for visualization
    image_np = image.permute(1, 2, 0).numpy()
    original_image_np = image_np * 255
    original_image_np = original_image_np.astype(np.uint8)

    # Apply each transformation individually and save the images
    for individual_transform in transform_list:
        transform_compose = A.Compose([individual_transform, ToTensorV2()])
        transformed_image = transform_compose(image=image_np)['image']
        
        # Convert to a format suitable for saving
        transformed_image_np = transformed_image.permute(1, 2, 0).numpy()
        transformed_image_np = transformed_image_np * 255
        transformed_image_np = transformed_image_np.astype(np.uint8)
        
        # Resize transformed image to match the original image size
        transformed_image_np = resize_image(transformed_image_np, original_image_np.shape[:2])
        
        # Concatenate original and transformed images side by side
        concatenated_image = np.concatenate((original_image_np, transformed_image_np), axis=1)

        # Save the concatenated image
        transform_name = type(individual_transform).__name__
        transformed_image_path = os.path.join(save_dir, f'{transform_name}.jpg')
        cv2.imwrite(transformed_image_path, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

    # Apply all transformations sequentially and save the final image
    all_transforms_compose = A.Compose(transform_list + [ToTensorV2()])
    transformed_image = all_transforms_compose(image=image_np)['image']
    
    # Convert to a format suitable for saving
    transformed_image_np = transformed_image.permute(1, 2, 0).numpy()
    transformed_image_np = transformed_image_np * 255
    transformed_image_np = transformed_image_np.astype(np.uint8)
    
    # Resize transformed image to match the original image size
    transformed_image_np = resize_image(transformed_image_np, original_image_np.shape[:2])

    # Concatenate original and fully transformed images side by side
    concatenated_image_all = np.concatenate((original_image_np, transformed_image_np), axis=1)

    # Save the concatenated image with all transformations
    all_transformed_image_path = os.path.join(save_dir, 'all_transformations.jpg')
    cv2.imwrite(all_transformed_image_path, cv2.cvtColor(concatenated_image_all, cv2.COLOR_RGB2BGR))

# Example usage with the transformations split from Compose
transform_list = [
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomRotate90(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.GaussianBlur(p=1.0)
]

save_transformations(train_dataset, transform_list)

def save_random_image_from_loader(data_loader, save_dir='img'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Select a random batch from the data loader
    random_batch_idx = random.randint(0, len(data_loader) - 1)
    for i, (images, targets) in enumerate(data_loader):
        if i == random_batch_idx:
            break

    # Select a random image from the batch
    random_image_idx = random.randint(0, len(images) - 1)
    image = images[random_image_idx]
    image_np = image.astype(np.uint8)

    # Save the random image
    random_image_path = os.path.join(save_dir, 'random_image_from_loader.jpg')
    cv2.imwrite(random_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


train_dataset_all_transforms = BoneFractureDataset(image_dir='do1/train/images', label_dir='do1/train/labels', transform=rand_transform)
train_loader_all_transforms = DataLoader(train_dataset_all_transforms, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
save_random_image_from_loader(train_loader_all_transforms)
