import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class AerialDataset(Dataset):
    """Aerial landscape dataset class, supporting training/validation/test data splitting"""
    def __init__(self, root_dir, transform=None, split_ratio=[0.6, 0.2, 0.2], mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.split_ratio = split_ratio
        self.mode = mode
        self.imgs = self._load_images()

    def _load_images(self):
        imgs = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            img_files = sorted(os.listdir(class_path))

            # Calculate the number of images for each set
            total_images = len(img_files)
            train_size = int(total_images * self.split_ratio[0])
            val_size = int(total_images * self.split_ratio[1])

            # Select corresponding images based on mode
            if self.mode == 'train':
                selected_files = img_files[:train_size]
            elif self.mode == 'val':
                selected_files = img_files[train_size:train_size + val_size]
            else:  # test
                selected_files = img_files[train_size + val_size:]

            for img_name in selected_files:
                imgs.append((os.path.join(class_path, img_name), self.class_to_idx[cls]))
        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(augmentation_strategy='default'):
    if augmentation_strategy == 'minimal':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_strategy == 'extensive':
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    elif augmentation_strategy == 'new':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.RandomPerspective(),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform

def create_dataloaders(root_dir, batch_size=32, split_ratio=[0.6, 0.2, 0.2],
                       augmentation_strategy='default', random_seed=42, num_workers=0, verbose=True):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if verbose:
        print(f"Creating data loaders, batch_size={batch_size}, split_ratio={split_ratio}, augmentation={augmentation_strategy}...")

    # Get data preprocessing and augmentation
    train_transform, val_test_transform = get_transforms(augmentation_strategy)
    train_dataset = AerialDataset(root_dir=root_dir, transform=train_transform,
                                  split_ratio=split_ratio, mode='train')
    val_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform,
                                split_ratio=split_ratio, mode='val')
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform,
                                 split_ratio=split_ratio, mode='test')

    if verbose:
        print(f"Dataset size - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Print the data distribution for each class
        class_counts = {}
        for _, label in train_dataset.imgs:
            class_name = train_dataset.classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1

        print("Training set class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"- {cls}: {count} images")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == '__main__':
    # Test code
    data_root = 'Aerial_Landscapes'
    print("Testing data loading module...")

    # Test different data augmentation strategies
    for strategy in ['default', 'minimal', 'extensive']:
        print(f"\n=== Testing augmentation strategy: {strategy} ===")
        train_loader, val_loader, test_loader, classes = create_dataloaders(
            data_root, batch_size=16, augmentation_strategy=strategy
        )
