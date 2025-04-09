import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class AerialDataset(Dataset):
    """空中景观数据集类，支持训练/验证/测试数据划分"""
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

            # 计算每个集合的图像数量
            total_images = len(img_files)
            train_size = int(total_images * self.split_ratio[0])
            val_size = int(total_images * self.split_ratio[1])

            # 根据模式选择相应的图像
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
    """获取数据预处理与增强的transforms

    参数:
    augmentation_strategy: 增强策略，可选 'default', 'minimal', 'extensive'
    """
    if augmentation_strategy == 'minimal':
        # 最小数据增强
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_strategy == 'extensive':
        # 广泛数据增强
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
    else:  # default
        # 默认数据增强
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

    # 验证和测试集的transform保持不变
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def create_dataloaders(root_dir, batch_size=32, split_ratio=[0.6, 0.2, 0.2],
                       augmentation_strategy='default', random_seed=42, num_workers=0, verbose=True):
    """创建数据加载器

    参数:
    root_dir: 数据集根目录
    batch_size: 批次大小
    split_ratio: 训练集、验证集、测试集的划分比例，如[0.6, 0.2, 0.2]
    augmentation_strategy: 数据增强策略，可选 'default', 'minimal', 'extensive'
    random_seed: 随机种子
    num_workers: 数据加载线程数，Windows下建议设为0
    verbose: 是否打印详细信息

    返回:
    train_loader, val_loader, test_loader, classes
    """
    # 设置随机种子
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if verbose:
        print(f"创建数据加载器，batch_size={batch_size}, split_ratio={split_ratio}, augmentation={augmentation_strategy}...")

    # 获取数据预处理与增强
    train_transform, val_test_transform = get_transforms(augmentation_strategy)

    # 创建数据集
    train_dataset = AerialDataset(root_dir=root_dir, transform=train_transform,
                                  split_ratio=split_ratio, mode='train')
    val_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform,
                                split_ratio=split_ratio, mode='val')
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform,
                                 split_ratio=split_ratio, mode='test')

    if verbose:
        print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

        # 打印每个类的数据分布
        class_counts = {}
        for _, label in train_dataset.imgs:
            class_name = train_dataset.classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1

        print("训练集类别分布:")
        for cls, count in sorted(class_counts.items()):
            print(f"- {cls}: {count}张图像")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == '__main__':
    # 测试代码
    data_root = 'Aerial_Landscapes'
    print("测试数据加载模块...")

    # 测试不同的数据增强策略
    for strategy in ['default', 'minimal', 'extensive']:
        print(f"\n=== 测试增强策略: {strategy} ===")
        train_loader, val_loader, test_loader, classes = create_dataloaders(
            data_root, batch_size=16, augmentation_strategy=strategy
        )