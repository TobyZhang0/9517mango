import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 数据增强策略 ----------------------
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        image = np.array(img)
        augmented = self.transform(image=image)
        return augmented['image']

class KorniaTransform:
    def __init__(self):
        self.transforms = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=15.0)
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])
    def __call__(self, img):
        tensor_img = self.to_tensor(img).unsqueeze(0)  # [1, C, H, W]
        tensor_img = self.transforms(tensor_img)
        tensor_img = tensor_img.squeeze(0)
        tensor_img = self.normalize(tensor_img)
        return tensor_img

def get_train_transform(strategy="baseline"):
    if strategy == "baseline":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    elif strategy == "torchvision":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    elif strategy == "albumentations":
        alb_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])
        return AlbumentationsTransform(alb_transform)
    elif strategy == "kornia":
        return KorniaTransform()
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ---------------------- 数据集 ----------------------
# 实现80-20随机分割（每个类别内部打乱）
class AerialDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.mode = mode
        self.imgs = self._load_images()
    def _load_images(self):
        imgs = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            img_files = sorted(os.listdir(class_path))
            np.random.seed(42)  # 固定随机种子确保可重复性
            np.random.shuffle(img_files)
            total = len(img_files)
            train_count = int(total * 0.8)
            if self.mode == 'train':
                selected_files = img_files[:train_count]
            elif self.mode == 'test':
                selected_files = img_files[train_count:]
            else:
                raise ValueError("mode must be 'train' or 'test'")
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

def create_dataloaders(root_dir, batch_size=32, aug_strategy="baseline"):
    train_dataset = AerialDataset(root_dir=root_dir, transform=get_train_transform(aug_strategy), mode='train')
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='test')
    print(f"数据集大小 - 训练: {len(train_dataset)}, 测试: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_dataset.classes

# ---------------------- 模型及训练 ----------------------
def create_resnet_model(num_classes=15):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=15, aug_strategy="baseline"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0; correct = 0; total = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        model.eval()
        test_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = correct / total
        scheduler.step(test_loss)
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'best_resnet_model_{aug_strategy}.pth')
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    return model, history

def test_model(model, test_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_true = []; y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    accuracy = accuracy_score(y_true, y_pred)
    return conf_matrix, report, accuracy, y_true, y_pred

def plot_training_history(history, aug_strategy):
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training/Test Loss ({aug_strategy})')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training/Test Accuracy ({aug_strategy})')
    plt.legend()
    plt.tight_layout()
    history_filename = f'training_history_{aug_strategy}.png'
    plt.savefig(history_filename)
    print(f"训练历史保存为: {history_filename}")
    plt.close()

def plot_confusion_matrix(conf_matrix, classes, aug_strategy):
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix ({aug_strategy})')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{aug_strategy}.png'
    plt.savefig(cm_filename)
    print(f"混淆矩阵保存为: {cm_filename}")
    plt.close()

# ---------------------- 主函数 ----------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    data_root = 'Aerial_Landscapes'
    if not os.path.exists(data_root):
        print(f"错误: 数据目录 '{data_root}' 不存在!")
        return
    batch_size = 16
    num_epochs = 2
    learning_rate = 0.0001
    strategies = ["baseline", "torchvision", "albumentations", "kornia"]
    results = {}  # 保存各策略评估结果
    for aug_strategy in strategies:
        print(f"\n===== 训练策略: {aug_strategy} =====")
        best_model_path = f'best_resnet_model_{aug_strategy}.pth'
        train_loader, test_loader, classes = create_dataloaders(data_root, batch_size=batch_size, aug_strategy=aug_strategy)
        print(f"数据加载完成。类别数: {len(classes)}, 类别: {classes}")
        if os.path.exists(best_model_path):
            print(f"加载已有的模型 {best_model_path} 进行评估...")
        else:
            print("正在创建并训练模型...")
            model = create_resnet_model(num_classes=len(classes))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, aug_strategy)
            torch.save(trained_model.state_dict(), best_model_path)
            plot_training_history(history, aug_strategy)
        best_model = create_resnet_model(num_classes=len(classes))
        best_model.load_state_dict(torch.load(best_model_path))
        print("在测试集上评估模型...")
        conf_matrix, report, accuracy, y_true, y_pred = test_model(best_model, test_loader, classes)
        results[aug_strategy] = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }
        plot_confusion_matrix(conf_matrix, classes, aug_strategy)
        print(f"策略 {aug_strategy} 的测试准确率: {accuracy:.4f}\n")
    for strategy, metrics in results.items():
        print(f"策略: {strategy} - 测试准确率: {metrics['accuracy']:.4f}")
        print(metrics['classification_report'])
if __name__ == '__main__':
    main()
