import torch
import timm
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 创建ViT模型
def create_vit_model(num_classes=15):
    model = timm.create_model('vit_huge_patch14_224', pretrained=False)  # 不使用预训练权重
    model.head = nn.Linear(model.num_features, num_classes)  # 使用num_features而不是head.in_features
    return model

# 加载本地预训练权重
def load_local_model(model_path, num_classes=15):
    model = create_vit_model(num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # 移除不匹配的头部权重
    state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model

# 定义数据集类，支持数据划分
class AerialDataset(Dataset):
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
                selected_files = img_files[train_size:train_size+val_size]
            else:  # test
                selected_files = img_files[train_size+val_size:]

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

# 数据预处理与增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据加载器
def create_dataloaders(root_dir, batch_size=8):  # 减小批量大小
    train_dataset = AerialDataset(root_dir=root_dir, transform=train_transform, mode='train')
    val_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='val')
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='test')

    print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    # Windows环境下将num_workers设为0以避免线程错误
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

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
            print(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.4f}", end='\r')

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # 学习率调整
        scheduler.step(val_loss)

        # 计算epoch时间
        epoch_time = time.time() - start_time

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Time: {epoch_time:.2f}s')

    return model, history

# 测试函数
def test_model(model, test_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算分类报告
    report = classification_report(y_true, y_pred, target_names=classes)

    # 计算总体准确率
    accuracy = accuracy_score(y_true, y_pred)

    return conf_matrix, report, accuracy, y_true, y_pred

# 可视化训练历史
def plot_training_history(history):
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 可视化混淆矩阵
def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查数据目录是否存在
    data_root = 'Aerial_Landscapes'
    if not os.path.exists(data_root):
        print(f"错误: 数据目录 '{data_root}' 不存在!")
        return

    # 减小批次大小和工作进程数
    batch_size = 16  # 进一步减小批次大小以适应更大的模型
    num_epochs = 3
    learning_rate = 0.0001  # 适当调整学习率

    try:
        # 创建数据加载器，降低worker数量
        print("正在加载数据...")
        train_loader, val_loader, test_loader, classes = create_dataloaders(
            data_root, batch_size=batch_size)
        print(f"数据加载完成。类别数: {len(classes)}, 类别: {classes}")

        # 创建模型
        print("正在创建模型...")
        model_path = 'autodl-tmp/vit-huge-patch14-224-in21k/pytorch_model.bin'
        model = load_local_model(model_path, num_classes=15)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # 训练模型
        print("开始模型训练...")
        trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

        # 可视化训练历史
        plot_training_history(history)

        # 加载最佳模型
        best_model = create_vit_model(num_classes=len(classes))
        best_model.load_state_dict(torch.load('best_vit_model.pth'))

        # 测试模型
        print("\n在测试集上评估模型...")
        conf_matrix, report, accuracy, y_true, y_pred = test_model(best_model, test_loader, classes)

        # 打印测试结果
        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(report)

        # 可视化混淆矩阵
        plot_confusion_matrix(conf_matrix, classes)

        print("训练和评估完成!")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

