import torch
import timm
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from packaging import version

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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
# 创建数据加载器
def create_dataloaders(root_dir, batch_size=32):
    train_dataset = AerialDataset(root_dir=root_dir, transform=train_transform, mode='train')
    val_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='val')
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='test')

    print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

    # Windows环境下将num_workers设为0以避免线程错误
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes

# 创建ViT模型
def create_vit_model(num_classes=15, compile_model=True):
    import torch  # 添加在函数内部导入torch
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # 检查是否支持torch.compile并应用
    if compile_model and version.parse(torch.__version__) >= version.parse('2.0.0'):
        try:
            print("尝试使用torch.compile加速模型训练...")
            # 设置错误抑制，允许在编译失败时自动回退
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            # 使用默认模式编译模型
            model = torch.compile(model)
            print("模型编译成功")
        except Exception as e:
            print(f"torch.compile失败，回退到标准执行模式: {str(e)}")
    elif compile_model:
        print(f"当前PyTorch版本 {torch.__version__} 不支持torch.compile，需要2.0.0或更高版本")
        
    return model

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}, PyTorch版本: {torch.__version__}")
    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': [],
        'batch_losses': []  # 添加记录每个batch的损失
    }

    best_val_acc = 0.0
    
    # 创建实时绘图窗口
    plt.figure(figsize=(10, 5))
    plt.title('实时Batch损失曲线')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    line, = plt.plot([], [], 'b-')
    batch_losses = []
    batch_indices = []
    total_batch = 0
    
    # 设置图表可交互模式
    plt.ion()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            # 记录当前batch的损失并更新图表
            batch_losses.append(loss.item())
            batch_indices.append(total_batch)
            total_batch += 1
            
            # 更新损失图表
            line.set_data(batch_indices, batch_losses)
            plt.xlim(0, max(1, max(batch_indices)))
            plt.ylim(0, max(1, max(batch_losses) * 1.1))
            plt.draw()
            plt.pause(0.01)  # 短暂停顿以更新图表
            
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # 保存每个epoch结束时的所有batch损失
        history['batch_losses'].extend(batch_losses)

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
    
    # 关闭交互模式
    plt.ioff()
    
    # 保存最终的batch loss曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(history['batch_losses'])), history['batch_losses'])
    plt.title('所有Batch的损失曲线')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('batch_loss_curve.png')
    plt.close()

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
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    
    # 绘制每个batch的损失曲线
    plt.subplot(2, 1, 2)
    plt.plot(range(len(history['batch_losses'])), history['batch_losses'])
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss per Batch')
    plt.grid(True)

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
    batch_size = 16  # 减小批次大小
    num_epochs = 5
    learning_rate = 0.0001
    use_compile = True  # 新增参数，控制是否使用torch.compile

    try:
        # 创建数据加载器，降低worker数量
        print("正在加载数据...")
        train_loader, val_loader, test_loader, classes = create_dataloaders(
            data_root, batch_size=batch_size)
        print(f"数据加载完成。类别数: {len(classes)}, 类别: {classes}")

        # 创建模型
        print("正在创建模型...")
        model = create_vit_model(num_classes=len(classes), compile_model=use_compile)

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

