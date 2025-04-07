import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
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

# 数据预处理
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建测试数据加载器
def create_test_dataloader(root_dir, batch_size=32):
    test_dataset = AerialDataset(root_dir=root_dir, transform=val_test_transform, mode='test')
    print(f"测试集大小: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    return test_loader, test_dataset.classes

# 创建ViT模型
def create_vit_model(num_classes=15, compile_model=False):
    import torch  # 添加在函数内部导入torch
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # 在测试时通常不需要编译，但提供选项
    if compile_model and version.parse(torch.__version__) >= version.parse('2.0.0'):
        try:
            print("尝试使用torch.compile加速模型...")
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model)
            print("模型编译成功")
        except Exception as e:
            print(f"torch.compile失败，回退到标准执行模式: {str(e)}")
    
    return model

# 测试函数
def test_model(model, test_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
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

    # 检查数据目录和模型文件是否存在
    data_root = 'Aerial_Landscapes'
    model_path = 'best_vit_model.pth'
    batch_size = 16

    if not os.path.exists(data_root):
        print(f"错误: 数据目录 '{data_root}' 不存在!")
        return

    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return

    try:
        # 创建测试数据加载器
        print("正在加载测试数据...")
        test_loader, classes = create_test_dataloader(data_root, batch_size=batch_size)
        print(f"数据加载完成。类别数: {len(classes)}, 类别: {classes}")

        # 创建模型并加载预训练权重
        print("正在加载模型...")
        model = create_vit_model(num_classes=len(classes), compile_model=False)  # 测试阶段一般无需编译
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("模型加载完成")

        # 测试模型
        print("\n在测试集上评估模型...")
        conf_matrix, report, accuracy, y_true, y_pred = test_model(model, test_loader, classes)

        # 打印测试结果
        print(f"测试准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(report)

        # 可视化混淆矩阵
        plot_confusion_matrix(conf_matrix, classes)

        print("评估完成!")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

