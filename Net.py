import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from medmnist import INFO
from medmnist.dataset import DermaMNIST
from collections import Counter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
batch_size = 32          
num_epochs = 50       
lr = 1e-3               
patience = 8            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Outputs will be saved to: {OUTPUT_DIR}")

data_flag = "dermamnist"
info = INFO[data_flag]
n_classes = len(info["label"])

# 增强数据增强以防止过拟合
train_transform = transforms.Compose([
    transforms.Resize(224),  # 适配预训练模型输入尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 标准化
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DermaMNIST(split="train", transform=train_transform, download=True)
val_dataset   = DermaMNIST(split="val",   transform=test_transform,  download=True)
test_dataset  = DermaMNIST(split="test",  transform=test_transform,  download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# 可视化 1: 类别分布 (Class Distribution)
def plot_class_distribution(dataset, title="数据集类别分布"):
    labels = [int(y) for _, y in dataset]
    counts = Counter(labels)
    class_names = [info['label'][str(i)] for i in range(n_classes)]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=[counts[i] for i in range(n_classes)])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("样本数量")
    dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    print(f"📁 类别分布图已保存为: {dist_path}")

plot_class_distribution(train_dataset, "DermaMNIST 训练集类别分布")
labels = [int(y) for _, y in train_dataset]
counts = Counter(labels)
weights = torch.tensor([1.0 / np.sqrt(counts[i]) for i in range(n_classes)], dtype=torch.float).to(device)

# =========================
# 模型定义 (标准 ResNet18 - Class 封装版)
# =========================
class StandardResNet(nn.Module):
    def __init__(self, n_classes):
        super(StandardResNet, self).__init__()

        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        self.in_features = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, n_classes)
        )
        
        self._print_summary()

    def _print_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f" {self.__class__.__name__} 初始化成功 | 总参数量: {total_params:,}")

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

# 实例化模型
model = StandardResNet(n_classes=n_classes).to(device)

# 使用标准的交叉熵损失函数
# 使用 PyTorch 官方推荐的标准损失函数：带标签平滑的交叉熵
# label_smoothing=0.1 可以有效防止模型过拟合，提高泛化能力

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
#AdamW 优化器
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
#余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# =========================
# 训练与验证核心函数 (函数式封装)
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", ncols=100)

    for images, labels in pbar:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")
    return running_loss / len(loader), 100. * correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return val_loss / len(loader), 100. * correct / total

# =========================
# 主训练循环
# =========================
best_val_loss = float("inf")
patience_counter = 0
best_model_path = os.path.join(OUTPUT_DIR, "best_resnet18_fixed.pth")

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\n 开始训练 | 目标路径: {best_model_path}")

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    #记录历史
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"📊 结果: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 5. 早停与模型保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"⭐ 发现更好的模型，已保存至: {best_model_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹ 触发早停机制 (连续 {patience} 个 Epoch 未提升)")
            break

# =========================
# 深度评估与更多可视化
# =========================
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_targets, all_preds, all_probs = [], [], []
sample_images, sample_labels, sample_preds = [], [], []

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        if i == 0:
            sample_images, sample_labels, sample_preds = images.cpu().numpy()[:9], labels.cpu().numpy()[:9], predicted.cpu().numpy()[:9]

all_targets, all_preds, all_probs = np.array(all_targets), np.array(all_preds), np.array(all_probs)

# 1. 训练曲线 (仅 Loss)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Val Loss", linewidth=2)
plt.title("模型训练 Loss 变化曲线", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history_resnet18.png"), dpi=200, bbox_inches='tight')

# 2. 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.title("混淆矩阵")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_resnet18.png"), dpi=200)

# 3. ROC 曲线 (Receiver Operating Characteristic)
plt.figure(figsize=(10, 8))
Y_test_bin = label_binarize(all_targets, classes=range(n_classes))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(Y_test_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'类别 {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("多分类 ROC 曲线")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_resnet18.png"), dpi=200)

# 4. PR 曲线
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(Y_test_bin[:, i], all_probs[:, i])
    ap = average_precision_score(Y_test_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, label=f'类别 {i} (AP = {ap:.2f})')
plt.title("多分类 PR 曲线")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve_resnet18.png"), dpi=200)

# 5. 预测样例展示
plt.figure(figsize=(12, 10))
for i in range(min(9, len(sample_images))):
    plt.subplot(3, 3, i+1)
    img = sample_images[i].transpose(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img)
    color = 'green' if sample_labels[i] == sample_preds[i] else 'red'
    plt.title(f"真值:{sample_labels[i]} 预测:{sample_preds[i]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_preds_resnet18.png"), dpi=200)
plt.show()
