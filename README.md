# DermaMNIST 皮肤病图像分类项目

本项目基于 [MedMNIST](https://medmnist.com/) 数据集中的 **DermaMNIST** 子集，利用深度学习模型 **ResNet-18** 实现皮肤病病灶的自动化分类。

## 📋 项目简介
DermaMNIST 是一个包含 10,015 张多模态皮肤镜图像的数据集，分为 7 个类别。本项目通过构建一个经过结构优化的 ResNet-18 模型，并结合标签平滑（Label Smoothing）和类别权重平衡等技术，旨在提供一个高性能、可解释性强的分类方案。

## 📁 文件结构
- `Net.py`: 核心脚本，包含数据预处理、模型定义、训练循环及评估逻辑。
- `requirements.txt`: 项目所需的 Python 依赖库列表。
- `best_resnet18_fixed.pth`: 训练过程中保存的最优模型权重。
- `class_distribution.png`: 训练集类别分布直方图。
- `training_history_resnet18.png`: 训练与验证过程中的 Loss 变化曲线。
- `confusion_matrix_resnet18.png`: 混淆矩阵，直观展示模型在各类别间的误诊情况。
- `roc_curve_resnet18.png`: ROC 曲线，展示模型区分正负样本的能力（看左上角）。
- `pr_curve_resnet18.png`: 精确率-召回率曲线，针对不平衡数据的严苛评估（看右上角）。
- `sample_preds_resnet18.png`: 测试集预测结果样例展示。

## 🚀 快速开始

### 1. 安装依赖
建议使用 Python 3.10+ 环境，执行以下命令安装必要库：
```bash
pip install -r requirements.txt
```

### 2. 运行程序
直接运行主脚本即可开始训练（数据会自动下载）：
```bash
python Net.py
```

## 🛠️ 技术亮点
- **模型架构**: 采用 **ResNet-18** 作为主干网络，通过显式的 `features` 和 `classifier` 模块拆解，提升了代码的可读性。
- **损失函数**: 使用带 **Label Smoothing (0.1)** 的交叉熵损失，有效缓解过拟合。
- **数据增强**: 集成了随机翻转、旋转、色彩抖动等增强手段，模拟真实场景下的图像差异。
- **评估体系**: 提供了从混淆矩阵到 ROC/PR 曲线的全方位评估，深入剖析模型在不平衡数据集上的表现。

## 📊 评估结果说明
- **混淆矩阵**: 对角线代表预测正确，非对角线数字越大代表该两个类别越容易混淆。
- **ROC 曲线 (AUC)**: AUC 越接近 1 代表模型整体分类实力越强。
- **PR 曲线**: 在处理皮肤病这种类别极度不平衡的数据时，PR 曲线比 ROC 曲线更能反映模型在少数类别上的真实水准。
