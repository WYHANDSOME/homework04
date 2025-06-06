# CIFAR-10 图像分类对抗攻击（PGD）

本项目基于 PyTorch 实现了针对 CIFAR-10 图像分类任务的 **白盒** 和 **黑盒对抗攻击**。攻击算法使用 PGD（Projected Gradient Descent）。目标模型为 **ResNet-20**，黑盒攻击使用 **VGG11** 作为代理模型。

---

## 🛠️ 环境依赖

安装基本依赖：

```bash
pip install torch torchvision matplotlib
````

## 🔧 模型训练

### ✅ 训练 ResNet-20（目标模型）

```bash
python resnet_train.py
```

模型会保存在：

```
D:\neural_network\作业4\Resnet20\resnet20-12fca82f.th
```

### ✅ 训练 VGG11（代理模型）

创建并运行 `vgg11_train.py`，用于黑盒攻击：

```bash
python vgg11_train.py
```

保存路径：

```
D:\neural_network\作业4\VGG11\vgg11_cifar10.pth
```

---

## 🎯 执行攻击

运行主程序：

```bash
python main.py
```

### 白盒攻击（White-box）

对 VGG11 自身进行 PGD 攻击，测试不同 ε 下模型准确率下降情况，并保存：

* 准确率对比图：`whitebox_accuracy_plot.png`
* 对抗样本图示：`adv_examples_eps_8.png` 等

### 黑盒攻击（Black-box）

使用 VGG11 生成对抗样本攻击 ResNet-20（目标模型）

---

## 📊 实验结果

### 白盒攻击（PGD on VGG11）

| ε 值（L∞ 范数） | 模型准确率  |
| ---------- | ------ |
| 0.000      | 89.25% |
| 0.008      | 39.24% |
| 0.016      | 8.26%  |
| 0.024      | 2.50%  |
| 0.031      | 1.83%  |

### 黑盒攻击（PGD Transfer to ResNet-20）

| ε 值（L∞ 范数） | 准确率    |
| ---------- | ------ |
| 0.031      | 44.17% |

---

## 📌 参数说明

可在 `main.py` 中调整攻击强度参数：

```python
epsilons = [0, 2/255, 4/255, 6/255, 8/255]
alpha = 2/255
iters = 10
```

---

## 📷 示例图预览（部分）

| 对抗扰动 ε | 白盒攻击示例图（自动保存）                |
| ------ | ---------------------------- |
| 8/255  | `adv_examples_eps_8.png`     |
| 准确率图   | `whitebox_accuracy_plot.png` |
### 📈 白盒攻击准确率变化图

![Accuracy vs Epsilon](https://github.com/WYHANDSOME/homework04/blob/main/whitebox_accuracy_plot.png)

### 🎯 ε = 8/255 的对抗样本示例

![Adversarial Examples](https://github.com/WYHANDSOME/homework04/blob/main/eps%20examples/adv_examples_eps_8.png)

---
