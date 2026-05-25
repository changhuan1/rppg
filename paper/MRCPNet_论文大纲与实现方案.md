# RT-MRCPNet 论文大纲与模型实现方案

## 1. 拟定题目

### 中文题目

面向非接触式压力识别的区域-时间注意力多区域颜色脉动网络

### 英文题目

Region-Temporal Attention Multi-Region Color Pulse Network for Contactless Stress Recognition

### 方法简称

RT-MRCPNet: Region-Temporal Attention Multi-Region Color Pulse Network

---

## 2. 核心定位

本文关注普通 RGB 人脸视频中的微弱颜色时序变化。不同于直接使用整帧视频进行压力识别，也不同于先显式提取心率或心率变异性特征再分类的方法，本文将人脸划分为多个稳定区域，并将每个区域的颜色变化序列作为建模对象。

核心思想是：

> 将人脸视频压缩为多个面部区域的 RGB 颜色脉动序列，通过区域-时间双注意力机制学习关键时间片段和关键面部区域，实现非接触式压力状态识别。

本文不强调图结构，不强调传统手工 HRV 特征，也不把方法写成某个已有框架的扩展。论文主线应当突出：

- 输入轻量：不直接处理完整视频帧，而是处理区域级颜色时序。
- 建模明确：同时建模区域内关键时间片段和区域间贡献差异。
- 任务专注：面向 stress / non-stress 非接触识别。
- 实现务实：模型小、训练快、易于复现实验。

---

## 3. 摘要草稿

非接触式压力识别旨在从普通摄像头视频中判断个体的压力状态，在远程健康监测、人机交互和在线评估等场景中具有重要应用价值。现有方法通常依赖整帧视频建模或先提取远程生理指标再进行分类，前者计算开销较大且容易受到背景与头部运动干扰，后者则依赖人工设计的信号处理流程，难以充分利用面部局部区域的时序差异。

为此，本文提出一种区域-时间注意力多区域颜色脉动网络 RT-MRCPNet。该方法首先将人脸划分为多个稳定区域，并提取每个区域的 RGB 均值序列，形成紧凑的多区域颜色脉动表示。随后，模型通过共享时序编码器学习局部颜色动态，通过时间注意力模块强调每个区域中更有效的颜色变化片段，并通过区域注意力模块自适应融合不同面部区域的信息，最终输出压力状态预测。相比整帧视频模型，RT-MRCPNet 显著降低了输入冗余；相比传统特征工程方法，RT-MRCPNet 能够端到端学习与压力状态相关的区域级时序模式。

在社会压力场景数据集上的实验表明，RT-MRCPNet 在 subject-independent 设置下优于全脸颜色序列、单区域序列、传统 rPPG 特征分类方法以及若干视频时序基线。消融实验进一步验证了多区域表示、时间注意力和区域注意力的有效性。

---

## 4. 论文贡献点

### 贡献 1：多区域颜色脉动表示

提出一种面向非接触压力识别的多区域颜色脉动表示。该表示将人脸视频转化为多个面部区域的 RGB 时序序列，在保留微弱颜色动态信息的同时减少整帧视频中的冗余背景和空间噪声。

### 贡献 2：区域-时间双注意力机制

设计一种轻量级区域-时间双注意力网络 RT-MRCPNet，在时间维度上自适应强调更有效的颜色动态片段，在区域维度上自适应选择更具判别性的面部区域，从而提升非接触压力识别的鲁棒性。

### 贡献 3：完整的压力识别评估

在社会压力任务数据上进行 subject-independent 评估，和传统 rPPG 特征方法、全脸颜色序列方法、单区域方法、简单多区域拼接方法以及视频时序网络进行对比，并通过消融实验验证各模块作用。

---

## 5. 文章结构大纲

## 5.1 Introduction

### 第一段：任务背景

压力状态识别在人机交互、远程心理健康、在线面试、驾驶监测和数字医疗中具有实际意义。传统压力识别常依赖 ECG、EDA、PPG 等接触式传感器，虽然信号质量较高，但在远程场景和自然交互场景中使用不便。

### 第二段：非接触式视频压力识别的价值

普通摄像头能够捕捉面部外观变化、运动变化以及微弱颜色变化，因此为非接触式压力识别提供了可能。相比接触式设备，视频方法更自然、成本更低，也更适合线上交互场景。

### 第三段：现有方法问题

现有方法主要有两类：

1. 整帧视频建模方法：直接输入人脸视频帧，模型可能学习到表情、头动、任务动作等非目标因素，计算开销也较大。
2. 传统生理特征方法：先提取心率、心率变异性等指标，再使用分类器识别压力，但依赖人工信号处理流程，对局部区域差异利用不足。

### 第四段：本文观察

人脸不同区域的颜色时序变化并不完全相同。额头、脸颊、鼻部和下巴等区域受到光照、运动、遮挡和皮肤反射的影响不同。将这些区域混合为一个全脸平均序列可能会损失局部差异，也可能引入低质量区域的干扰。

### 第五段：本文方法

本文提出 RT-MRCPNet。方法将人脸视频转化为多个面部区域的 RGB 时序序列，并使用轻量时序网络进行压力识别。模型由多区域颜色脉动表示、区域内时序编码、时间注意力和区域注意力四部分组成。

### 第六段：贡献总结

列出三条贡献点。

---

## 5.2 Related Work

### 5.2.1 Contact-based Stress Recognition

介绍 ECG、EDA、PPG、呼吸等接触式压力识别方法。强调这类方法信号可靠，但需要佩戴设备，不适合远程自然场景。

可写重点：

- ECG/PPG 用于 HR 和 HRV。
- EDA 与压力状态相关。
- 接触式方法在实验环境中表现好，但部署成本较高。

### 5.2.2 Video-based Stress Recognition

介绍基于人脸视频的压力识别方法，包括表情、头部运动、眼动、颜色变化等线索。

需要强调：

- 整帧视频方法容易受表情和动作混杂影响。
- 视频模型计算成本较高。
- 小数据压力场景下，大模型容易过拟合。

### 5.2.3 Remote Physiological Measurement from Face Videos

介绍从人脸视频中捕捉微弱颜色变化的研究，包括传统 rPPG 方法和深度学习方法。

写法注意：

- 不要把本文写成传统 rPPG 方法。
- 可以说本文受到区域颜色时序建模的启发，但目标是压力识别。

### 5.2.4 Multi-region Facial Representation

介绍多区域人脸表示的必要性。不同区域在光照、运动、遮挡、皮肤纹理方面存在差异，多区域表示有助于保留局部动态。

本文区别：

- 不是做图模型。
- 不是只做心率估计。
- 是用多区域颜色时序直接服务于压力分类。

---

## 5.3 Method

## 5.3.1 Problem Definition

给定一个人脸视频片段：

```text
V = {I_1, I_2, ..., I_T}
```

其中 `T` 是视频片段长度，`I_t` 是第 `t` 帧 RGB 图像。目标是预测该片段的压力标签：

```text
y ∈ {0, 1}
```

其中：

```text
0 = non-stress
1 = stress
```

对于 UBFC-Phys 类型的任务，可以定义：

```text
T1 → non-stress
T2, T3 → stress
```

模型学习函数：

```text
f(V) → y
```

但本文不直接输入完整视频帧，而是先将视频转化为多区域颜色脉动序列：

```text
X ∈ R^{K × T × C}
```

其中：

- `K` 是面部区域数量，例如 5。
- `T` 是 clip 长度。
- `C = 3`，对应 RGB 三通道。

---

## 5.3.2 Multi-Region Color Pulse Representation

### 面部区域划分

对每一帧检测人脸区域，并根据人脸框划分固定比例 ROI。

推荐第一版使用 5 个区域：

```text
ROI 1: forehead
ROI 2: left cheek
ROI 3: right cheek
ROI 4: nose
ROI 5: chin
```

如果暂时不使用 landmark，可以用人脸框内的相对位置近似划分：

```text
face box = [x, y, w, h]

forehead:
  x + 0.25w : x + 0.75w
  y + 0.08h : y + 0.25h

left cheek:
  x + 0.12w : x + 0.38w
  y + 0.38h : y + 0.68h

right cheek:
  x + 0.62w : x + 0.88w
  y + 0.38h : y + 0.68h

nose:
  x + 0.40w : x + 0.60w
  y + 0.32h : y + 0.62h

chin:
  x + 0.35w : x + 0.65w
  y + 0.68h : y + 0.88h
```

后续如果时间允许，可以升级为 landmark-based ROI。

### 区域颜色序列提取

对第 `k` 个 ROI，在第 `t` 帧中计算 RGB 均值：

```text
x_{k,t} = mean_RGB(ROI_k(I_t))
```

得到：

```text
X_k = [x_{k,1}, x_{k,2}, ..., x_{k,T}]
```

所有区域组合为：

```text
X = [X_1, X_2, ..., X_K]
X ∈ R^{K × T × 3}
```

### 序列归一化

对每个 ROI 的每个颜色通道做 temporal normalization：

```text
X'_{k,:,c} = (X_{k,:,c} - mean(X_{k,:,c})) / (std(X_{k,:,c}) + eps)
```

可选增强：

```text
diff sequence:
D_{k,t,c} = X'_{k,t,c} - X'_{k,t-1,c}
```

最终输入可以是：

```text
RGB only:        K × T × 3
RGB + Diff RGB:  K × T × 6
```

建议第一版使用：

```text
K = 5
T = 160
C = 3
```

---

## 5.3.3 RT-MRCPNet Overall Architecture

RT-MRCPNet 包含五个部分：

```text
Multi-region color sequence
→ ROI-wise temporal encoder
→ Temporal attention module
→ Region attention module
→ Stress classifier
```

输入：

```text
X ∈ R^{B × K × T × C}
```

输出：

```text
logits ∈ R^{B × 2}
```

其中：

- `B`: batch size
- `K`: ROI 数量
- `T`: clip 长度
- `C`: 输入通道数，RGB 时为 3

---

## 5.3.4 ROI-wise Temporal Encoder

每个 ROI 的颜色时序单独编码。

将输入 reshape：

```text
X ∈ R^{B × K × T × C}
→ X_flat ∈ R^{(B*K) × C × T}
```

使用轻量 1D temporal encoder：

```text
Conv1d(C, 32, kernel_size=5, padding=2)
BatchNorm1d
ReLU
Dropout

Conv1d(32, 64, kernel_size=5, padding=2)
BatchNorm1d
ReLU
Dropout

Conv1d(64, 128, kernel_size=3, padding=1)
BatchNorm1d
ReLU
```

得到：

```text
H_flat ∈ R^{(B*K) × 128 × T}
```

然后做 temporal pooling：

```text
H_seq = permute(H_flat)  # R^{(B*K) × T × 128}
```

得到每个 ROI 的向量：

```text
H_seq ∈ R^{(B*K) × T × D}
```

其中：

```text
D = 128
```

这里保留时间维度，用于后续时间注意力建模。

---

## 5.3.5 Temporal Attention Module

时间注意力模块用于在每个 ROI 内部选择更有效的时间片段。视频片段中可能出现眨眼、头部轻微运动、局部遮挡或光照波动，因此并不是所有时间点都同等可靠。直接对时间维度做平均池化会把有效片段和干扰片段混在一起，而时间注意力可以让模型自动关注更有判别力的颜色动态。

对于第 `k` 个 ROI 的时序特征：

```text
H_k = [h_{k,1}, h_{k,2}, ..., h_{k,T}]
H_k ∈ R^{T × D}
```

计算时间注意力分数：

```text
e_{k,t} = w_t^T tanh(W_t h_{k,t})
```

归一化得到时间权重：

```text
α_{k,t} = softmax(e_{k,t})
```

加权汇聚得到 ROI 级表示：

```text
r_k = Σ_t α_{k,t} h_{k,t}
```

所有 ROI 得到：

```text
R ∈ R^{B × K × D}
```

时间注意力的作用：

- 降低无效时间片段的干扰。
- 强调更稳定、更有判别力的颜色变化片段。
- 提供可视化解释，展示模型关注的视频时间位置。

---

## 5.3.6 Region Attention Module

目标是学习不同 ROI 对压力识别的贡献。时间注意力已经为每个 ROI 生成一个区域级表示 `r_k`，区域注意力进一步判断哪些面部区域在当前视频片段中更可靠、更有判别力。

对每个 ROI 特征 `r_k` 计算权重：

```text
β_k = softmax(W_2 ReLU(W_1 r_k))
```

其中：

```text
β ∈ R^{B × K}
```

区域融合：

```text
z = Σ_k β_k r_k
```

得到视频片段级表示：

```text
z ∈ R^{B × D}
```

为了稳定训练，可以加入 residual fusion：

```text
z_avg = mean(R, dim=K)
z_final = z + z_avg
```

这样即使区域注意力初期不稳定，模型仍然能利用所有区域。

区域注意力的作用：

- 自适应选择更可靠的面部区域。
- 减少低质量 ROI 的影响，例如嘴部动作影响下的下巴区域。
- 提供可解释的区域权重，可视化模型在不同压力状态下依赖的面部区域。

---

## 5.3.7 Stress Classification Head

分类头：

```text
Linear(D, 128)
ReLU
Dropout(0.5)
Linear(128, 2)
```

输出：

```text
logits = [logit_non_stress, logit_stress]
```

训练损失：

```text
L_cls = CrossEntropyLoss(logits, y)
```

如果类别不均衡，可以使用 class weight。

---

## 5.3.8 Optional: Attention Regularization

如果训练中发现注意力过度集中在单一区域，可以加入轻量正则项，但第一版不建议默认使用。

可选区域注意力熵正则：

```text
L_ent = - Σ_k β_k log(β_k)
```

如果希望避免注意力塌缩，可以最大化熵，或者只在训练后期加入很小权重。第一版建议保持简单，只使用分类损失。

---

## 5.3.9 Optional: Auxiliary Color Reconstruction

如果希望加一点额外新意但不想引入 BVP/EDA，可以加入轻量自监督辅助任务：

```text
z → reconstruct normalized global color sequence summary
```

但十天内不建议第一版做这个。主线保持简洁更重要。

---

## 5.3.10 Final Loss

第一版只使用分类损失：

```text
L = L_cls
```

如果后续加入辅助分支：

```text
L = L_cls + λ L_aux
```

建议第一版先不要加辅助分支，避免方法叙事变散。

---

## 6. 输入输出设计

## 6.1 Dataset 输出

每个样本输出：

```python
data, label, filename, chunk_id
```

其中：

```python
data.shape = [K, T, C]
label.shape = []
```

推荐：

```python
K = 5
T = 160
C = 3
```

如果使用 `DataLoader` 后：

```python
batch_data.shape = [B, K, T, C]
batch_label.shape = [B]
```

## 6.2 模型输入

```python
inputs: torch.Tensor
shape: [B, K, T, C]
dtype: float32
```

## 6.3 模型输出

```python
logits: torch.Tensor
shape: [B, 2]
```

为了支持可视化，模型还返回两类注意力权重：

```python
region_attn.shape = [B, K]
temp_attn.shape = [B, K, T]
```

推理：

```python
pred = torch.argmax(logits, dim=1)
```

## 6.4 指标输出

必须报告：

```text
Accuracy
Precision
Recall
F1-score
AUC
Confusion Matrix
```

如果类别不均衡，F1 和 AUC 比 Accuracy 更重要。

---

## 7. 代码实现方案

## 7.1 新增文件建议

```text
dataset/data_loader/UBFCPHYSStressRGBLoader.py
neural_methods/model/RT_MRCPNet.py
neural_methods/trainer/RT_MRCPNetTrainer.py
evaluation/stress_metrics.py
configs/train_configs/UBFCPHYS_RT_MRCPNET.yaml
```

需要注册的位置：

```text
main.py
dataset/data_loader/__init__.py
neural_methods/trainer/__init__.py
```

论文正文不要描述工程来源，只描述方法和实验设置。

---

## 7.2 UBFCPHYSStressRGBLoader 设计

### 功能

读取 UBFC-Phys 视频，将每个视频切分为 clip，并为每个 clip 生成多 ROI RGB 时序。

### 原始数据假设

```text
RawData/
  s1/
    vid_s1_T1.avi
    vid_s1_T2.avi
    vid_s1_T3.avi
    bvp_s1_T1.csv
    bvp_s1_T2.csv
    bvp_s1_T3.csv
  s2/
    ...
```

### 标签规则

第一版二分类：

```python
if task == "T1":
    label = 0
else:
    label = 1
```

可选三分类：

```python
T1 = 0
T2 = 1
T3 = 2
```

建议论文主实验先做二分类，附加实验可以做三分类。

### 预处理流程

```text
1. 读取视频帧
2. 检测人脸框
3. 根据人脸框切分 5 个 ROI
4. 计算每个 ROI 的 RGB 均值
5. 得到 K × T_video × 3 序列
6. 按 CHUNK_LENGTH 切成 clip
7. 保存为 .npy
```

### 保存格式

输入文件：

```python
input_clip.shape = [K, T, 3]
```

标签文件：

```python
label = 0 or 1
```

为了兼容原有 `__getitem__` 风格，label 可以保存成：

```python
np.array(label)
```

---

## 7.3 RT_MRCPNet.py 伪代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        # h: [B*K, T, D]
        score = self.attn(h).squeeze(-1)      # [B*K, T]
        weight = torch.softmax(score, dim=1)  # [B*K, T]
        out = torch.sum(h * weight.unsqueeze(-1), dim=1)
        return out, weight


class RegionAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        # h: [B, K, D]
        score = self.attn(h).squeeze(-1)      # [B, K]
        weight = torch.softmax(score, dim=1)  # [B, K]
        out = torch.sum(h * weight.unsqueeze(-1), dim=1)
        return out, weight


class RTMRCPNet(nn.Module):
    def __init__(self, num_rois=5, in_channels=3, hidden_dim=128, num_classes=2):
        super().__init__()
        self.num_rois = num_rois

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.temporal_attn = TemporalAttention(hidden_dim)
        self.region_attn = RegionAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, K, T, C]
        B, K, T, C = x.shape

        # [B, K, T, C] -> [B*K, C, T]
        x = x.reshape(B * K, T, C).permute(0, 2, 1)

        h = self.temporal_encoder(x)       # [B*K, D, T]
        h = h.permute(0, 2, 1)             # [B*K, T, D]

        h, temp_attn = self.temporal_attn(h)  # [B*K, D], [B*K, T]
        h = h.reshape(B, K, -1)               # [B, K, D]

        z_attn, region_attn = self.region_attn(h)  # [B, D], [B, K]
        z_avg = torch.mean(h, dim=1)
        z = z_attn + z_avg

        logits = self.classifier(z)

        temp_attn = temp_attn.reshape(B, K, T)
        return logits, region_attn, temp_attn
```

---

## 7.4 RT-MRCPNetTrainer 设计

### 训练流程

```text
for epoch:
  for batch:
    data, labels = batch
    data = data.to(device)
    labels = labels.long().to(device)

    logits, region_attn, temp_attn = model(data)
    loss = CrossEntropyLoss(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 验证流程

保存：

```text
y_true
y_pred
y_score
```

其中：

```python
y_score = softmax(logits)[:, 1]
```

计算：

```text
ACC
F1
AUC
Precision
Recall
```

### 保存最佳模型

建议按 validation F1 保存：

```text
best_model = max(valid_f1)
```

如果没有 validation set，就在 subject-independent split 中使用固定 train/test，不要用 test 选择模型。

---

## 8. 实验设计

## 8.1 数据集设置

### 主数据集

UBFC-Phys 类型社会压力数据。

任务设置：

```text
T1: non-stress
T2/T3: stress
```

### 评估协议

必须使用 subject-independent 协议，避免同一个人的 clip 同时出现在训练和测试中。

推荐协议：

```text
Protocol A: 70% subjects train, 10% validation, 20% test
Protocol B: 5-fold subject-independent cross validation
Protocol C: leave-one-subject-out, 如果时间允许
```

十天内建议：

```text
5-fold subject-independent cross validation
```

如果时间紧：

```text
固定 subject split + 3 次随机种子
```

注意：

不要做 clip-level random split。这样会造成 subject leakage，审稿人很容易质疑。

---

## 8.2 对比方法

### Baseline 1: Full-face RGB Temporal Network

只提取全脸 RGB 均值序列：

```text
input: 1 × T × 3
model: same temporal encoder
```

目的：

验证多区域表示是否优于全脸平均。

### Baseline 2: Single ROI Models

分别只使用单个 ROI：

```text
forehead only
left cheek only
right cheek only
nose only
chin only
```

目的：

验证不同区域表现差异，并说明区域融合必要性。

### Baseline 3: Multi-ROI Concatenation

将所有 ROI 特征直接 flatten 或平均，不使用 region attention：

```text
H_concat = concat(h_1, h_2, ..., h_K)
classifier(H_concat)
```

或者：

```text
H_avg = mean(H, dim=K)
classifier(H_avg)
```

目的：

证明区域注意力模块有效。

### Baseline 4: Traditional rPPG Features + SVM

流程：

```text
full face RGB sequence
→ POS/CHROM
→ HR/HRV features
→ SVM / RandomForest
```

目的：

和传统手工特征方案对比。

### Baseline 5: Video CNN/3D CNN

如果时间允许，可以加一个轻量视频模型：

```text
face clip
→ 3D CNN / CNN-LSTM
→ stress classifier
```

目的：

说明 RT-MRCPNet 输入更轻量，性能仍有竞争力。

### Baseline 6: Existing rPPG Models with Stress Head

如果工程时间允许，可以将已有 rPPG 模型的输出改成分类：

```text
TSCAN + classifier
PhysNet + classifier
EfficientPhys + classifier
```

目的：

和更强视频生理模型对比。

十天内优先级：

```text
必须做:
1. Full-face RGB temporal
2. Single ROI
3. Multi-ROI average
4. RT-MRCPNet
5. Traditional rPPG + SVM

有时间再做:
6. TSCAN/PhysNet stress head
7. 3D CNN
```

---

## 8.3 消融实验

### Ablation 1: Number of ROIs

比较：

```text
K = 1
K = 3
K = 5
K = 7
```

目的：

验证区域数量影响。

### Ablation 2: Region-Temporal Attention

比较：

```text
Average pooling over ROIs
Max pooling over ROIs
Concatenation
Temporal attention only
Region attention only
Region-temporal attention
```

目的：

验证时间注意力和区域注意力是否分别有效，以及二者结合是否带来进一步提升。

### Ablation 3: Input Channel

比较：

```text
RGB
Normalized RGB
RGB + temporal difference
Green channel only
```

目的：

说明颜色时序表示选择的合理性。

### Ablation 4: Clip Length

比较：

```text
T = 80
T = 160
T = 240
```

目的：

分析时间窗口长度对压力识别的影响。

### Ablation 5: Temporal Encoder

比较：

```text
1D CNN
BiLSTM
Temporal Transformer
```

建议主方法用 1D CNN，因为稳定、轻量、训练快。

---

## 8.4 可视化实验

### Visualization 1: Region Attention Weights

展示不同任务下 ROI 权重：

```text
T1 non-stress
T2 stress
T3 stress
```

图形式：

```text
bar chart
face heatmap
```

目的：

增强方法可解释性。

### Visualization 2: Feature Distribution

使用 t-SNE 或 UMAP 可视化：

```text
full-face baseline feature
RT-MRCPNet feature
```

目的：

展示 RT-MRCPNet 学到更可分的表示。

### Visualization 3: Confusion Matrix

展示：

```text
non-stress vs stress
```

目的：

分析错误类型。

---

## 9. 结果表格模板

## 9.1 主结果表

| Method | Input | ACC | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| Full-face RGB + 1D CNN | 1 ROI RGB seq |  |  |  |  |  |
| Forehead only | 1 ROI RGB seq |  |  |  |  |  |
| Left cheek only | 1 ROI RGB seq |  |  |  |  |  |
| Right cheek only | 1 ROI RGB seq |  |  |  |  |  |
| Multi-ROI Average | 5 ROI RGB seq |  |  |  |  |  |
| Multi-ROI Concat | 5 ROI RGB seq |  |  |  |  |  |
| rPPG Features + SVM | hand-crafted |  |  |  |  |  |
| RT-MRCPNet | 5 ROI RGB seq |  |  |  |  |  |

## 9.2 消融表

| Variant | ACC | F1 | AUC |
|---|---:|---:|---:|
| RT-MRCPNet w/o attention |  |  |  |
| RT-MRCPNet w/ temporal attention only |  |  |  |
| RT-MRCPNet w/ region attention only |  |  |  |
| RT-MRCPNet w/ RGB + Diff |  |  |  |
| RT-MRCPNet full |  |  |  |

## 9.3 复杂度表

| Method | Params | FLOPs | Input Size | Inference Time |
|---|---:|---:|---:|---:|
| 3D CNN |  |  |  |  |
| Full-frame CNN-LSTM |  |  |  |  |
| RT-MRCPNet |  |  |  |  |

---

## 10. 论文方法部分可直接写的版本

### 10.1 Multi-Region Color Pulse Extraction

Given a facial video clip, we first detect the face region and divide it into multiple predefined facial regions, including the forehead, left cheek, right cheek, nose, and chin. For each region, we compute the average RGB value at each frame, resulting in a compact temporal sequence. Compared with raw video frames, this representation removes most background and texture redundancy while retaining subtle color fluctuations over time.

### 10.2 ROI-wise Temporal Modeling

Each facial region is treated as an independent temporal observation. We apply a shared temporal encoder to all ROI sequences. The shared encoder ensures that all regions are processed with the same temporal filters and reduces the number of parameters. The encoder consists of several 1D convolutional layers followed by normalization and nonlinear activation.

### 10.3 Region-Temporal Attention Modeling

We further introduce a region-temporal attention module to model both temporal reliability and regional contribution. For each ROI sequence, temporal attention assigns larger weights to informative time positions and suppresses unstable segments caused by motion or illumination changes. After temporal aggregation, region attention estimates the contribution of each facial region and adaptively fuses ROI-level features. The final clip-level representation is obtained by combining attention-based aggregation with average ROI aggregation for stable training.

### 10.4 Stress Classification

The aggregated representation is fed into a lightweight classification head to predict the stress state of the input clip. The whole model is trained using cross-entropy loss under a subject-independent protocol.

---

## 11. 十天实施计划

## Day 1: 数据读取和标签生成

目标：

```text
完成 UBFCPHYSStressRGBLoader 的 raw data scan。
确认 T1/T2/T3 标签。
输出 metadata。
```

产物：

```text
subject, task, video_path, label
```

## Day 2: ROI RGB 序列提取

目标：

```text
读取视频
检测人脸
划分 5 个 ROI
保存 K × T × 3 clip
```

产物：

```text
*_input.npy
*_label.npy
```

## Day 3: Dataset Loader 跑通

目标：

```text
DataLoader 能输出 [B, K, T, C]
label 能输出 [B]
```

检查：

```text
打印 batch shape
可视化 ROI 位置
检查 label 分布
```

## Day 4: RT-MRCPNet 模型实现

目标：

```text
实现 RT_MRCPNet.py
输入 [B, K, T, C]
输出 logits、region_attn 和 temp_attn
```

检查：

```text
随机 tensor forward
loss backward
```

## Day 5: Trainer 和 Metrics

目标：

```text
实现训练、验证、测试
实现 ACC/F1/AUC
保存 best model
```

## Day 6: 主实验初跑

目标：

```text
RT-MRCPNet 跑出第一版结果
Full-face baseline 跑出结果
Multi-ROI average 跑出结果
```

## Day 7: 补 baseline

目标：

```text
Single ROI baseline
rPPG features + SVM baseline
Multi-ROI concat baseline
```

## Day 8: 消融和可视化

目标：

```text
ROI attention 可视化
混淆矩阵
t-SNE/UMAP
```

## Day 9: 写论文初稿

目标：

```text
Introduction
Related Work
Method
Experiments
```

## Day 10: 补实验和润色

目标：

```text
补表格
检查 subject split
润色贡献点
整理代码和结果
```

---

## 12. 风险和规避

## 风险 1：模型效果不如传统特征

解决：

```text
加入 RGB + Diff RGB 输入
增加 clip length
使用 5-fold subject-independent 取平均
加入 class weight
```

## 风险 2：ROI 检测不稳定

解决：

```text
先用人脸框固定比例 ROI
对人脸框做 temporal smoothing
如果某帧检测失败，使用上一帧框
```

## 风险 3：数据量小，模型过拟合

解决：

```text
模型保持小
Dropout = 0.3 到 0.5
weight decay = 1e-4
early stopping
subject-independent split
```

## 风险 4：审稿人觉得只是简单注意力

解决：

论文强调：

```text
紧凑颜色脉动表示
区域内时序建模
时间注意力和区域注意力自适应融合
轻量、可解释、适合小数据压力识别
```

实验补充：

```text
复杂度对比
单 ROI 对比
全脸对比
concat/average/attention 对比
```

---

## 13. 推荐超参数

```yaml
MODEL:
  NAME: RTMRCPNet
  NUM_ROIS: 5
  INPUT_CHANNELS: 3
  HIDDEN_DIM: 128
  NUM_CLASSES: 2
  DROPOUT: 0.5

TRAIN:
  BATCH_SIZE: 64
  EPOCHS: 50
  LR: 1e-3
  WEIGHT_DECAY: 1e-4
  OPTIMIZER: AdamW
  LOSS: CrossEntropy

DATA:
  FS: 30
  CHUNK_LENGTH: 160
  DATA_FORMAT: KTC
  NUM_ROIS: 5

TEST:
  METRICS: ['ACC', 'Precision', 'Recall', 'F1', 'AUC']
```

如果显存紧张：

```text
BATCH_SIZE = 32
```

如果过拟合：

```text
Dropout = 0.6
HIDDEN_DIM = 64
```

---

## 14. 最终建议

这篇文章不要写成“大而全”的压力识别系统。最稳的写法是：

> 我们提出一种轻量的区域-时间注意力多区域颜色脉动网络，将人脸视频转化为紧凑的区域级 RGB 时序表示，并通过时间注意力和区域注意力建模关键颜色动态与关键面部区域，实现非接触式压力识别。

方法要保持简单，但实验要完整。只要 subject-independent 协议、对比实验和消融实验做扎实，这个方向就有投稿价值。
