# AIM-main
Self-supervised contrastive learning for label-efficient EEG-based sleep stage classification.
```markdown
# Few-Shot Self-Supervised Learning for Sleep Stage Classification

本项目面向睡眠脑电（EEG）信号分析，基于自监督学习（Self-Supervised Learning, SSL）提取睡眠信号表征，并在有限标注数据条件下进行睡眠分期分类评估。

项目支持以下两个睡眠数据集：

- **Sleep-EDF Cassette**
- **SHHS（Sleep Heart Health Study）**

与常规全监督训练不同，本项目重点关注**少样本（few-shot / low-label）场景**：先利用预训练集中的数据学习通用表征，再仅使用下游训练集中的部分标注样本训练线性分类器。可通过 `--train_ratio` 控制下游任务中实际使用的标注训练数据比例。

---

## 1. Project Structure

```text
.
├── sleepEDF_cassette_process.py   # Sleep-EDF Cassette 数据预处理脚本
├── shhs_process.py                # SHHS 数据预处理脚本
├── self_supservised.py            # 自监督预训练与下游少样本评估主程序
├── model.py                       # 睡眠信号编码器模型定义
├── loss.py                        # 自监督学习损失函数
├── utils.py                       # 数据加载器及辅助工具函数
└── README.md
```

各文件功能如下：

| 文件 | 功能 |
|---|---|
| `sleepEDF_cassette_process.py` | 对 Sleep-EDF Cassette 数据进行读取、切片和划分。 |
| `shhs_process.py` | 对 SHHS 数据进行读取、切片和划分。 |
| `self_supservised.py` | 执行自监督训练，并使用 Logistic Regression 进行下游线性评估。 |
| `model.py` | 定义睡眠 EEG 编码器网络。 |
| `loss.py` | 定义对比学习/自监督训练所需的损失函数。 |
| `utils.py` | 包含数据集加载器、信号处理与其他通用函数。 |

---

## 2. Environment

建议使用 Python 3.8 或更高版本。

主要依赖包括：

```bash
pip install torch torchvision
pip install numpy scipy pandas
pip install scikit-learn
pip install mne
pip install tqdm
```

如使用 GPU，请根据本机 CUDA 版本安装对应版本的 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. Data Preparation

数据预处理流程与参考项目保持一致。原始睡眠数据会被切分为固定长度的睡眠 epoch，并划分为：

- `pretext`：用于自监督预训练；
- `train`：用于下游分类器训练；
- `test`：用于最终测试评估。

默认情况下，每个 epoch 的长度为 **30 秒**，这也是睡眠分期任务中最常见的标注粒度。

> `--windowsize` 表示每个信号片段（epoch）的时长，单位为秒。  
> `--multiprocess` 表示预处理时使用的进程数量。

---

### 3.1 Sleep-EDF Cassette

#### Step 1: Download Sleep-EDF Dataset

从 PhysioNet 下载 Sleep-EDF Expanded 数据集：

<https://physionet.org/content/sleep-edfx/1.0.0/>

本项目使用其中的 **Sleep Cassette** 部分。

```bash
# 创建数据目录
mkdir SLEEP_data
cd SLEEP_data

# 下载 Sleep-EDF Expanded 数据集
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
```

下载完成后，请确认 `sleepEDF_cassette_process.py` 中配置的原始数据路径与本地数据路径一致。

#### Step 2: Preprocess Sleep-EDF Data

运行以下命令进行数据预处理：

```bash
python sleepEDF_cassette_process.py --windowsize 30 --multiprocess 8
```

预处理完成后，数据将按照如下形式划分：

```text
pretext/
train/
test/
```

其中：

- `pretext/` 中的数据用于自监督表征学习；
- `train/` 中的数据用于训练下游分类器；
- `test/` 中的数据用于测试模型性能。

---

### 3.2 SHHS Dataset

#### Step 1: Download SHHS Dataset

从 National Sleep Research Resource（NSRR）下载 SHHS 数据集：

<https://sleepdata.org/datasets/shhs>

> 注意：下载 SHHS 数据通常需要注册账号、申请访问权限并同意数据使用协议。

建议的数据目录组织方式如下：

```bash
mkdir SHHS_data
cd SHHS_data

# 请将下载后的 SHHS 数据放置在当前目录下，
# 并将数据文件夹命名为 SHHS
```

目录示例：

```text
SHHS_data/
└── SHHS/
    ├── edf/
    ├── annotations-events-nsrr/
    └── ...
```

请根据实际下载的数据版本，检查并修改 `shhs_process.py` 中的原始数据路径配置。

#### Step 2: Preprocess SHHS Data

运行以下命令：

```bash
python shhs_process.py --windowsize 30 --multiprocess 8
```

预处理后，同样会生成以下数据划分：

```text
pretext/
train/
test/
```

---

## 4. Self-Supervised Training and Few-Shot Evaluation

项目采用“**自监督预训练 + 线性评估（linear evaluation）**”的范式：

1. 使用 `pretext` 数据训练自监督编码器；
2. 使用训练好的 encoder 提取下游数据特征；
3. 从 `train` 数据中随机选取指定比例的标注样本；
4. 在这些有限标注样本上训练 Logistic Regression 分类器；
5. 在完整 `test` 集上报告分类结果。

### 4.1 Few-Shot Setting

`--train_ratio` 用于指定下游分类阶段所使用的标注训练数据比例：

\[
N_{\text{used}} = \max\left(1,\left\lfloor N_{\text{train}} \times \text{train\_ratio}\right\rfloor\right)
\]

例如：

- `--train_ratio 1.0`：使用全部下游训练数据；
- `--train_ratio 0.1`：仅使用 10% 的下游标注训练数据；
- `--train_ratio 0.05`：仅使用 5% 的下游标注训练数据。

本项目默认重点使用少样本设置。以下示例使用：

```bash
--train_ratio 0.1
```

即仅使用下游训练集中 **10% 的标注样本**训练分类器。

---

### 4.2 Run on Sleep-EDF Cassette

完成 Sleep-EDF 数据预处理后，运行：

```bash
python self_supservised.py \
    --dataset SLEEP \
    --model AIM \
    --train_ratio 0.1
```

该命令将：

- 在 Sleep-EDF 的 `pretext` 数据上进行自监督训练；
- 从 `train` 集中随机选择 10% 标注样本；
- 基于有限标注样本训练下游线性分类器；
- 在 `test` 集上输出分类性能和混淆矩阵。

---

### 4.3 Run on SHHS

完成 SHHS 数据预处理后，运行：

```bash
python self_supservised.py \
    --dataset SHHS \
    --model AIM \
    --train_ratio 0.1
```

该命令将使用 SHHS 数据集进行自监督预训练和少样本下游睡眠分期评估。

---

## 5. Important Arguments

`self_supservised.py` 中常用参数说明如下：

| 参数 | 含义 | 示例 |
|---|---|---|
| `--dataset` | 指定数据集名称 | `SLEEP` 或 `SHHS` |
| `--model` | 指定自监督模型名称 | `AIM` |
| `--train_ratio` | 下游分类器使用的标注训练数据比例 | `0.1` |
| `--T` | 对比学习温度参数（temperature） | `0.3` |
| `--sigma` | 模型或损失函数中的超参数 | `2.0` |
| `--delta` | 模型或损失函数中的超参数 | `0.2` |

例如，使用 10% 标注样本运行 Sleep-EDF：

```bash
python self_supservised.py --dataset SLEEP --model AIM --train_ratio 0.1
```

---

## 6. Output and Evaluation

训练过程中，程序会输出各数据划分中的样本数量，例如：

```text
pretext (all patient): ...
train (all patient): ... / ... (Ratio: 0.1)
test (all patient): ...
```

其中：

- `pretext (all patient)`：参与自监督预训练的数据量；
- `train (all patient)`：实际用于下游分类器训练的样本量 / 完整训练集样本量；
- `Ratio: 0.1`：表示仅使用 10% 标注训练数据；
- `test (all patient)`：用于最终评估的测试样本量。

下游任务使用 Logistic Regression 进行线性评估，并输出测试集预测结果、分类准确率及混淆矩阵等指标。

---

## 7. Notes

1. **路径配置**  
   在运行预处理或训练脚本前，请确认脚本中的数据根目录、预处理数据目录和模型保存目录与本地环境一致。

2. **少样本采样随机性**  
   下游训练数据会在训练集中随机采样。因此，不同运行之间可能会使用不同的 10% 标注样本，并导致结果存在一定波动。

3. **建议重复实验**  
   为获得更稳定、可靠的少样本实验结论，建议在不同随机种子下重复运行多次，并报告平均性能和标准差。

4. **数据访问要求**  
   SHHS 数据集具有访问权限要求，请遵守 NSRR 的数据使用协议。

---

## 8. Example: 10% Labeled Data Setting

以下是本项目推荐的少样本实验流程：

```bash
# 1. 预处理数据（以 Sleep-EDF 为例）
python sleepEDF_cassette_process.py --windowsize 30 --multiprocess 8

# 2. 自监督预训练 + 使用 10% 标注数据进行下游分类评估
python self_supservised.py \
    --dataset SLEEP \
    --model AIM \
    --train_ratio 0.1
```

该设置用于评估：在仅使用 **10% 标注训练数据**的条件下，自监督学习得到的睡眠信号表征能否支持有效的睡眠分期分类。
```
