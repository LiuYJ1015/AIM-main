
### AIM-STAGE

This repository contains the official implementation of **AIM-STAGE**, a self-supervised framework for **few-shot EEG sleep stage classification**. The framework learns robust sleep EEG representations from unlabeled data and evaluates them with limited labeled samples.

### 📊 Datasets

The implementation supports the following public sleep datasets:

### 1. SLEEP-2013

- **Description:** SLEEP-2013 contains approximately 39 overnight polysomnography (PSG) recordings from 20 healthy subjects.
- **EEG channels:** `Fpz-Cz` and `Pz-Oz`
- **Sampling rate:** 100 Hz
- **Role:** Used for self-supervised pre-training and few-shot downstream sleep-stage classification.
- **Download:** [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)

### 2. SLEEP-2018

- **Description:** SLEEP-2018 contains 153 overnight PSG recordings from 78 subjects.
- **EEG channels:** `Fpz-Cz` and `Pz-Oz`
- **Sampling rate:** 100 Hz
- **Role:** Used for self-supervised pre-training and few-shot downstream sleep-stage classification.
- **Download:** [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)

**3. SHHS (Sleep Heart Health Study)**
- **Role:** Used for large-scale self-supervised pre-training and few-shot downstream evaluation.
- **Download:** [NSRR SHHS](https://sleepdata.org/datasets/shhs)

> Access to SHHS requires registration and compliance with the NSRR data-use agreement.

---

### ⚙️ Data Preprocessing

The preprocessing scripts split raw sleep recordings into **non-overlapping 30-second epochs**, consistent with standard sleep-stage annotations. The processed data are divided into:

- `pretext/`: self-supervised pre-training data;
- `train/`: labeled data for downstream linear evaluation;
- `test/`: test data for final evaluation.

### Sleep-EDF Cassette


python sleepEDF_cassette_process.py --windowsize 30 --multiprocess 8
```

### SHHS

```bash
python shhs_process.py --windowsize 30 --multiprocess 8
```

Here, `--windowsize 30` specifies a 30-second epoch, and `--multiprocess 8` enables eight preprocessing workers. Please configure the raw-data paths in the preprocessing scripts before running them.

---

## 🚀 Usage & Training Commands

AIM-STAGE follows a two-stage paradigm: self-supervised representation learning on `pretext/` data, followed by linear evaluation using a limited fraction of labeled `train/` data.

### 1. Sleep-EDF Cassette

```bash
python self_supservised.py \
    --dataset SLEEP \
    --model AIM \
    --n_dim 128 \
    --train_ratio 0.1
```

### 2. SHHS

```bash
python self_supservised.py \
    --dataset SHHS \
    --model AIM \
    --n_dim 256 \
    --train_ratio 0.1
```

The argument `--train_ratio 0.1` means that only **10% of labeled downstream training samples** are used for linear classification. Set `--train_ratio 1.0` to use the full training set.

---

## 📖 Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- MNE

```bash
pip install torch numpy scipy scikit-learn mne
```
```
