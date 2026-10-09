# AIM-STAGE

Official implementation of **AIM-STAGE**, a self-supervised framework for **few-shot EEG sleep-stage classification**. AIM-STAGE learns transferable sleep EEG representations from unlabeled recordings and classifies sleep stages using only a small number of labeled epochs.

The framework combines three components:

- **Spectro-temporal explorer.** Each 30 s epoch is mapped by a short-time Fourier transform (STFT, `n_fft=256`, hop length `64`) into the time-frequency domain. The log-power spectrogram and the cross-channel phase difference, encoded as sine and cosine with low-amplitude bins masked, are concatenated into a multi-channel input.
- **Local-global hybrid encoder.** A convolutional branch (channels `4 -> 6 -> 8 -> 16 -> 48`) extracts local spectro-temporal patterns, and a bottleneck Transformer (`d_model=48`, 6 layers, 3 heads) models long-range temporal dependencies.
- **Adaptive interval-aware multi-kernel calibration loss.** For each anchor, multiple Gaussian kernels are placed between the mean and the maximum of the similarity distribution, and their responses are aggregated into a dynamic negative representation that is used in a margin-based contrastive objective.

## Files

| File | Description |
| --- | --- |
| `model.py` | Spectro-temporal explorer and local-global hybrid encoder. |
| `loss.py` | Contrastive losses, including the proposed calibration loss and the MoCo, BYOL, SimSiam and SimCLR baselines. |
| `utils.py` | Dataset loaders and the four physiology-aware augmentation operators. |
| `self_supservised.py` | Self-supervised pre-training and downstream linear evaluation. |

## Datasets

| Dataset | Subjects | Recordings | EEG channels | Sampling rate | Pretrain / train / test |
| --- | --- | --- | --- | --- | --- |
| SLEEP-2013 | 20 | 39 | Fpz-Cz, Pz-Oz | 100 Hz | 90 / 5 / 5 % |
| SLEEP-2018 | 58 | 114 | Fpz-Cz, Pz-Oz | 100 Hz | 90 / 5 / 5 % |
| SHHS-D | 1229 | 1229 | two EEG | 125 Hz | 98 / 1 / 1 % |
| SHHS-H | 2823 | 2823 | two EEG | 125 Hz | 98 / 1 / 1 % |

- SLEEP-2013 and SLEEP-2018 are from [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/).
- SHHS-D (apnea-hypopnea index above 15, obstructive sleep apnea) and SHHS-H (apnea-hypopnea index below 5, healthy) are from [NSRR SHHS](https://sleepdata.org/datasets/shhs). Access requires registration and compliance with the NSRR data-use agreement.

## Preprocessing

Each recording is processed as follows:

1. band-pass filtering between 0.5 and 45 Hz with a zero-phase finite-impulse-response filter;
2. 60 Hz notch filtering for SHHS;
3. signals kept at their native sampling rate, 100 Hz for SLEEP-2013/SLEEP-2018 and 125 Hz for SHHS;
4. segmentation into non-overlapping 30 s epochs, with epochs that lack a valid label discarded;
5. per-epoch, per-channel z-score normalization before the STFT.

The epochs are stored as pickles and split at the subject level into pre-training, training and test subsets.

## Usage

AIM-STAGE follows a two-stage paradigm: self-supervised pre-training on unlabeled recordings, followed by frozen-encoder linear evaluation with a limited fraction of labeled data.

```bash
# SLEEP-2013 / SLEEP-2018
python self_supservised.py --dataset SLEEP --n_dim 128 --train_ratio 0.05

# SHHS-D / SHHS-H
python self_supservised.py --dataset SHHS --n_dim 256 --train_ratio 0.05
```

`--train_ratio` sets the fraction of labeled downstream epochs (for example, `0.05` for 5%). The paper reports results at 1%, 2%, 5%, 10% and 15% labeled data.


## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- PyWavelets
- tqdm

```bash
pip install torch numpy scipy scikit-learn PyWavelets tqdm
```
