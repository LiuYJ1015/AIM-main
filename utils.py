
import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
import os
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
import pywt

S4_OPERATORS = ['noise', 'shift', 'wavelet', 'phase']
S4_PROBS = np.array([0.30, 0.35, 0.15, 0.20], dtype=np.float64)
S4_PARAMS = dict(noise_degree=0.04,
                 shift_frac=0.12,
                 wavelet_scale=0.10,
                 wavelet_noise=0.02,
                 phase_shift=0.25)

R10_OPERATORS = S4_OPERATORS
R10_PROBS = S4_PROBS
R10_PARAMS = S4_PARAMS

def _s4_weights(selected):
    idx = [S4_OPERATORS.index(s) for s in selected]
    w = S4_PROBS[idx]
    return w / w.sum()

_r10_weights = _s4_weights

def _resolve_aug_options(aug_menu):
    if aug_menu in ('s4', 'r10', None, ''):
        return list(S4_OPERATORS)
    if aug_menu == 'legacy':
        return ['noise_legacy', 'filter', 'crop', 'swap']
    if aug_menu == 'none':
        return []
    for base in ('s4', 'r10'):
        if aug_menu.startswith(base + '_minus_'):
            drop = aug_menu[len(base) + len('_minus_'):]
            return [o for o in S4_OPERATORS if o != drop]
        if aug_menu.startswith(base + '_') and aug_menu.endswith('_only'):
            keep = aug_menu[len(base) + 1:-len('_only')]
            return [keep]
    return [x for x in aug_menu.split('+') if x]

def _restricted_shift(x, max_frac=0.12):
    max_l = max(1, int(round(x.shape[1] * max_frac)))
    shift = int(np.random.randint(1, max_l + 1))
    if np.random.rand() < 0.5:
        shift = -shift
    return np.roll(x, shift=shift, axis=1)

def wavelet_channel(ts, level=3, wave='db4', scale_range=(0.9, 1.1), noise_std=0.02):
    coeffs = pywt.wavedec(ts, wavelet=wave, level=level, mode='periodization')
    out = [coeffs[0]]
    for c in coeffs[1:]:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        noise = np.random.normal(0, noise_std, c.shape) * (np.std(c) + 1e-6)
        out.append(c * scale + noise)
    rec = pywt.waverec(out, wavelet=wave, mode='periodization')
    return rec[:len(ts)]

def phase_channel(ts, max_shift=0.25):
    spec = np.fft.rfft(ts)
    phase = np.angle(spec) + np.random.uniform(-max_shift, max_shift, size=spec.shape)
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=len(ts))

def phase_augment(x, max_shift=0.25):
    spec = np.fft.rfft(x, axis=1)
    phase = np.angle(spec) + np.random.uniform(-max_shift, max_shift, size=spec.shape)
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=x.shape[1], axis=1)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
LABEL_MAP_SLEEP = {"W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4}

def _map_sleep_label(value):
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 4:
            return 3
        if value > 4:
            return 4
        return value
    return LABEL_MAP_SLEEP[str(value)]

def _map_shhs_label(value):
    value = int(value)
    if value == 4:
        return 3
    if value > 4:
        return 4
    return value

def read_labels(data_dir, files, kind="sleep"):
    data_dir = data_dir if data_dir.endswith(os.sep) else data_dir + os.sep
    mapper = _map_shhs_label if kind == "shhs" else _map_sleep_label
    labels = []
    for name in files:
        with open(os.path.join(data_dir, name), "rb") as fh:
            sample = pickle.load(fh)
        labels.append(mapper(sample["y"]))
    return np.asarray(labels, dtype=np.int64)

def class_counts(y, indices=None):
    y = np.asarray(y)
    if indices is not None:
        y = y[np.asarray(indices, dtype=np.int64)]
    return {int(c): int(np.sum(y == c)) for c in range(5)}

def stratified_indices(y, keep, seed, min_per_class=1):
    y = np.asarray(y)
    n_total = len(y)
    keep = int(min(max(int(keep), 1), n_total))
    rng = np.random.RandomState(seed)

    if keep >= n_total:
        selected = np.arange(n_total, dtype=np.int64)
        rng.shuffle(selected)
        return selected

    classes = [int(c) for c in np.unique(y)]
    counts = {c: int(np.sum(y == c)) for c in classes}

    if keep <= len(classes):
        probs = np.asarray([counts[c] for c in classes], dtype=float)
        probs = probs / probs.sum()
        chosen = rng.choice(np.asarray(classes), size=keep, replace=False, p=probs)
        selected = []
        for c in chosen:
            pool = np.where(y == c)[0]
            selected.append(int(rng.choice(pool, size=1, replace=False)[0]))
        selected = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected)
        return selected

    quota = {c: keep * counts[c] / float(n_total) for c in classes}
    alloc = {c: int(np.floor(quota[c])) for c in classes}

    remainders = {c: quota[c] - alloc[c] for c in classes}
    order = sorted(classes, key=lambda c: (-remainders[c], -counts[c]))
    for c in order:
        if sum(alloc.values()) >= keep:
            break
        if alloc[c] < counts[c]:
            alloc[c] += 1
    while sum(alloc.values()) < keep:
        candidates = [c for c in classes if alloc[c] < counts[c]]
        if not candidates:
            break
        c = max(candidates, key=lambda x: counts[x])
        alloc[c] += 1

    for c in classes:
        target = min(int(min_per_class), counts[c])
        while alloc[c] < target:
            donors = [d for d in classes if alloc[d] > max(int(min_per_class), 1)]
            if not donors:
                donors = [d for d in classes if alloc[d] > 1]
            if not donors:
                break
            d = max(donors, key=lambda x: alloc[x])
            alloc[d] -= 1
            alloc[c] += 1

    selected = []
    for c in classes:
        pool = np.where(y == c)[0]
        take = min(alloc[c], len(pool))
        if take > 0:
            selected.extend(rng.choice(pool, size=take, replace=False).tolist())

    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) < keep:
        remaining = np.setdiff1d(np.arange(n_total, dtype=np.int64), selected, assume_unique=False)
        extra = rng.choice(remaining, size=keep - len(selected), replace=False)
        selected = np.concatenate([selected, extra.astype(np.int64)])
    rng.shuffle(selected)
    return selected

def select_indices(y, keep, seed, strategy="stratified", min_per_class=1, n1_oversample=1.0):
    y = np.asarray(y)
    keep = int(min(max(int(keep), 1), len(y)))
    if strategy == "random":
        selected = np.random.RandomState(seed).permutation(len(y))[:keep]
    elif strategy == "stratified":
        selected = stratified_indices(y, keep, seed, min_per_class=min_per_class)
    else:
        raise ValueError("strategy must be 'random' or 'stratified'")

    if n1_oversample and float(n1_oversample) > 1.0:
        n1_pool = selected[y[selected] == 1]
        if len(n1_pool) > 0:
            extra_n = int(round(len(n1_pool) * (float(n1_oversample) - 1.0)))
            if extra_n > 0:
                extra = np.random.RandomState(seed + 1000003).choice(
                    n1_pool, size=extra_n, replace=True
                )
                selected = np.concatenate([selected, extra.astype(np.int64)])
    return np.asarray(selected, dtype=np.int64)

def stratified_folds(y, n_splits=5, seed=0):
    y = np.asarray(y)
    n_total = len(y)
    n_splits = int(min(max(int(n_splits), 1), max(n_total, 1)))
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = np.where(y == c)[0].astype(np.int64)
        rng.shuffle(idx)
        for j, item in enumerate(idx):
            folds[j % n_splits].append(int(item))
    return [np.asarray(sorted(f), dtype=np.int64) for f in folds if len(f) > 0]

def denoise_channel(ts, bandpass, signal_freq, bound):
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4
    
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts

class SHHSLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=True, aug_menu='s4'):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = [0, 1, 2, 3, 4]
        self.bandpass1 = (1, 3)
        self.bandpass2 = (30, 60)
        self.n_length = 125 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 125
        self.bound = 0.000125
        self.aug_menu = aug_menu
        self.aug_options = _resolve_aug_options(aug_menu)
        self.aug_weights = _s4_weights(self.aug_options) if aug_menu.startswith(('s4', 'r10')) else None
        self._cur_scale = None

    def _chan_bound(self, i):
        scale = getattr(self, '_cur_scale', None)
        if scale is None:
            return self.bound
        return float(self.bound / (float(scale[i]) + 1e-12))

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio, degree=0.05, allow_none=True):
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                modes = ['high', 'low', 'both', 'no'] if allow_none else ['high', 'low', 'both']
                mode = np.random.choice(modes)
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=degree, bound=self._chan_bound(i))
        return x
    
    def remove_noise(self, x, ratio):
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self._chan_bound(i)) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self._chan_bound(i))
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self._chan_bound(i))
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self._chan_bound(i))
            else:
                pass

        return x
    
    def crop(self, x):
        l = np.random.randint(1, 3749)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        if not self.aug_options:
            return x
        if self.aug_weights is not None:
            choice = np.random.choice(self.aug_options, p=self.aug_weights)
        else:
            choice = np.random.choice(self.aug_options)
        if choice == 'noise':
            x = self.add_noise(x, ratio=0.5, degree=S4_PARAMS['noise_degree'])
        elif choice == 'noise_legacy':
            x = self.add_noise(x, ratio=0.5, degree=0.05)
        elif choice == 'filter':
            x = self.remove_noise(x, ratio=0.5)
        elif choice == 'shift':
            x = _restricted_shift(x, max_frac=S4_PARAMS['shift_frac'])
        elif choice == 'crop':
            x = self.crop(x)
        elif choice == 'swap':
            x = x[[1, 0], :]
        elif choice == 'wavelet':
            s = S4_PARAMS['wavelet_scale']
            for i in range(self.n_channels):
                x[i, :] = wavelet_channel(x[i, :], scale_range=(1.0 - s, 1.0 + s),
                                          noise_std=S4_PARAMS['wavelet_noise'])
        elif choice == 'phase':
            x = phase_augment(x, max_shift=S4_PARAMS['phase_shift'])
        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        if y == 4:
            y = 3
        elif y > 4:
            y = 4
        y = torch.LongTensor([y])

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            return torch.FloatTensor(X), y

class SLEEPCALoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=True, aug_menu='s4'):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025
        self.aug_menu = aug_menu
        self.aug_options = _resolve_aug_options(aug_menu)
        self.aug_weights = _s4_weights(self.aug_options) if aug_menu.startswith(('s4', 'r10')) else None
        self._cur_scale = None

    def _chan_bound(self, i):
        scale = getattr(self, '_cur_scale', None)
        if scale is None:
            return self.bound
        return float(self.bound / (float(scale[i]) + 1e-12))

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio, degree=0.05, allow_none=True):
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                modes = ['high', 'low', 'both', 'no'] if allow_none else ['high', 'low', 'both']
                mode = np.random.choice(modes)
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=degree, bound=self._chan_bound(i))
        return x
    
    def remove_noise(self, x, ratio):
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self._chan_bound(i)) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self._chan_bound(i))
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self._chan_bound(i))
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self._chan_bound(i))
            else:
                pass
        return x
    
    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        if not self.aug_options:
            return x
        if self.aug_weights is not None:
            choice = np.random.choice(self.aug_options, p=self.aug_weights)
        else:
            choice = np.random.choice(self.aug_options)
        if choice == 'noise':
            x = self.add_noise(x, ratio=0.5, degree=S4_PARAMS['noise_degree'])
        elif choice == 'noise_legacy':
            x = self.add_noise(x, ratio=0.5, degree=0.05)
        elif choice == 'filter':
            x = self.remove_noise(x, ratio=0.5)
        elif choice == 'shift':
            x = _restricted_shift(x, max_frac=S4_PARAMS['shift_frac'])
        elif choice == 'crop':
            x = self.crop(x)
        elif choice == 'swap':
            x = x[[1, 0], :]
        elif choice == 'wavelet':
            s = S4_PARAMS['wavelet_scale']
            for i in range(self.n_channels):
                x[i, :] = wavelet_channel(x[i, :], scale_range=(1.0 - s, 1.0 + s),
                                          noise_std=S4_PARAMS['wavelet_noise'])
        elif choice == 'phase':
            x = phase_augment(x, max_shift=S4_PARAMS['phase_shift'])
        return x

        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        if y == 'W':
            y = 0
        elif y == 'R':
            y = 4
        elif y in ['1', '2', '3']:
            y = int(y)
        elif y == '4':
            y = 3
        else:
            y = 0
        
        y = torch.LongTensor([y])

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            return torch.FloatTensor(X), y