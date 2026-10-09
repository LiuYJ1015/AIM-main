
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class BottleneckTransformer(nn.Module):
    def __init__(self, d_model=48, nhead=3, num_layers=6, max_seq_len=64):
        super(BottleneckTransformer, self).__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.3, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        x = x.view(B, C, L).permute(0, 2, 1)
        
        x = x + self.pos_embed[:, :L, :]
        
        out = self.transformer(x)
        
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2) 
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out



DEFAULT_PHASE_MODE = 'sincos_xchan'
PHASE_MASK_RATIO = 0.05

NORMALIZE_INPUT = True


def _normalize_signal(x, eps=1e-8):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def _stft_out_channels(in_ch, phase_mode):
    if phase_mode in ('logpower', 'poweronly'):
        return in_ch
    if phase_mode in ('original', 'angle', 'realimag', 'phase_only'):
        return 2 * in_ch
    if phase_mode in ('sincos', 'sincos_masked', 'sincos_lowamp'):
        return 3 * in_ch
    if phase_mode == 'sincos_xchan':
        return in_ch + 2
    if phase_mode == 'sincos_plus_xchan':
        return 3 * in_ch + 2
    if phase_mode == 'sincos_realimag':
        return 5 * in_ch
    raise ValueError(f'Unknown phase_mode: {phase_mode}')


def infer_phase_mode_from_state_dict(state_dict, in_ch=2):
    for key, value in state_dict.items():
        if key.endswith('conv1.0.weight'):
            return DEFAULT_PHASE_MODE if int(value.shape[1]) == 3 * in_ch else 'original'
    return DEFAULT_PHASE_MODE


def _spectrogram_features(spectral_tensor, phase_mode, dataset):
    power = torch.abs(spectral_tensor) ** 2
    log_power = torch.log(power + 1e-8)
    phase = torch.angle(spectral_tensor)

    if phase_mode in ('logpower',):
        return log_power
    if phase_mode == 'poweronly':
        mu = power.mean(dim=(1, 2, 3), keepdim=True)
        std = power.std(dim=(1, 2, 3), keepdim=True)
        return (power - mu) / (std + 1e-6)
    if phase_mode == 'angle':
        return torch.cat([log_power, phase], dim=1)
    if phase_mode == 'sincos':
        return torch.cat([log_power, torch.sin(phase), torch.cos(phase)], dim=1)
    if phase_mode == 'sincos_masked':
        magnitude = torch.sqrt(power + 1e-12)
        threshold = PHASE_MASK_RATIO * magnitude.mean(dim=(1, 2, 3), keepdim=True)
        gate = (magnitude >= threshold).to(log_power.dtype)
        return torch.cat([log_power,
                          torch.sin(phase) * gate,
                          torch.cos(phase) * gate], dim=1)
    if phase_mode == 'sincos_lowamp':
        magnitude = torch.sqrt(power)
        ref = magnitude.mean(dim=(2, 3), keepdim=True)
        floor_mag = PHASE_MASK_RATIO * ref
        gate = (magnitude >= floor_mag).to(log_power.dtype)
        return torch.cat([torch.log(power + floor_mag ** 2 + (1e-6 * ref) ** 2 + 1e-30),
                          torch.sin(phase) * gate,
                          torch.cos(phase) * gate], dim=1)
    if phase_mode in ('sincos_xchan', 'sincos_plus_xchan'):
        magnitude = torch.sqrt(power)
        ref = magnitude.mean(dim=(2, 3), keepdim=True)
        floor_mag = PHASE_MASK_RATIO * ref
        gate = (magnitude >= floor_mag).to(log_power.dtype)
        robust_log_power = torch.log(power + floor_mag ** 2 + (1e-6 * ref) ** 2 + 1e-30)
        cross = spectral_tensor[:, 1:2] * torch.conj(spectral_tensor[:, 0:1])
        d_phase = torch.angle(cross)
        cross_gate = gate[:, 1:2] * gate[:, 0:1]
        xchan = torch.cat([torch.sin(d_phase) * cross_gate,
                           torch.cos(d_phase) * cross_gate], dim=1)
        if phase_mode == 'sincos_xchan':
            return torch.cat([robust_log_power, xchan], dim=1)
        return torch.cat([robust_log_power,
                          torch.sin(phase) * gate,
                          torch.cos(phase) * gate,
                          xchan], dim=1)
    if phase_mode == 'realimag':
        return torch.cat([torch.log(torch.abs(spectral_tensor.real) + 1e-8),
                          torch.log(torch.abs(spectral_tensor.imag) + 1e-8)], dim=1)
    if phase_mode == 'sincos_realimag':
        return torch.cat([log_power, torch.sin(phase), torch.cos(phase),
                          torch.log(torch.abs(spectral_tensor.real) + 1e-8),
                          torch.log(torch.abs(spectral_tensor.imag) + 1e-8)], dim=1)
    if phase_mode == 'phase_only':
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
    if phase_mode == 'original':
        if dataset == 'SLEEP':
            return torch.cat([torch.log(torch.abs(spectral_tensor.real) + 1e-8),
                              torch.log(torch.abs(spectral_tensor.imag) + 1e-8)], dim=1)
        return torch.cat([log_power, phase], dim=1)
    raise ValueError(f'Unknown phase_mode: {phase_mode}')


class CNNEncoder2D_SHHS(nn.Module):
    def __init__(self, n_dim, in_ch=2, phase_mode=DEFAULT_PHASE_MODE,
                 normalize_input=NORMALIZE_INPUT):
        super(CNNEncoder2D_SHHS, self).__init__()

        self.in_ch = in_ch
        self.phase_mode = phase_mode
        self.normalize_input = normalize_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(_stft_out_channels(in_ch, phase_mode), 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 48, 2, True, True)
        
        self.global_transformer = BottleneckTransformer(d_model=48, nhead=3, num_layers=6)
        
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(384, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sup = nn.Sequential(
            nn.Linear(384, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(384, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, 384, bias=True),
        )

    def torch_stft(self, X_train):
        if self.normalize_input:
            X_train = _normalize_signal(X_train)
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=True)
            signal.append(spectral)
        
        spectral_tensor = torch.stack(signal).permute(1, 0, 2, 3)
        return _spectrogram_features(spectral_tensor, self.phase_mode, 'SHHS')


    def forward(self, x, simsiam=False, mid=True, byol=False, sup=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_transformer(x)
        
        x = x.reshape(x.shape[0], -1)

        if sup:
            return self.sup(x)
        elif simsiam:
            return x, self.fc(x)
        elif mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x


class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, n_dim, in_ch=2, phase_mode=DEFAULT_PHASE_MODE,
                 normalize_input=NORMALIZE_INPUT):
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.in_ch = in_ch
        self.phase_mode = phase_mode
        self.normalize_input = normalize_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(_stft_out_channels(in_ch, phase_mode), 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 48, 2, True, True)
        
        self.global_transformer = BottleneckTransformer(d_model=48, nhead=3, num_layers=6)
        
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(192, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Linear(192, self.n_dim, bias=True),
            nn.BatchNorm1d(self.n_dim)
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.n_dim, self.n_dim // 2, bias=False),
            nn.BatchNorm1d(self.n_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim // 2, self.n_dim)
        )

        self.sup = nn.Sequential(
            nn.Linear(192, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(192, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        if self.normalize_input:
            X_train = _normalize_signal(X_train)
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)

        spectral_tensor = torch.stack(signal).permute(1, 0, 2, 3, 4)
        spectral_tensor = torch.complex(spectral_tensor[..., 0], spectral_tensor[..., 1])
        return _spectrogram_features(spectral_tensor, self.phase_mode, 'SLEEP')

    def forward(self, x, simsiam=False, mid=True, byol=False, sup=False):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_transformer(x)

        x = x.reshape(x.shape[0], -1)

        if sup:
            return self.sup(x)
        elif simsiam:
            return x, self.fc(x)
        elif mid:
            return x
        elif byol:
            x = self.fc(x)
            x = self.byol_mapping(x)
            return x
        else:
            x = self.fc(x)
            return x
