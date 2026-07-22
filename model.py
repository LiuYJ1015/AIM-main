import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class BottleneckTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, max_seq_len=64):
      
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


class CNNEncoder2D_SHHS(nn.Module):
    def __init__(self, n_dim):
        super(CNNEncoder2D_SHHS, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        
        self.global_transformer = BottleneckTransformer(d_model=32, nhead=4, num_layers=2)
        
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(256, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sup = nn.Sequential(
            nn.Linear(256, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(256, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, 256, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=True) 
            signal.append(spectral)
        
        spectral_tensor = torch.stack(signal).permute(1, 0, 2, 3) # shape: (B, Channels, Freq, Time)
        
        power = torch.abs(spectral_tensor) ** 2
        log_power = torch.log(power + 1e-8)
        
        phase = torch.angle(spectral_tensor)
        
        return torch.cat([log_power, phase], dim=1)


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
    def __init__(self, n_dim):
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        
        self.global_transformer = BottleneckTransformer(d_model=32, nhead=4, num_layers=2)
        
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, 128, bias=False),  
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_dim, bias=True),
            nn.BatchNorm1d(self.n_dim) 
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.n_dim, self.n_dim // 2, bias=False),
            nn.BatchNorm1d(self.n_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim // 2, self.n_dim)
        )

        self.sup = nn.Sequential(
            nn.Linear(128, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 5, bias=True),
        )

        self.byol_mapping = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

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
