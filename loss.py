import torch
import torch.nn as nn
import torch.nn.functional as F


class OurLoss(nn.Module):
    def __init__(self, device, margin=0.2, sigma=3.0, T=2.0, kernel_bandwidth=0.5, num_kernels=3):
        super(OurLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.sigma = sigma
        self.T = T
        self.kernel_bw = kernel_bandwidth
        self.num_kernels = num_kernels

    def forward(self, emb_anchor, emb_positive):
        emb_anchor = F.normalize(emb_anchor, p=2, dim=1)
        emb_positive = F.normalize(emb_positive, p=2, dim=1)

        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_mean = torch.mean(sim, dim=1, keepdim=True)

        total_weights = torch.zeros_like(sim)
        if self.num_kernels > 1:
            step = (sim_max - sim_mean) / (self.num_kernels - 1 + 1e-6)
            for i in range(self.num_kernels):
                current_mu = sim_mean + i * step
                w_i = torch.exp(-torch.pow(sim - current_mu, 2) / (2 * self.kernel_bw ** 2))
                total_weights += w_i
        else:
            total_weights = torch.exp(-torch.pow(sim - sim_max, 2) / (2 * self.kernel_bw ** 2))

        total_weights = total_weights / (torch.sum(total_weights, dim=1, keepdim=True) + 1e-8)
        neg = torch.mm(total_weights, emb_positive)

        l_pos = torch.exp(-torch.sum(torch.pow(emb_anchor - emb_positive, 2), dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum(torch.pow(emb_anchor - neg, 2), dim=1) / (2 * self.sigma ** 2))
        return torch.max(torch.zeros_like(l_pos), l_neg - l_pos + self.margin).mean()
