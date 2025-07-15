import torch
import torch.nn as nn
import torch.nn.functional as F

class DataEmbedding(nn.Module):
    """Projects input EEG channels into a higher-dimensional space."""
    def __init__(self, c_in, d_model):
        super().__init__()
        self.value_proj = nn.Linear(c_in, d_model)  # Channel-wise projection
        self.dropout = nn.Dropout(0.1)              # Regularization

    def forward(self, x, x_mark=None):
        # Input: [B, T, C] → Output: [B, T, d_model]
        return self.dropout(self.value_proj(x))


class InceptionBlock(nn.Module):
    """Multi-scale 1D convolution block (captures different temporal patterns)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Parallel convs with kernel sizes 2, 3, 5, 7
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=k, padding=k // 2)
            for k in [2, 3, 5, 7]
        ])

    def forward(self, x):
        # Concatenate multi-scale features along channel dimension
        outputs = []
        for conv in self.convs:
            out = conv(x)
            outputs.append(out)

        # Ensure temporal alignment
        min_len = min(out.shape[-1] for out in outputs)
        outputs = [o[:, :, :min_len] for o in outputs]
        return torch.cat(outputs, dim=1)  # [B, out_channels, T]


class TimesBlock(nn.Module):
    """Core TimesNet block for frequency-aware temporal modeling."""
    def __init__(self, d_model, seq_len, top_k=3):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.top_k = top_k  # Number of dominant frequencies to consider
        
        # Inception-based convolution stack
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_model * 2),  # Expand channels
            nn.GELU(),                             # Non-linearity
            InceptionBlock(d_model * 2, d_model)   # Compress back
        )

    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Frequency analysis via FFT
        x_freq = torch.fft.rfft(x, dim=1)  # [B, T//2+1, C]
        freqs = torch.mean(torch.abs(x_freq), dim=-1)  # Average over channels
        _, top_k_indices = torch.topk(freqs, self.top_k, dim=1)  # Dominant freqs

        # 2. Process each dominant frequency
        res = []
        for i in range(self.top_k):
            period = max(1, int(self.seq_len / (top_k_indices[0, i].item() + 1)))
            
            # Padding for period alignment
            if T % period != 0:
                pad_len = ((T // period) + 1) * period - T
                pad = x[:, :pad_len, :]  # Repeat initial values
                out = torch.cat([x, pad], dim=1)
            else:
                out = x

            # Reshape for period-based processing
            out = out.reshape(B, -1, period, C).permute(0, 3, 1, 2)  # [B, C, n_periods, P]
            out = out.reshape(B, C, -1)  # Flatten periods → [B, C, n_periods*P]
            
            # Multi-scale convolution
            out = self.conv(out)
            
            # Restore original shape
            out = out.reshape(B, C, -1, period).permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :T, :])  # Trim to original length

        # 3. Weighted combination of frequency components
        if not res:
            return x  # Fallback if no frequencies selected

        weights = F.softmax(freqs.gather(1, top_k_indices).float(), dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        res = torch.stack(res, dim=1)  # [B, K, T, C]
        return torch.sum(res * weights, dim=1) + x  # Residual connection


class TimesNet(nn.Module):
    """Full TimesNet architecture for EEG classification."""
    def __init__(self, config):
        super().__init__()
        # 1. Input embedding
        self.embedding = DataEmbedding(config.enc_in, config.d_model)
        
        # 2. Stacked TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(config.d_model, config.seq_len, config.top_k)
            for _ in range(config.e_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        
        # 3. Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.d_model * config.seq_len, 128),  # Temporal pooling
            nn.ReLU(),
            nn.Linear(128, config.num_class)                  # Binary output
        )

    def forward(self, x, x_mark=None, x_dec=None, x_mark_dec=None):
        # Input: [B, T, C]
        x = self.embedding(x)
        
        # Process through each TimesBlock
        for block in self.blocks:
            x = self.norm(block(x))
            
        return self.classifier(x)  # [B, num_class]