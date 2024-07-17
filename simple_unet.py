import math
import torch
from torch import nn

# if vanishing gradients occur, use residual connections a la ResNet https://arxiv.org/pdf/1512.03385
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Assign sine and cosine values to the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        return x + self.pe[:x.size(0), :]


class Simple_UNet(nn.Module):
    def __init__(self, embd_counts=[2**i for i in range(6,10)]) -> None:
        super().__init__()
        self.embd_counts = embd_counts
        timestamp_emb_dim = 32


        # represent beta_t as a timestamp
        self.timestamp_emb = nn.Sequential(
            SinusoidalPositionalEmbedding(timestamp_emb_dim),
            nn.Linear(timestamp_emb_dim, timestamp_emb_dim),
            nn.ReLU(),
        )


        # downsample
        self.downsample = nn.ModuleList()
        in_channels = 1
        for embd_count in self.embd_counts:
            self.downsample.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=embd_count, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_channels=embd_count, out_channels=embd_count, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = embd_count

        # bottleneck
        bottleneck_floor = self.embd_counts[-1]
        bottleneck_peak = bottleneck_floor*2
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=bottleneck_floor, out_channels=bottleneck_peak, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=bottleneck_peak, out_channels=bottleneck_floor, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # upsample
        self.upsample = nn.ModuleList()
        upsample_embed_counts = self.embd_counts[2::-1] # 256, 128, 64
        breakpoint()
        for embd_count in upsample_embed_counts:
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels=embd_count*2, out_channels=embd_count, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(in_channels=embd_count, out_channels=embd_count, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        self.output_layer = nn.Conv1d(in_channels=self.embd_counts[0], out_channels=1, kernel_size=1)


    def forward(self, x, timestamp):
        # add residual connections here

        # add timestamp awareness
        t = self.timestamp_emb(timestamp)

        for layer in self.downsample:
            x = layer(x)
        x = self.bottleneck(x)
        for layer in self.upsample:
            x = layer(x)
        return self.output_layer(x)

model = Simple_UNet()
print(model)