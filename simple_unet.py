import torch
from torch import nn

# if vanishing gradients occur, use residual connections a la ResNet https://arxiv.org/pdf/1512.03385

class Simple_UNet(nn.Module):
    def __init__(self, embd_counts=[2**i for i in range(6,10)]) -> None:
        super().__init__()
        self.embd_counts = embd_counts

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


    def forward(self, x):
        # residual connections here
        for layer in self.downsample:
            x = layer(x)
        x = self.bottleneck(x)
        for layer in self.upsample:
            x = layer(x)
        return self.output_layer(x)

model = Simple_UNet()
print(model)