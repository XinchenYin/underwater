import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention Module
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            conv1x1(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# Position Attention Module
class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

# Combined Channel and Position Attention (CPAM)
class CPAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.pa = PositionAttention(in_channels)

    def forward(self, x):
        x_ca = x * self.ca(x)
        x_pa = x_ca * self.pa(x_ca)
        return x_pa

# Residual Block for TFDMod
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# Time-Frequency-Domain Feature Extraction Module (TFDMod)
class TFDMod(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 3, 256, 256)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Stacked residual blocks with CPAM
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.layer5 = self._make_layer(512, 512, blocks=2, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(CPAM(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# Gated Linear Unit Block for TDMod
class GLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.conv_res = conv1x1(in_channels, out_channels)
        self.cpam = CPAM(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv_res(x)
        out = self.bn(self.conv(x))
        a, b = out.chunk(2, dim=1)
        out = torch.sigmoid(b) * a
        out = out + res
        out = self.cpam(out)
        return self.relu(out)

# Time-Domain Feature Extraction Module (TDMod)
class TDMod(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 4, 6250)    time-domain input (4 sensors × 6250 samples)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Stacked GLU blocks
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, 512)
        self.layer4 = self._make_layer(512, 512)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            GLUBlock(in_channels, out_channels),
            GLUBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # x: (batch, 4, 6250)
        # reshape to (batch,1,4,6250) for conv2d
        x = x.unsqueeze(1)
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Overall AMSS-FFN Model
class AMSS_FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tfdmod = TFDMod()
        self.tdmod = TDMod()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x_tf, x_t):
        # x_tf: (batch, 3, 256, 256) time-frequency input
        # x_t:  (batch, 1, 4, 6250)    time-domain input
        f1 = self.tfdmod(x_tf)
        f2 = self.tdmod(x_t)
        v1 = self.global_pool(f1).view(f1.size(0), -1)
        v2 = self.global_pool(f2).view(f2.size(0), -1)
        v = torch.cat([v1, v2], dim=1)
        out = self.fc(v)
        return out


if __name__ == "__main__":
    # quick sanity check
    # Example usage:
    model = AMSS_FFN()
    tf_input = torch.randn(8, 3, 256, 256)
    td_input = torch.randn(8, 4, 6250)
    coords = model(tf_input, td_input)
    print("Output shape:", coords.shape)  # should be (8, 2)