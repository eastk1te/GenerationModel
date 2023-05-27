import torch
import torch.nn as nn

# 생성자 (Generator) 모델 정의
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_filters=64):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(num_filters * 4),
            ResidualBlock(num_filters * 4),
            ResidualBlock(num_filters * 4),
            # 추가적인 ResidualBlock을 필요한 만큼 반복할 수 있음
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        residual = self.residual_blocks(encoded)
        decoded = self.decoder(residual + encoded)
        return decoded

# 판별자 (Discriminator) 모델 정의
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        output = self.model(x)
        return output

# Residual 블록 정의
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        residual = self.model(x)
        return x + residual
