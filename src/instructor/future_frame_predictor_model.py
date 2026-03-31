import torch.nn as nn

class FrameDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=3, img_size=(224, 224)):
        super(FrameDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.img_size = img_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.output_channels, kernel_size=4, stride=2, padding=1),  # Final output
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim, 1, 1)  # Reshape to (batch_size, input_dim, 1, 1)
        x = self.decoder(x)
        # TODO: Necessary?
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)  # Resize to match input image size
        return x
