import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetModified(nn.Module):
    def __init__(self, output_dim=512):
        super(ResNetModified, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        return x

class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, noise_scale=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Feature extractor (ResNet)
        self.feature_extractor = ResNetModified(feature_dim)
        self.feature_dim = feature_dim # Output dimension of ResNet features
        self.noise_scale = noise_scale
        # Encoder layers
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        if noise_scale is None:
            self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder layers
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        # Feature decoder (inverse of ResNet)
        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_dim, self.feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_dim // 2, self.feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_dim // 2, self.feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.feature_dim // 4, self.feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.feature_dim // 4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten

        mu = self.fc_mu(features)
        logvar = None
        if self.noise_scale is None:
            logvar = self.fc_logvar(features)

        return mu, logvar

    def decode(self, z):
        features = self.fc_decode(z)
        features = features.view(features.size(0), self.feature_dim, 1, 1)  # Reshape

        reconstruction = self.feature_decoder(features)
        reconstruction = F.interpolate(reconstruction, size=(224, 224), mode='bilinear', align_corners=False)
        return reconstruction

    def reparameterize(self, mu, logvar):
        if self.noise_scale is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mu)*std
        else:
            eps = torch.randn_like(mu)*self.noise_scale
        z = mu + eps
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar