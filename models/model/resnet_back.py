from torch import nn
from models.layers.multi_head_attention import MultiHeadAttention
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + out
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = shortcut + out
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()
        start_channel = 64
        self.in_channels = start_channel

        self.conv1 = nn.Conv2d(1, start_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(start_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dlayer1 = self._make_downlayer(downblock, start_channel, 2)
        self.dlayer2 = self._make_downlayer(downblock, start_channel*2, 2,
                                            stride=2)
        self.dlayer3 = self._make_downlayer(downblock, start_channel*4, 2,
                                            stride=2)
        self.dlayer4 = self._make_downlayer(downblock, start_channel*8, 2,
                                            stride=2)
        # self.dlayer5 = self._make_downlayer(downblock, start_channel*16, 6,
        #                                     stride=2)
        # self.dlayer6 = self._make_downlayer(downblock, start_channel*32, 3,
        #                                     stride=2)
        # self.encoder_linear = nn.Linear(1024, 1024)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mattn = MultiHeadAttention(d_model=1024, n_head=8)

        self.uplayer1 = self._make_up_block(upblock,start_channel*32, 1, stride=2)
        self.uplayer2 = self._make_up_block(upblock, start_channel*16, 3, stride=2)
        self.uplayer3 = self._make_up_block(upblock, start_channel*8, 6, stride=2)
        self.uplayer4 = self._make_up_block(upblock, start_channel*4, 3, stride=2)
        self.uplayer5 = self._make_up_block(upblock, start_channel*2, 2, stride=2)
        self.uplayer6 = self._make_up_block(upblock, start_channel, 2, stride=2)
        self.uplayer7 = self._make_up_block(upblock, start_channel//2, 2, stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               start_channel//2,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(start_channel//2),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, start_channel//2, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(start_channel//2, n_classes, kernel_size=1, stride=1,
                                          bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encode(self, x):
        x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)
        bs, dim, w, h = x.size()
        query = self.avg_pool(x)
        q = query.permute(0,2,3,1).reshape(bs, 1, dim)
        k = v = x.permute(0,2,3,1).reshape(bs, w * h, dim)
        fused_embedding = self.mattn(q, k, v)
        # x = self.dlayer5(x)
        # x = self.dlayer6(x)
        # x = self.encoder_linear(x.squeeze())
        return fused_embedding.squeeze()

    def decode(self, x, img_size):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer5(x)
        x = self.uplayer6(x)
        x = self.uplayer7(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=img_size)
        return x

    def generate(self, encoder_repr, img_size=(256,256)):
        x = encoder_repr.reshape(encoder_repr.shape[0], 64, 8, 8)
        logits = self.decode(x, img_size)
        return logits.sigmoid()

    def forward(self, x, return_embedding = False):
        x_size = x.size()
        repr = self.encode(x)
        if return_embedding:
            return repr.reshape(repr.shape[0], -1)
        reconstruction = self.decode(repr, x_size)

        return reconstruction


def ResNet50(**kwargs):
    return ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 1, **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 2], 22, **kwargs)
