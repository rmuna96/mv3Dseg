import torch
import torch.nn as nn
from utils import TensorList


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
        Classic UNet implementation.
    """

    def __init__(self,
                 in_channels=1, out_channels=4, features=[16, 32, 64, 128, 256],
                 ):

        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.out_channels = out_channels

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1]  # reverse the list because we are going upwards

        x_d = self.bottleneck(x)  # define a temp variable a bottleneck level to pass to the decoders
        for idx in range(0, len(self.ups),
                         2):  # step of 2 because in ups there is the transpose conv and the double conv,
            # but the concatenation is done at the first step
            x_d = self.ups[idx](x_d)  # upsampling
            skip_connection = skip_connections[idx // 2]

            try:
                x_d.shape = skip_connection.shape
            except:
                x_d = nn.functional.interpolate(x_d, skip_connection.shape[
                                                2:])  # resize with respect to length, height, width

            concat_skip = torch.cat((skip_connection, x_d),
                                    dim=1)  # concatention on channel size [batch, channel, length, height, width]
            x_d = self.ups[idx + 1](concat_skip)  # doubleConV

        return self.final_conv(x_d)


class ResUnit(nn.Module):
    """
        Implementation of residual unit.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResUnit, self).__init__()

        # convolutional layer
        self.b1 = nn.BatchNorm3d(in_channels)
        self.rl1 = nn.PReLU()
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.b2 = nn.BatchNorm3d(out_channels)
        self.rl2 = nn.PReLU()
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # shortcut connection (identity mapping)
        self.s = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.b3 = nn.BatchNorm3d(out_channels)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.rl1(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.rl2(x)
        x = self.c2(x)
        s = self.s(inputs)
        s = self.b3(s)

        skip = x + s
        return skip


class ResUNet(nn.Module):
    """
        UNet implementation with residual unit.
    """

    def __init__(self,
                 in_channels=1, out_channels=4, features=[16, 32, 64, 128, 256]
                 ):

        super(ResUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.out_channels = out_channels

        # Encoder 1
        self.c11 = nn.Conv3d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.b11 = nn.BatchNorm3d(features[0])
        self.rl11 = nn.PReLU()
        self.c12 = nn.Conv3d(features[0], features[0], kernel_size=3, stride=1, padding=1)

        self.c13 = nn.Conv3d(in_channels, features[0], kernel_size=1, stride=1, padding=0)
        self.b12 = nn.BatchNorm3d(features[0])

        # Down part of UNET (encoder 2 and 3)
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(ResUnit(in_channels, feature, stride=2))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(

                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResUnit(feature*2, feature, stride=1))

        # bottleneck
        self.bottleneck = ResUnit(features[-1], features[-1]*2, stride=2)
        # finalconv
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1, padding=0)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        skip_connections = []

        # encoder 1
        x = self.c11(inputs)
        x = self.b11(x)
        x = self.rl11(x)
        x = self.c12(x)

        s = self.b12(self.c13(inputs))

        x = x + s
        skip_connections.append(x)

        # encoder 2 and 3
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]  # reverse the list because we are going upwards

        x_d = self.bottleneck(x)  # define a temp variable a bottleneck level to pass to the decoders
        for idx in range(0, len(self.ups),
                         2):  # step of 2 because in ups there is the transpose conv and the double conv,
            # but the concatenation is done at the first step
            x_d = self.ups[idx](x_d)  # upsampling
            skip_connection = skip_connections[idx // 2]

            try:
                x_d.shape = skip_connection.shape
            except:
                x_d = nn.functional.interpolate(x_d, skip_connection.shape[
                                                     2:])  # resize with respect to length, height, width

            concat_skip = torch.cat((skip_connection, x_d),
                                    dim=1)  # concatention on channel size [batch, channel, length, height, width]
            x_d = self.ups[idx + 1](concat_skip)  # doubleConV

        return self.final_conv(x_d)


class MdResUNet(nn.Module):
    """
            UNet implementation with residual unit and multi decoders up to the number of channels
    """
    def __init__(self,
                 in_channels=1, out_channels=4, features=[16, 32, 64, 128, 256], num_decoders=3,
                 ):

        super(MdResUNet, self).__init__()
        self.num_decoders = num_decoders
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in range(self.num_decoders):
            self.decoders.append(nn.ModuleList())
        self.downs = nn.ModuleList()
        self.out_channels = out_channels

        # Encoder 1
        self.c11 = nn.Conv3d(in_channels, features[0], kernel_size=3, stride=1, padding=1)
        self.b11 = nn.BatchNorm3d(features[0])
        self.rl11 = nn.PReLU()
        self.c12 = nn.Conv3d(features[0], features[0], kernel_size=3, stride=1, padding=1)

        self.c13 = nn.Conv3d(in_channels, features[0], kernel_size=1, stride=1, padding=0)
        self.b12 = nn.BatchNorm3d(features[0])

        # Down part of UNET (encoder 2 and 3)
        in_channels = features[0]
        for feature in features[1:]:
            self.downs.append(ResUnit(in_channels, feature, stride=2))
            in_channels = feature

        # Up part of UNET (decoder)
        for decoder in self.decoders:
            for feature in reversed(features):
                decoder.append(

                    nn.ConvTranspose3d(
                        feature*2, feature, kernel_size=2, stride=2
                    )
                )
                decoder.append(ResUnit(feature*2, feature, stride=1))
        # bottleneck
        self.bottleneck = ResUnit(features[-1], features[-1]*2, stride=2)
        # finalconv
        self.final_conv = nn.Conv3d(features[0], 2, kernel_size=1, padding=0)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        skip_connections = []

        # encoder 1
        x = self.c11(inputs)
        x = self.b11(x)
        x = self.rl11(x)
        x = self.c12(x)

        s = self.b12(self.c13(inputs))

        x = x + s
        skip_connections.append(x)

        # encoder 2 and 3
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]  # reverse the list because we are going upwards

        # decoder
        outputs = TensorList()    #!todo this must be tensor list now
        for decoder in self.decoders:
            x_d = self.bottleneck(x)  # define a temp variable a bottleneck level to pass to the decoders
            for idx in range(0, len(decoder),
                             2):  # step of 2 because in ups there is the transpose conv and the double conv,
                # but the concatenation is done at the first step
                x_d = decoder[idx](x_d)  # upsampling
                skip_connection = skip_connections[idx // 2]

                try:
                    x_d.shape = skip_connection.shape
                except:
                    x_d = nn.functional.interpolate(x_d, skip_connection.shape[
                                                         2:])  # resize with respect to length, height, width

                concat_skip = torch.cat((skip_connection, x_d),
                                        dim=1)  # concatention on channel size [batch, channel, length, height, width]
                x_d = decoder[idx + 1](concat_skip)  # doubleConV
            outputs.append(self.final_conv(x_d))

        return outputs


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 64, 64)   #batchsize, channels, depth, height, width
    model = MdResUNet(in_channels=1, out_channels=4, num_decoders=3, features=[16, 32, 64, 128, 256])
    pred = model(x)
    print(pred.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)