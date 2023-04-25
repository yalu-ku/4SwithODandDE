import torch
from torch.nn import *
from torchvision.transforms import CenterCrop


class Model(Module):
    def __init__(self):
        super().__init__()

        # Contracting
        self.conv_1 = Sequential(
            self.cbr(3, 64),
            self.cbr(64, 64, 3, 1))
        self.pool_1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = Sequential(
            self.cbr(64, 128),
            self.cbr(128, 128, 3, 1))
        self.pool_2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = Sequential(
            self.cbr(128, 256),
            self.cbr(256, 256, 3, 1))
        self.pool_3 = MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = Sequential(
            self.cbr(256, 512),
            self.cbr(512, 512, 3, 1),
            Dropout(p=0.5))
        self.pool_4 = MaxPool2d(kernel_size=2, stride=2)

        # Bottle Neck
        self.bottle_neck = Sequential(
            self.cbr(512, 1024),
            self.cbr(1024, 1024)
        )

        # Expanding
        self.up_conv_5 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv_5 = Sequential(
            self.cbr(1024, 512),
            self.cbr(512, 512)
        )

        self.up_conv_6 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_6 = Sequential(
            self.cbr(512, 256),
            self.cbr(256, 256)
        )

        self.up_conv_7 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_7 = Sequential(
            self.cbr(256, 128),
            self.cbr(128, 128)
        )

        self.up_conv_8 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_8 = Sequential(
            self.cbr(128, 64),
            self.cbr(64, 64)
        )

        self.fc = Conv2d(64, 21, kernel_size=1, stride=1)

    def forward(self, x):
        layer_1 = self.pool_1(self.conv_1(x))
        layer_2 = self.pool_2(self.conv_2(layer_1))
        layer_3 = self.pool_3(self.conv_3(layer_2))
        layer_4 = self.pool_4(self.conv_4(layer_3))

        bottle_neck = self.bottle_neck(layer_4)

        up_conv_5 = self.up_conv_5(bottle_neck)
        cat_1 = torch.cat((CenterCrop((up_conv_5.shape[2], up_conv_5.shape[3]))(layer_4), up_conv_5), dim=1)
        layer_5 = self.conv_5(cat_1)

        up_conv_6 = self.up_conv_5(layer_5)
        cat_2 = torch.cat((CenterCrop((up_conv_6.shape[2], up_conv_6.shape[3]))(layer_3), up_conv_6), dim=1)
        layer_6 = self.conv_6(cat_2)

        up_conv_7 = self.up_conv_5(layer_6)
        cat_3 = torch.cat((CenterCrop((up_conv_7.shape[2], up_conv_7.shape[3]))(layer_2), up_conv_7), dim=1)
        layer_7 = self.conv_7(cat_3)

        up_conv_8 = self.up_conv_5(layer_7)
        cat_4 = torch.cat((CenterCrop((up_conv_8.shape[2], up_conv_8.shape[3]))(layer_1), up_conv_8), dim=1)
        layer_8 = self.conv_8(cat_4)

        return self.fc(layer_8)

    @staticmethod
    def cbr(i_channel, o_channel, kernel_size=3, stride=1):
        return Sequential(
            Conv2d(i_channel, o_channel, kernel_size=kernel_size, stride=stride),
            BatchNorm2d(num_features=o_channel),
            ReLU()
        )
