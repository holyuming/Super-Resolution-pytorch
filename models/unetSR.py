import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    ''' 2D convolution with padding = kernel_size // 2 --> make sure the input HW and output HW have same feature size '''
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)


class ConvBlock(nn.Module):
    ''' conv3x3 -> leakyrelu -> conv3x3 -> leakyrelu + conv1x1 '''
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True), # default negative slope = 0.01
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class Reconstruct(nn.Module):
    def __init__(self, in_channel, out_channel, scale=3):
        super(Reconstruct, self).__init__()
        self.scale = scale
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.Conv = nn.Conv2d(in_channel, out_channel * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.PS   = nn.PixelShuffle(scale)


    def forward(self, x):
        x = self.Conv(x)
        x = self.PS(x)
        return x


class UNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()

        self.dim = dim

        '''Downsample'''
        self.ConvBlock1 = block(3, dim)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim*4, dim*8)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16)

        '''Upsample'''
        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

        '''Reconstruction'''
        self.Reconstruct = Reconstruct(3, 3)

    def forward(self, x): # x --> [-1, 3, 256, 256]
        '''Downsample'''
        conv1 = self.ConvBlock1(x)          # [-1, 32, 256, 256]
        pool1 = self.pool1(conv1)           # [-1, 32, 128, 128]

        conv2 = self.ConvBlock2(pool1)      # [-1, 64, 128, 128]
        pool2 = self.pool2(conv2)           # [-1, 64, 64, 64] 

        conv3 = self.ConvBlock3(pool2)      # [-1, 128, 64, 64] 
        pool3 = self.pool3(conv3)           # [-1, 128, 32, 32]

        conv4 = self.ConvBlock4(pool3)      # [-1, 256, 32, 32]
        pool4 = self.pool4(conv4)           # [-1, 256, 16, 16]

        conv5 = self.ConvBlock5(pool4)      # [-1, 512, 16, 16]

        '''Upsample'''
        up6 = self.upv6(conv5)              # [-1, 256, 32, 32]
        up6 = torch.cat([up6, conv4], 1)    # [-1, 512, 32, 32]
        conv6 = self.ConvBlock6(up6)        # [-1, 256, 32, 32]

        up7 = self.upv7(conv6)              # [-1, 128, 64, 64]
        up7 = torch.cat([up7, conv3], 1)    # [-1, 256, 64, 64]
        conv7 = self.ConvBlock7(up7)        # [-1, 128, 64, 64]

        up8 = self.upv8(conv7)              # [-1, 64, 128, 128]
        up8 = torch.cat([up8, conv2], 1)    # [-1, 128, 128, 128]
        conv8 = self.ConvBlock8(up8)        # [-1, 64, 128, 128]

        up9 = self.upv9(conv8)              # [-1, 32, 256, 256]
        up9 = torch.cat([up9, conv1], 1)    # [-1, 64, 256, 256]
        conv9 = self.ConvBlock9(up9)        # [-1, 32, 256, 256]

        conv10 = self.conv10(conv9)         # [-1, 3, 256, 256]

        '''Original Long Skip Connection'''
        out = x + conv10                    # [-1, 3, 256, 256]

        '''Reconstruction to HQ resolution'''
        out = self.Reconstruct(out)         # [-1, 3, 768, 768]

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(dim=18).to(device)
    summary(model, (3, 256, 256))
    Img = torch.randn(1, 3, 256, 256).to(device)
    Out = model(Img)
    print('Input shape  : ', Img.shape)
    print('Output shape : ', Out.shape)
    macs, params = profile(model, inputs=(Img, ))
    print('macs: {} G, flops: {} G'.format(macs / 1e9, macs * 2 / 1e9))