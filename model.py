from __future__ import print_function
import torch.nn as nn
import torch
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),  # 64*32*32

            nn.Conv2d(ndf, ndf * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 128*16*16

            nn.Conv2d(ndf*2, ndf * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 256*8*8

            nn.Conv2d(ndf*4, ndf * 8, 5, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 512*4*4
        )
        self.fc1=nn.Sequential(
            nn.Linear(512*3*3,100),               # 50
            nn.Tanh()                  # 压缩图像均值
        )

        self.fc2= nn.Sequential(
            nn.Linear(512 * 3 * 3, 100),
            nn.Tanh()                  # 图像方差对数
        )

        self.e= torch.randn(1, 100).cuda().view([-1,100])
    def forward(self,input):
        feature = self.main(input)
        return self.fc1(feature.view(-1,512*3*3))+self.fc2(feature.view(-1,512*3*3))+self.e

# summary( Encoder(0,64,3),input_size=(3,64,64))

class Decoder(nn.Module):
    def __init__(self, ngpu,num,ngf):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.fc= nn.Sequential(
            nn.Linear(num,8192),
            nn.BatchNorm1d(num_features=8192),  # 8192=512*4*4
            nn.ReLU(inplace=True)
        )
        self.main = nn.Sequential(

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),           # [-1, 3, 79, 79]   生成图片尺寸
            nn.Tanh()
            # 输出是假图片的维度
        )

    def forward(self,input):
        feature=self.fc(input.view(-1,200*1*1))
        feature=feature.view(-1,512,4,4)
        feature=self.main(feature)
        return feature



class  Discriminator(nn.Module):
    def __init__(self,ngpu,ndf,nc):
        super( Discriminator,self).__init__()
        self.ngpu=ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc=nn.Sequential(
            nn.Linear(512*4*4,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        feature3=self.main(input)
        return self.fc(feature3.view(-1,512*4*4))

# # summary( Discriminator(0,64,3),input_size=(3,64,64))


# class Recover(nn.Module):
#     def __init__(self, nc, ndf, ngf):
#         super(Recover, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # (nc) x 64 x 64
#             nn.ReLU(True),
#
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # (ndf) x 32 x 32
#             nn.BatchNorm2d(ndf * 2),
#             nn.ReLU(True),
#
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # (ndf*2) x 16 x 16
#             nn.BatchNorm2d(ndf * 4),
#             nn.ReLU(True),
#
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # (ndf*4) x 8 x 8
#             nn.BatchNorm2d(ndf * 8),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, input):
#         return self.main(input)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Recover(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Recover, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# summary( GeneratorResNet((3,64,64),5).cuda(),input_size=(3,64,64))