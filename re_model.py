from __future__ import print_function
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self,ngpu,ndf,nc):
        super( Extractor,self).__init__()
        self.ngpu=ngpu
        self.main=nn.Sequential(
            nn.Conv2d(nc,ndf,5,2,2,bias=False),
            nn.LeakyReLU(0.2,inplace=True),          # 64*32*32

            nn.Conv2d(ndf,ndf*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),          # 128*16*16

            nn.Conv2d(ndf*2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 256*8*8

            nn.Conv2d(ndf*4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 512*4*4
        )
        self.fc=nn.Sequential(
            nn.Linear(512*4*4,50),
            nn.Tanh()
        )
    def forward(self,input):
        feature = self.main(input)
        return self.fc(feature.view(-1,512 * 4 * 4))

# summary( Extractor(0,64,3),input_size=(3,64,64))

class Generator(nn.Module):
    def __init__(self, ngpu,num,ngf):
        super(Generator, self).__init__()
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
        feature=self.fc(input.view(-1,50*1*1))
        feature=feature.view(-1,512,4,4)
        feature=self.main(feature)

        return feature

# summary( Generator(0,128,64),input_size=(128,1,1))


class Discriminator(nn.Module):
    def __init__(self,ngpu,ndf,nc):
        super( Discriminator,self).__init__()
        self.ngpu=ngpu
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc=nn.Sequential(
            nn.Linear(512*4*4,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        feature1=self.conv1(input)
        feature2=self.conv2(feature1)
        feature3=self.conv3(feature2)
        feature4= self.conv4(feature3)
        feature5=self.fc(feature4.view(-1,512*4*4))
        return feature1,feature2,feature3,feature4,feature5

# summary( Discriminator2(0,64,3),input_size=(3,64,64))


#  数据提取器

class TExtractor(nn.Module):
    def __init__(self,ndf,nc):
        super( TExtractor,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),          # 64*32*32

            nn.Conv2d(ndf,ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),          # 128*16*16

            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 256*8*8

            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 512*4*4
        )
        self.fc=nn.Sequential(
            nn.Linear(512*4*4,100),
            nn.Tanh()
        )

    def forward(self,input):
        feature = self.main(input)
        return self.fc(feature.view(-1,512 * 4 * 4))