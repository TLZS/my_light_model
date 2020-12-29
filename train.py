from __future__ import print_function
from model.model import Discriminator
from model.model import Encoder
from model.model import Decoder
import torch
import torch.nn as nn
from Loss_Function import Encoder_loss
from Loss_Function import Decoder_loss
import torch.optim as optim
import torchvision.datasets as dset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time

# 数据集的根目录
dataroot='.//train'
batch_size=64
image_size=64
nc=3
ngf=64
ndf = 64
num= 200
num_epochs = 101
lr = 0.0002
beta1 = 0.5
# 可用的GPU数量。使用0表示CPU模式。
ngpu=1

# 创建数据集
dataset=dset.ImageFolder(root=dataroot,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netE=Encoder( ngpu, ndf, nc).cuda()
if (device.type == 'cuda') and (ngpu > 1):
    netE= nn.DataParallel(netE, list(range(ngpu)))
netE.apply(weights_init)

netG=Decoder(ngpu,num,ngf).cuda()
if (device.type == 'cuda') and (ngpu > 1):
    netG= nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

netD=Discriminator(ngpu,ndf,nc).cuda()
if (device.type == 'cuda') and (ngpu > 1):
    netD= nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

# 损失函数和优化器
criterionE=Encoder_loss
criterionG=Decoder_loss
criterionD=nn.BCELoss()

# 优化器
optimizerE=optim.Adam(netE.parameters(),lr=lr,betas=(beta1,0.999))
optimizerD=optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG=optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))
real_label=1
fake_label=0
E_losses=[]
G_losses=[]   # 计算生成器损失()
D_losses=[]
iters=0

# checkpoint = torch.load('./model/checkpoints/checkpointEG.tar')
# netE.load_state_dict(checkpoint['modelE_state_dict'])
# optimizerE.load_state_dict(checkpoint['optimizerE_state_dict'])
# netG.load_state_dict(checkpoint['modelG_state_dict'])
# optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
# netD.load_state_dict(checkpoint['modelD_state_dict'])
# optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
# epoch = checkpoint['epoch']
# print(epoch)
t1 = time.time()
print('显示程序开始的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
for epoch in range(num_epochs):
    for i,data in enumerate(dataloader,0):
        netE.zero_grad()
        real_cpu = data[0].to(device)
        E_out=netE(real_cpu)
        bit_z = (2 * torch.rand(real_cpu.size(0), 100) - 1).cuda()
        z = torch.cat([E_out,bit_z], dim=1)
        G_out=netG(z)
        errE=criterionE(E_out)
        errE.backward(retain_graph=True)
        E_x= E_out.mean().item()
        optimizerE.step()

        netD.zero_grad()                    # 更新D
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # 前向传播真实图片让真实图片通过D
        output = netD(real_cpu).view(-1)
        errD_real = criterionD(output, label)
        # 计算D反向传播梯度
        errD_real.backward()
        D_x = output.mean()

        # 在假图片中训练
        fake = netG(z)
        b_size_f=fake.size(0)
        label_f= torch.full((b_size_f,), fake_label, device=device)
        # 用D来判别假图片
        output = netD(fake.detach()).view(-1)     # detach()
        # 计算D在假图中的损失
        errD_fake = criterionD(output, label_f)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        # 所有损失加到一块
        errD = errD_fake + errD_real
        optimizerD.step()

        # 跟新G
        netG.zero_grad()
        label.fill_(real_label)
        #  Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # 基于这个输出计算G的损失
        errG = criterionG(real_cpu,fake,output)
        # 计算G的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # 输出训练数据
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_E: %.4f\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errE.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        # 保存损失用于画图
        E_losses.append(errE.item())
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (i % 100 == 0):
            with torch.no_grad():
                fake=netG(z).detach().cpu()
                ims = vutils.make_grid(fake, padding=2, normalize=True)  # make_grid可以使得图像按照网格进行排列
                torchvision.utils.save_image(ims, 'image/train/fake_bit_is_100/'+str(i)+'.jpg', nrow=8, padding=2,
                                             normalize=False, range=None,
                                             scale_each=False)

        iters += 1

    if epoch % 50 == 0 and epoch != 0:
        torch.save(netE, '%s/netE_epoch_%d.pth' % ('./model/checkpoints_EG_bit100_epoch100', epoch))
        torch.save(netG, '%s/netG_epoch_%d.pth' % ('./model/checkpoints_EG_bit100_epoch100', epoch))


    torch.save({
        'epoch': epoch,
        'modelE_state_dict': netE.state_dict(),
        'modelG_state_dict': netG.state_dict(),
        'modelD_state_dict': netD.state_dict(),
        'optimizerE_state_dict': optimizerE.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),

    }, './model/checkpoints_EG_bit100_epoch100/checkpointEGD.tar')

t2 = time.time()
print('显示程序结束的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print("用时：%.6fs" % (t2 - t1))