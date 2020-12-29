import torch.nn as nn
import torch
from model.re_model import TExtractor
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model.model import Encoder
from model.model import Decoder
import torchvision.utils as vutils
import torchvision
from common import I2N
from common.N2I import N2S
import time
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

dataroot='.//train'
ndf=64
nc=3
num = 100
ngf=64
image_size=64
batch_size=64
beta1=0.5
num_epochs= 201
ngpu=1

dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last= True)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netE = torch.load('./model/checkpoint_EG2/netE_epoch_100.pth')
netG = torch.load('./model/checkpoint_EG2/netG_epoch_100.pth')

# netE=Encoder( ngpu, ndf, nc).cuda()
# netG=Decoder(ngpu,num,ngf).cuda()
#
# checkpoint = torch.load('./model/checkpoints_EG/checkpointEGD.tar')
# netE.load_state_dict(checkpoint['modelE_state_dict'])
# netG.load_state_dict(checkpoint['modelG_state_dict'])
netE.eval()
netG.eval()
netT = TExtractor(64, 3).cuda()
netT.apply(weights_init)

if (device.type == 'cuda') and (ngpu > 1):
    netT = nn.DataParallel(netT, list(range(ngpu)))

optimizerT = torch.optim.Adam(netT.parameters(), lr=0.00002, betas=(0.5, 0.999))   # 0.0002  0.00002
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerT , 'max',factor=0.5,patience=10)
mse = nn.MSELoss()
batches_done = 0

# checkpoint = torch.load('./checkpoints/checkpointT.tar')
# netT.load_state_dict(checkpoint['model_state_dict'])
# optimizerT.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
#
# netE.train()
t1 = time.time()
print('显示程序开始的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        optimizerT.zero_grad()
        netT.zero_grad()
        real_cpu = data[0].to(device)
        E_out = netE(real_cpu)

        bit_z = (2 * torch.rand(real_cpu.size(0), 50) - 1).cuda()
        z = torch.cat([E_out, bit_z], dim=1)

        G_out = netG(z.detach())  # 生成假图像
        T_out = netT(G_out.detach())  # 提取出噪声向量
        errT = mse(T_out, z)
        errT.backward()
        optimizerT.step()

        if i % 200 == 0:
            with torch.no_grad():
                x = real_cpu
                out1 = netE(x)
                bit_z = I2N.z.cuda()
                z = torch.cat([out1, bit_z], dim=1)

                fake = netG(z).detach()

                # ims = vutils.make_grid(fake, padding=2, normalize=True)  # make_grid可以使得图像按照网格进行排列
                # torchvision.utils.save_image(ims, 'image/train/fake_bit_is_100/' + str(i) + '.jpg', nrow=8, padding=2,
                #                              normalize=False, range=None,
                #                              scale_each=False)

                out2 = netT(fake)
                r_image_z, r_bit_z = out2.split(50, dim=1)

                gz = N2S(r_bit_z)
                ori_z = I2N.s
                p = 0
                for w in range(gz.shape[0]):
                    if (gz[w] == ori_z[w]):
                        p = p + 1
                num = p
                acc = num / 3200      # acc = num / 3200
                writer.add_scalar('Acc', acc, batches_done)
                batches_done += 200
                print("[Epoch %d/%d] [Batch %d/%d] [Acc: %f] [lr: %f]" % (
                    epoch, num_epochs, i, len(dataloader), acc,optimizerT.state_dict()['param_groups'][0]['lr']))

    if epoch % 5 == 0 and epoch != 0:
        torch.save(netT, '%s/netT_epoch_%d.pth' % ('./model/checkpoints_T_epoch100', epoch))

    torch.save({
            'epoch': epoch,
            'model_state_dict': netT.state_dict(),
            'optimizer_state_dict': optimizerT.state_dict()
            }, './model/checkpoints_T_epoch100/checkpointT.tar')

    # scheduler.step(acc)


t2 = time.time()
print('显示程序结束的时间:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print("用时：%.6fs" % (t2 - t1))