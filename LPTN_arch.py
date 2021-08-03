import os
import torch.nn as nn
import torch
WORK_PATH = os.getcwd()
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3) # x=(32,3,128,128) 在i=3通道相加 cc=(32,3,128,256)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3]) # (32,3,256,128) 相当与resize
        cc = cc.permute(0, 1, 3, 2) # 将tensor的维度换位 (32,3,128,256)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3) # (32,3,128,512)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2) # (32,3,256,256)
        x_up = cc.permute(0, 1, 3, 2) # (32,3,256,256)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect') # (32,3,260,260)
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1]) # (32,3,256,256) kernel.shape=(3,1,5,5)
        return out

    def pyramid_decom(self, img):
        current = img # 获取图像 (32,3,256,256)
        pyr = [] # 建立一个空的列表
        for _ in range(self.num_high): # self.num_high=3
            filtered = self.conv_gauss(current, self.kernel) # (32,3,256,256) self.kernel=(3,1,5,5)
            down = self.downsample(filtered) # (32,3,128,128)
            up = self.upsample(down) # (32,3,256,256)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff) # 高频分量：[(32,3,256,256), (32,3,128,128), (32,3,64,64)]
            current = down # (32,3,128,128)
        pyr.append(current) # 加入低频分量： [(32,3,256,256), (32,3,128,128), (32,3,64,64), (32,3,32,32)]
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image
