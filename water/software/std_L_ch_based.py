import torch
from torch import nn
from kornia.color import rgb_to_lab,lab_to_rgb
from kornia.filters import gaussian_blur2d

import numpy as np
import torch
from torch import nn
from torch.nn import init
import cv2

ifch = 8


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(PALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y



class CPLayer(nn.Module):

    def __init__(self, channel=64,reduction=4):
        super().__init__()
        self.ca=CALayer(channel=channel,reduction=reduction)
        self.pa=PALayer(channel=channel,reduction=reduction)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()

        out=x*self.ca(x)
        out=out*self.pa(out)
        return out


class Conv_Atten_Block(nn.Module):
    def __init__(self):
        super(Conv_Atten_Block, self).__init__()
        self.conv1 = nn.Conv2d(ifch, ifch, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ifch, ifch, kernel_size=3, padding=3 // 2)
        self.attention = CPLayer(channel=ifch, reduction=4)

    def forward(self, x):
        res = x
        x2 = self.conv1(x)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x6 = self.attention(x4)
        out = x6 + res

        return out
    
def log_approx(x):
    # return torch.log(x)
    x = torch.clamp(x, min=1e-6)  # 避免 log(0)
    return 0.69 * (x - 1) - 0.33 * (x - 1) ** 2  # Pade Approximation

class ResIERes(nn.Module):
    def __init__(self, num_channels=3):
        super(ResIERes, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, ifch, kernel_size=3, padding=3 // 2)
        self.ca1 = Conv_Atten_Block()  # 保留注意力機制
        self.conv2 = nn.Conv2d(ifch, num_channels, kernel_size=3, padding=3 // 2)
        self.sig1 = nn.Sigmoid()

        self.conv3 = nn.Conv2d(num_channels, ifch, kernel_size=3, padding=3 // 2)
        self.ca2 = Conv_Atten_Block()  # 使用縮放因子減少計算量
        self.conv4 = nn.Conv2d(ifch, num_channels, kernel_size=3, padding=3 // 2)

        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        res = x

        # 第一段卷積塊 + 注意力機制
        x1 = self.conv1(x)
        x2 = self.ca1(x1)
        x3 = self.conv2(x2)

        x = x3 + res
        x = self.sig1(x)
        ##########################################################################################################
        R, G, B = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        R_Mean, G_Mean, B_Mean = R.mean(dim=[1, 2]), G.mean(dim=[1, 2]), B.mean(dim=[1, 2])
        K = R_Mean + G_Mean + B_Mean

        # 判斷條件 G_mean / K > 2/3
        condition = G_Mean * 3 > 2 * K
        if condition.any():
            x = torch.clamp(x, min=0, max=1)
        else:
            R_Mean = R_Mean.unsqueeze(-1).unsqueeze(-1)
            G_Mean = G_Mean.unsqueeze(-1).unsqueeze(-1)
            B_Mean = B_Mean.unsqueeze(-1).unsqueeze(-1)
            alpha = 1 - log_approx(torch.clamp(G_Mean - R_Mean, min=1e-6))

            # Process beta for the second condition
            if (G_Mean - B_Mean).mean() >= 0.1:
                beta = -log_approx(torch.clamp(G_Mean - B_Mean, min=1e-6))

                if (R_Mean - B_Mean).mean() <= 0.1:
                    beta = 1 / (1 + beta)
                elif (R_Mean - B_Mean).mean() > 0.1:
                    beta = 1 + beta
                    alpha = 0

                # Update B channel
                B = B + (G_Mean - B_Mean) * (1 - B) * G * beta

            # Process gamma for the third condition
            if (B_Mean - G_Mean).mean() > 0.1:
                gamma = -log_approx(torch.clamp(B_Mean - R_Mean, min=1e-6))
                G = G + (B_Mean - R_Mean) * (1 - G) * B * gamma

            # Update R channel
            R = R + (G_Mean - R_Mean) * (1 - R) * G * alpha

            x = torch.stack([R, G, B], dim=1)
            x = torch.clamp(x, min=0, max=1)

        # stdclamp47
        batch_size, channels, height, width = x.shape
        balanced_images = torch.zeros_like(x)
        # k = 2.0
        for b in range(batch_size):
            for c in range(channels):
                ch = x[b, c, :, :]
                mean = ch.mean()
                std = ch.std()

                # 計算上下限
                lower_limit = mean - 2.0 * std
                upper_limit = mean + 2.0 * std

                # clamp + normalize
                ch = torch.clamp(ch, min=lower_limit, max=upper_limit)
                ch = (ch - lower_limit) / (upper_limit - lower_limit + 1e-8)
                ch = torch.clamp(ch, 0, 1)

                balanced_images[b, c, :, :] = ch
        x = torch.clamp(balanced_images, min=0, max=1)

        #L_ch_based
        R, G, B = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        Y = 0.299 * R + 0.587 * G + 0.114 * B

        B, H, W = Y.shape
        Y_out = torch.zeros_like(Y)

        for b in range(B):
            y = Y[b]
            mean = y.mean()
            std = y.std()

            lower = mean - 2.0 * std
            upper = mean + 2.0 * std

            y = torch.clamp(y, lower, upper)
            y = (y - lower) / (upper - lower + 1e-8)
            Y_out[b] = y

        # 增強亮度後的 Y，用來放大 RGB（比例法）
        scale = Y_out / (Y + 1e-6)
        R = torch.clamp(R * scale, 0, 1)
        G = torch.clamp(G * scale, 0, 1)
        B = torch.clamp(B * scale, 0, 1)
        x = torch.stack([R, G, B], dim=1)

        ###############################################################################################
        # 第二段卷積塊 + 注意力機制
        x4 = self.conv3(x)
        x5 = self.ca2(x4)
        x6 = self.conv4(x5)

        x = x6 + res
        out = self.sig2(x)
        
        return out

