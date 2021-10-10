import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader



class resblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(resblock, self).__init__()
        self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)
        self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(ch_out)
        self.ch_trans = nn.Sequential()
        if ch_in != ch_out:
            self.ch_trans = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                                          nn.BatchNorm2d(ch_out))
        # ch_trans表示通道数转变。因为要做short_cut,所以x_pro和x_ch的size应该完全一致

    def forward(self, x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))

        # short_cut:
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16))
        self.block1 = resblock(16, 32, 3)
        self.block2 = resblock(32, 64, 3)
        self.block3 = resblock(64, 128, 2)
        self.block4 = resblock(128, 256, 2)
        self.outlayer = nn.Linear(256 * 3 * 3, num_class)  # 这个256*3*3是根据forward中x经过4个resblock之后来决定的

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.reshape(x.size(0), -1)
        result = self.outlayer(x)
        return result


