import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models


class EncoderSimple(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.elayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5, 2, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True)
        )

        self.elayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 5, 2, 2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True)
        )

        self.elayer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 5, 2, 2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True)
        )

        self.elayer4 = torch.nn.Sequential(
            torch.nn.Linear(256 * 64, 1024),
            torch.nn.ReLU(True)
        )

        self.elayer5 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(True)
        )

        self.elayer6 = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(True)
        )


    def forward(self, x):

        out = self.elayer1(x)
        out = self.elayer2(out)
        out = self.elayer3(out)
        out = out.view(out.size(0), -1)
        out = self.elayer4(out)
        out = self.elayer5(out)
        out = self.elayer6(out)

        return out


class DecoderSimple(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dlayer1 = torch.nn.Sequential( 
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True)
        )

        self.dlayer2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(True)
        )

        self.dlayer3 = torch.nn.Sequential(
            torch.nn.Linear(1024, 32 * 4 * 4 * 4),
            torch.nn.ReLU(True)
        )

        self.dlayer4 = torch.nn.Sequential(
            torch.nn.Linear(32 * 4 * 4 * 4, 64 * 8 * 8 * 8),
            torch.nn.ReLU(True)
        )

        self.dlayer5_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()

        )

        self.dlayer5_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )

        self.dlayer5_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64 + 32 * 2, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )

        self.dlayer5_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64 + 32 * 3, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.dlayer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )

        self.dlayer6_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )

        self.dlayer6_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32 + 16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )

        self.dlayer6_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32 + 16 * 2, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )

        self.dlayer6_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32 + 16 * 3, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )

        self.dlayer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )

        self.dlayer7_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.dlayer7_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16 + 8, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.dlayer7_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16 + 8 * 2, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.dlayer7_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16 + 8 * 3, 16, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )

        self.dlayer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.dlayer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )


    def forward(self, x):

        out = self.dlayer1(x)
        out = self.dlayer2(out)
        out = self.dlayer3(out)
        out = self.dlayer4(out)
        out = out.view(out.size(0), 64, 8, 8, 8)
        out1 = self.dlayer5_1(out)
        out2 = self.dlayer5_2(torch.cat((out, out1), 1))
        out3 = self.dlayer5_3(torch.cat((out, out1, out2), 1))
        out = self.dlayer5_4(torch.cat((out, out1, out2, out3), 1))
        out = self.dlayer5(out) 
        out1 = self.dlayer6_1(out)
        out2 = self.dlayer6_2(torch.cat((out, out1), 1))
        out3 = self.dlayer6_3(torch.cat((out, out1, out2), 1))
        out = self.dlayer6_4(torch.cat((out, out1, out2, out3), 1)) 
        out = self.dlayer6(out) 
        out1 = self.dlayer7_1(out)
        out2 = self.dlayer7_2(torch.cat((out, out1), 1))
        out3 = self.dlayer7_3(torch.cat((out, out1, out2), 1)) 
        out = self.dlayer7_4(torch.cat((out, out1, out2, out3), 1))
        out = self.dlayer7(out)
        out = self.dlayer8(out)
        
        return out * 4 - 2
        

class Refiner(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.rlayer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=(2, 2, 2)),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.rlayer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=(2, 2, 2)),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.rlayer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=(2, 2, 2)),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.rlayer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=(2, 2, 2)),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.rlayer5 = torch.nn.Sequential(
            torch.nn.Linear(8192*2, 2048),
            torch.nn.ReLU(True)
        )

        self.rlayer6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192*2),
            torch.nn.ReLU(True)
        )

        self.rlayer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )

        self.rlayer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.rlayer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )

        self.rlayer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        # bx1x64x64x64
        volumes_32_l = self.rlayer1(coarse_volumes) 
        # bx32x32x32x32
        volumes_16_l = self.rlayer2(volumes_32_l)
        # bx64x16x16x16
        volumes_8_l = self.rlayer3(volumes_16_l)
        # bx128x8x8x8
        volumes_4_l = self.rlayer4(volumes_8_l)
        # bx256x4x4x4
        flatten_features = self.rlayer5(volumes_4_l.view(-1, 8192 * 2))
        flatten_features = self.rlayer6(flatten_features)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 256, 4, 4, 4)
        # bx256x4x4x4
        volumes_8_r = volumes_8_l + self.rlayer7(volumes_4_r)
        # bx128x8x8x8
        volumes_16_r = volumes_16_l + self.rlayer8(volumes_8_r)
        # bx64x16x16x16
        volumes_32_r = volumes_32_l + self.rlayer9(volumes_16_r)
        # bx32x32x32x32
        volumes_64_r = (coarse_volumes + self.rlayer10(volumes_32_r)) * 0.5
        # bx1x64x64x64

        return volumes_64_r * 4 - 2

