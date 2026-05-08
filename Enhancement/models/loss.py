import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class CharbonnierLoss(nn.Module):


    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss



class CharbonnierLoss2(nn.Module):


    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class ShiftFreeLoss(nn.Module):


    def __init__(self, eps=1e-6, max_shift=10):

        super(ShiftFreeLoss, self).__init__()
        self.eps = eps
        self.max_shift = max_shift

        self.directions = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1)
        ]

    def forward(self, x, y):

        B, C, H, W = x.size()
        losses = []

        for dx, dy in self.directions:

            if dx != 0 or dy != 0:
                shift = torch.randint(1, self.max_shift + 1, (1,), device=x.device).item()
            else:
                shift = 0

            shift_x = dx * shift
            shift_y = dy * shift


            if shift_y < 0:
                y_start_y = -shift_y
                y_end_y = H
                x_start_y = 0
                x_end_y = H + shift_y
            elif shift_y > 0:
                y_start_y = 0
                y_end_y = H - shift_y
                x_start_y = shift_y
                x_end_y = H
            else:
                y_start_y = 0
                y_end_y = H
                x_start_y = 0
                x_end_y = H

            if shift_x < 0:
                y_start_x = -shift_x
                y_end_x = W
                x_start_x = 0
                x_end_x = W + shift_x
            elif shift_x > 0:
                y_start_x = 0
                y_end_x = W - shift_x
                x_start_x = shift_x
                x_end_x = W
            else:
                y_start_x = 0
                y_end_x = W
                x_start_x = 0
                x_end_x = W


            if x_end_y <= 0 or y_end_y <= 0 or x_end_x <= 0 or y_end_x <= 0:
                continue


            x_shifted = x[:, :, x_start_y:x_end_y, x_start_x:x_end_x]
            y_aligned = y[:, :, y_start_y:y_end_y, y_start_x:y_end_x]


            if x_shifted.size() != y_aligned.size():
                continue


            diff = x_shifted - y_aligned
            loss = torch.mean(torch.sqrt(diff * diff + self.eps))
            losses.append(loss)


        diff_orig = x - y
        loss_orig = torch.mean(torch.sqrt(diff_orig * diff_orig + self.eps))
        losses.append(loss_orig)

        if not losses:

            final_loss = loss_orig
        else:

            final_loss = torch.min(torch.stack(losses))

        return final_loss

class SSIMLoss(nn.Module):


    def __init__(self, window_size=4, size_average=True, val_range=1.0, channels=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channels = channels
        self.register_buffer('window', self.create_window(window_size, channels))

    def create_window(self, window_size, channels):


        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2.0 * (window_size // 2) ** 2))
        gauss /= gauss.sum()


        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window_2d /= window_2d.sum()


        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):

        if img1.size() != img2.size():
            raise ValueError("Input images must have the same dimensions.")

        N, C, H, W = img1.size()
        if C != self.channels:

            self.window = self.create_window(self.window_size, C).to(img1.device).type(img1.dtype)
            self.channels = C


        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=C)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2


        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=C) - mu1_mu2


        c1 = (0.01 * self.val_range) ** 2
        c2 = (0.03 * self.val_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /\
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if self.size_average:
            return (1 - ssim_map.mean()) * 1e6
        else:
            return (1 - ssim_map) * 1e6

import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()

        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):

            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):

            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss
