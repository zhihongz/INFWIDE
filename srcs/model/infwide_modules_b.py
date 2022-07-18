from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------
# basic functions and modules
# --------------------------------------------

def pad2same_size(x1, x2):
    '''
    pad x1 or x2 to the same size (the size of the larger one)
    '''
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]

    if diffX == 0 and diffY == 0:
        return x1, x2

    if diffX >= 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, -diffY // 2, -diffY + diffY//2))
    elif diffX >= 0 and diffY < 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, 0, 0))
        x2 = nn.functional.pad(
            x2, (0, 0, -diffY // 2, -diffY + diffY//2))
    elif diffX < 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (0, 0, diffY // 2, diffY - diffY//2))
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, 0, 0))

    return x1, x2


def pad2size(x, size):
    '''
    pad x to given size
    x: N*C*H*W
    size: H'*W'
    '''
    diffX = size[1] - x.size()[3]
    diffY = size[0] - x.size()[2]

    if diffX == 0 and diffY == 0:
        return x

    if diffX >= 0 and diffY >= 0:
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                  diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY) //
              2, -diffX // 2: diffX + (-diffX)//2]
    elif diffX >= 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY)//2, :]
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2, 0, 0))
    elif diffX < 0 and diffY >= 0:
        x = x[:, :, :, -diffX // 2: diffX + (-diffX)//2]
        x = nn.functional.pad(x, (0, 0, diffY // 2, diffY - diffY//2))
    return x


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats,
                 kernel_size, stride, padding, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0, bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, bias=bias))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size,
                     padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class InBlock(nn.Module):
    def __init__(self, n_in, n_feats=32, kernel_size=5, padding=2, act=True):
        super(InBlock, self).__init__()
        # In block
        InLayers = [Conv(n_in, n_feats, kernel_size, padding=padding, act=act),
                    ResBlock(Conv, n_feats, kernel_size, padding=padding),
                    ResBlock(Conv, n_feats, kernel_size, padding=padding),
                    ResBlock(Conv, n_feats, kernel_size, padding=padding)]
        self.inBlock = nn.Sequential(*InLayers)

    def forward(self, x):
        return self.inBlock(x)


class OutBlock_a(nn.Module):
    def __init__(self, n_feats=32, n_resblock=3, kernel_size=5, padding=2):
        super(OutBlock_a, self).__init__()
        OutLayers = [ResBlock(Conv, n_feats, kernel_size, padding=padding)
                     for _ in range(n_resblock)]
        self.outLayers = nn.Sequential(*OutLayers)

    def forward(self, x):
        return self.outLayers(x)


class OutBlock_b(nn.Module):
    def __init__(self, n_feats=32, n_out=3, kernel_size=5, padding=2):
        super(OutBlock_b, self).__init__()
        OutLayers = [ResBlock(Conv, n_feats, kernel_size, padding=padding),
                     Conv(n_feats, n_feats, kernel_size, padding=padding),
                     Conv(n_feats, n_out, kernel_size, padding=padding)]

        self.outLayers = nn.Sequential(*OutLayers)

    def forward(self, x):
        return self.outLayers(x)



# --------------------------------------------
# Feature space refine modules
# --------------------------------------------


class RefineUnet(nn.Module):
    def __init__(self, n_feats, n_resblock=3, kernel_size=5, padding=2, act=True):
        super(RefineUnet, self).__init__()

        # encoder1
        Encoder_first = [Conv(n_feats, n_feats*2, kernel_size, padding=padding, stride=2, act=act),
                         ResBlock(Conv, n_feats*2, kernel_size,
                                  padding=padding),
                         ResBlock(Conv, n_feats*2, kernel_size,
                                  padding=padding),
                         ResBlock(Conv, n_feats*2, kernel_size, padding=padding)]
        # encoder2
        Encoder_second = [Conv(n_feats*2, n_feats*4, kernel_size, padding=padding, stride=2, act=act),
                          ResBlock(Conv, n_feats*4, kernel_size,
                                   padding=padding),
                          ResBlock(Conv, n_feats*4, kernel_size,
                                   padding=padding),
                          ResBlock(Conv, n_feats*4, kernel_size, padding=padding)]
        # decoder2
        Decoder_second = [
            ResBlock(Conv, n_feats*4, kernel_size, padding=padding) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(
            n_feats*4, n_feats*2, kernel_size=3, padding=1, output_padding=1, act=act))
        # decoder1
        Decoder_first = [
            ResBlock(Conv, n_feats*2, kernel_size, padding=padding) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(
            n_feats*2, n_feats, kernel_size=3, padding=1, output_padding=1, act=act))

        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)

    def forward(self, x):
        n, c, h, w = x.shape
        e1 = self.encoder_first(x)
        e2 = self.encoder_second(e1)
        d2 = self.decoder_second(e2)
        e1, d2 = pad2same_size(e1, d2)
        d1 = self.decoder_first(e1+d2)
        x, d1 = pad2size(x, [h, w]), pad2size(d1, [h, w])
        y_in = x+d1
        return y_in



class FSRModule(nn.Module):
    '''
    Feature space refine network
    '''

    def __init__(self, n_colors, n_resblock=3, n_in=16, n_feats=32, kernel_size=5, padding=2):
        super(FSRModule, self).__init__()

        # self.InBlock1 = InBlock(n_in=n_in, n_feats=n_feats,
        #                         kernel_size=kernel_size, padding=padding, act=True)
        # self.InBlock2 = InBlock(n_in=n_in + n_feats, n_feats=n_feats,
        #                         kernel_size=kernel_size, padding=padding, act=True)
        self.InBlock2 = InBlock(n_in=n_in, n_feats=n_feats,
                                kernel_size=kernel_size, padding=padding, act=True)

        self.RefineUnet = RefineUnet(
            n_feats=n_feats, n_resblock=n_resblock, kernel_size=kernel_size, padding=padding, act=True)

        self.OutBlock_a = OutBlock_a(
            n_feats=n_feats, n_resblock=n_resblock, kernel_size=kernel_size, padding=padding)
        self.OutBlock_b = OutBlock_b(
            n_feats=n_feats, n_out=n_colors, kernel_size=kernel_size, padding=padding)

    def forward(self, xx):
        in_2 = self.InBlock2(xx)
        refine_2 = self.RefineUnet(in_2)
        out_b2 = self.OutBlock_b(self.OutBlock_a(refine_2))
        return out_b2



# --------------------------------------------
# XRF: cross residual fusion modules
# --------------------------------------------

class XResDown(nn.Module):
    '''
    cross residual downscale
    '''

    def __init__(self, n_in, n_out, n_conv=3, kernel_size=5, padding=2):
        super(XResDown, self).__init__()
        conv1 = []
        conv1.extend([Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=True) for _ in range(n_conv-2)])
        conv1.append(Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=False))
        conv2 = []
        conv2.extend([Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=True) for _ in range(n_conv-2)])
        conv2.append(Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=False))

        conv12 = [nn.ReLU(inplace=True), Conv(n_in, n_out, kernel_size,
                                              padding=padding, stride=2, act=True)]
        conv21 = [nn.ReLU(inplace=True), Conv(n_in, n_out, kernel_size,
                                              padding=padding, stride=2, act=True)]

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv12 = nn.Sequential(*conv12)
        self.conv21 = nn.Sequential(*conv21)

    def forward(self, x1, x2):
        x_mean = (x1+x2)/2
        x1_conv1 = self.conv1(x1)
        x2_conv2 = self.conv2(x2)
        # x12 = self.conv12(torch.cat((x1_conv1, x_mean), 1))
        # x21 = self.conv21(torch.cat((x2_conv2, x_mean), 1))
        x12 = self.conv12(x1_conv1 + x_mean)
        x21 = self.conv21(x2_conv2 + x_mean)
        return x12, x21


class XResUp(nn.Module):
    '''
    cross residual upscale
    '''

    def __init__(self, n_in, n_out, n_conv=3, kernel_size=5, padding=2):
        super(XResUp, self).__init__()
        conv1 = []
        conv1.extend([Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=True) for _ in range(n_conv-2)])
        conv1.append(Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=False))
        conv2 = []
        conv2.extend([Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=True) for _ in range(n_conv-2)])
        conv2.append(Conv(n_in, n_in, kernel_size,
                     padding=padding, stride=1, act=False))

        deconv12 = [nn.ReLU(inplace=True), Deconv(n_in, n_out, kernel_size=3,
                                                  padding=1, output_padding=1, act=True)]
        deconv21 = [nn.ReLU(inplace=True), Deconv(n_in, n_out, kernel_size=3,
                                                  padding=1, output_padding=1, act=True)]

        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.deconv12 = nn.Sequential(*deconv12)
        self.deconv21 = nn.Sequential(*deconv21)

    def forward(self, x1, x2):
        x_mean = (x1+x2)/2
        x1_conv1 = self.conv1(x1)
        x2_conv2 = self.conv2(x2)
        # x12 = self.deconv12(torch.cat((x1_conv1, x_mean), 1))
        # x21 = self.deconv21(torch.cat((x2_conv2, x_mean), 1))
        x12 = self.deconv12(x1_conv1 + x_mean)
        x21 = self.deconv21(x2_conv2 + x_mean)
        return x12, x21


class XRFUnet(nn.Module):
    def __init__(self, n_feats, n_conv=4, kernel_size=5, padding=2):
        super(XRFUnet, self).__init__()
        self.XRD1 = XResDown(n_in=n_feats, n_out=n_feats*2, n_conv=n_conv,
                             kernel_size=kernel_size, padding=padding)
        self.XRD2 = XResDown(n_in=n_feats*2, n_out=n_feats*4, n_conv=n_conv,
                             kernel_size=kernel_size, padding=padding)
        self.XRU2 = XResUp(n_in=n_feats*4, n_out=n_feats*2, n_conv=n_conv,
                           kernel_size=kernel_size, padding=padding)
        self.XRU1 = XResUp(n_in=n_feats*2, n_out=n_feats, n_conv=n_conv,
                           kernel_size=kernel_size, padding=padding)

    def forward(self, in1, in2):
        n, c, h, w = in1.shape
        e11, e12 = self.XRD1(in1, in2)
        e21, e22 = self.XRD2(e11, e12)
        d21, d22 = self.XRU2(e21, e22)
        e11, d21 = pad2same_size(e11, d21)
        e12, d22 = pad2same_size(e12, d22)
        d11, d12 = self.XRU1(e11+d21, e12+d22)
        d11, d12 = pad2size(d11, [h, w]), pad2size(d12, [h, w])

        y1, y2 = in1+d11, in2+d12
        return y1, y2


class XRFModule(nn.Module):
    '''
    cross residual fusion modules: one scale input, interpolate to two scales. feature-input merge for two scale
    '''

    def __init__(self, n_colors, n_resblock=3, n_conv=4, n_feats=32, kernel_size=5, padding=2):
        super(XRFModule, self).__init__()

        self.InBlock11 = InBlock(n_in=n_colors, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding, act=True)
        self.InBlock12 = InBlock(n_in=n_colors, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding, act=True)
        self.InBlock21 = InBlock(n_in=n_colors + n_feats, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding, act=True)
        self.InBlock22 = InBlock(n_in=n_colors + n_feats, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding, act=True)

        self.XRFUnet = XRFUnet(
            n_feats=n_feats, n_conv=n_conv, kernel_size=kernel_size, padding=padding)

        self.OutBlock_a1 = OutBlock_a(
            n_feats=n_feats, n_resblock=n_resblock, kernel_size=kernel_size, padding=padding)
        self.OutBlock_a2 = OutBlock_a(
            n_feats=n_feats, n_resblock=n_resblock, kernel_size=kernel_size, padding=padding)
        self.OutBlock_b = OutBlock_b(
            n_feats=2*n_feats, n_out=n_colors, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2):

        n, c, h, w = x2.shape

        # first scale, x0.5
        x1_d = F.interpolate(
            x1, (int(round(h/2)), int(round(w/2))), mode='bilinear')
        x2_d = F.interpolate(
            x2, (int(round(h/2)), int(round(w/2))), mode='bilinear')
        in11 = self.InBlock11(x1_d)
        in12 = self.InBlock12(x2_d)
        y11, y12 = self.XRFUnet(in11, in12)

        out_a11 = self.OutBlock_a1(y11)
        out_a12 = self.OutBlock_a2(y12)
        out_b1 = self.OutBlock_b(torch.cat((out_a11, out_a12), 1))

        # second scale, x1
        out_a11_up = F.interpolate(out_a11, (h, w), mode='bilinear')
        out_a12_up = F.interpolate(out_a12, (h, w), mode='bilinear')

        x1_cat = torch.cat((out_a11_up, x1), 1)
        x2_cat = torch.cat((out_a12_up, x2), 1)

        in21 = self.InBlock21(x1_cat)
        in22 = self.InBlock22(x2_cat)
        y21, y22 = self.XRFUnet(in21, in22)

        out_a21 = self.OutBlock_a1(y21)
        out_a22 = self.OutBlock_a2(y22)
        out_b2 = self.OutBlock_b(torch.cat((out_a21, out_a22), 1))

        return [out_b1, out_b2]


