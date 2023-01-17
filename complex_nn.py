import torch
import torch.nn as nn

'''
Source: https://github.com/huyanxin/DeepComplexCRN/blob/master/complexnn.py
'''

class ComplexLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0., bidirectional: bool = False):
        super().__init__()

        ## Model components
        self.lstm_re = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        self.lstm_im = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, x):
        rr, _ = self.lstm_re(x[..., 0])
        ii, _ = self.lstm_im(x[..., 1])
        real = rr - ii
        ri, _ = self.lstm_re(x[..., 1])
        ir, _ = self.lstm_im(x[..., 0])
        imaginary = ri - ir
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias)
        self.linear_im = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        real = self.linear_re(x[..., 0]) - self.linear_im(x[..., 1])
        imaginary = self.linear_re(x[..., 1]) + self.linear_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 **kwargs):
        super().__init__()

        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 **kwargs):
        super().__init__()

        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, **kwargs):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output