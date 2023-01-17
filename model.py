import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from complex_nn import ComplexConv2d, ComplexBatchNorm2d, ComplexConvTranspose2d


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(in_channel=dim, out_channel=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ComplexConv2d(in_channel=dim, out_channel=dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        y = self.conv(x)
        return F.relu(x + y)


class CBAM(nn.Module):
    def __init__(self, in_dim, out_dim, ratio):
        super(CBAM, self).__init__()

        self.ratio = ratio
        self.linear_r = nn.Linear(in_dim, in_dim // ratio)
        self.linear = nn.Linear(in_dim // ratio, in_dim)
        self.conv = ComplexConv2d(in_channel=in_dim, out_channel=out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, Ti, Fr, _ = x.shape
        x_abs = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        x_ = x_abs.mean((2, 3))
        x_ = torch.relu(self.linear_r(x_))
        attention = torch.sigmoid(self.linear(x_)).reshape(B, C, 1, 1, 1)
        x = self.conv(x * attention)
        return x


class QKV(nn.Module):
    def __init__(self, K, C):
        super(QKV, self).__init__()

        self.conv = ComplexConv2d(in_channel=K * C, out_channel=C, kernel_size=3, stride=1, padding=1)

    def forward(self, q, k, v=None):
        B, C, Fr, T, _ = q.shape
        B, K, C, Fr, T, _ = k.shape

        q_ = torch.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2).reshape(B, C * Fr, T)
        k_ = torch.sqrt(k[..., 0] ** 2 + k[..., 1] ** 2).reshape(B, K, C * Fr, T)

        q_ = q_.permute(0, 2, 1).reshape(B * T, C * Fr)
        k_ = k_.permute(0, 3, 1, 2).reshape(B * T, K, C * Fr)

        score = torch.bmm(k_, q_.unsqueeze(-1)).squeeze(-1).reshape(B, T, K)
        score = torch.softmax(score, dim=-1)
        score = score.permute(0, 2, 1).reshape(B, K, 1, 1, T, 1)

        return self.conv((k * score).reshape(B, K * C, Fr, T, 2))


class ComplexUnet(nn.Module):
    def __init__(self):
        super(ComplexUnet, self).__init__()

        ### encoder
        self.conv_2 = nn.Sequential(
            ComplexConv2d(in_channel=1, out_channel=32, kernel_size=(7, 5), stride=(2, 1), padding=(2, 2)),
            ComplexBatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            ComplexConv2d(in_channel=32, out_channel=64, kernel_size=(6, 5), stride=(2, 1), padding=(2, 2)),
            ComplexBatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            ComplexConv2d(in_channel=64, out_channel=128, kernel_size=(6, 5), stride=(2, 1), padding=(2, 2)),
            ComplexBatchNorm2d(128),
            nn.LeakyReLU()
        )

        ### qkv: K, C
        self.qkv_2 = QKV(5, 32)
        self.qkv_3 = QKV(5, 64)
        self.qkv_4 = QKV(5, 128)

        ## channel_attention
        self.cbam_2 = CBAM(32 * 2, 32, 4)
        self.cbam_3 = CBAM(64 * 2, 64, 8)
        self.cbam_4 = CBAM(128 * 2, 128, 16)

        ### residual
        self.residual_1 = ResidualBlock(128)
        self.residual_2 = ResidualBlock(128)
        self.residual_3 = ResidualBlock(128)
        self.residual_4 = ResidualBlock(128)

        ### decoder
        self.deconv_1 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=128 * 2, out_channel=64, kernel_size=(6, 5), stride=(2, 1), padding=(2, 2)),
            ComplexBatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.deconv_2 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=64 * 2, out_channel=32, kernel_size=(6, 5), stride=(2, 1), padding=(2, 2)),
            ComplexBatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.deconv_3 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=32 * 2, out_channel=1, kernel_size=(7, 5), stride=(2, 1), padding=(2, 2)),
        )


    def select(self, ref, idx):
        B, C, Fr, T_, _ = ref.shape
        B, T, K = idx.shape

        ref_ = torch.stack([ref[i, :, :, idx[i], :] for i in range(B)], dim=0)
        ref_ = ref_.permute(0, 1, 4, 2, 3, 5).reshape(B, K, C, Fr, T, 2)

        return ref_


    def forward(self, x, ref, idx):

        input = x

        B, Fr, T, _ = x.shape
        _, _, T_, _ = ref.shape

        x = x.reshape(B, 1, Fr, T, 2)
        ref = ref.reshape(B, 1, Fr, T_, 2)

        skips = []

        ### encoder_1
        x = self.conv_2(x)
        ref = self.conv_2(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_2(x, ref_)
        skip = torch.cat([x, ref_], dim=1)
        skip = self.cbam_2(skip)
        skips.append(skip)

        ### encoder_2
        x = self.conv_3(x)
        ref = self.conv_3(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_3(x, ref_)
        skip = torch.cat([x, ref_], dim=1)
        skip = self.cbam_3(skip)
        skips.append(skip)

        ### encoder_3
        x = self.conv_4(x)
        ref = self.conv_4(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_4(x, ref_)
        skip = torch.cat([x, ref_], dim=1)
        skip = self.cbam_4(skip)
        skips.append(skip)


        ### res block
        x = skip
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)

        ### decoder
        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_1(x)

        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_2(x)

        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_3(x)

        x = x.squeeze(1)

        mag_mask = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        phase_mask = x / (mag_mask.unsqueeze(-1) + 1e-12)
        mag_mask = torch.tanh(mag_mask).unsqueeze(-1)

        return input * mag_mask * phase_mask


class ComplexUnet_(nn.Module):
    def __init__(self):
        super(ComplexUnet_, self).__init__()
        ### encoder
        self.conv_1 = nn.Sequential(
            ComplexConv2d(in_channel=1, out_channel=32, kernel_size=(7, 5), stride=(2, 1), padding=(2, 2), dilation=(1, 1)),
            ComplexBatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            ComplexConv2d(in_channel=32, out_channel=64, kernel_size=(6, 5), stride=(2, 1), padding=(2, 4), dilation=(1, 2)),
            ComplexBatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            ComplexConv2d(in_channel=64, out_channel=128, kernel_size=(6, 5), stride=(2, 1), padding=(2, 8), dilation=(1, 4)),
            ComplexBatchNorm2d(128),
            nn.LeakyReLU()
        )

        ### qkv: K, C
        self.qkv_1 = QKV(5, 32)
        self.qkv_2 = QKV(5, 64)
        self.qkv_3 = QKV(5, 128)

        ## channel_attention
        self.cbam_1 = CBAM(32 * 3, 32, 4)
        self.cbam_2 = CBAM(64 * 3, 64, 8)
        self.cbam_3 = CBAM(128 * 3, 128, 16)

        ### residual
        self.residual_1 = ResidualBlock(128)
        self.residual_2 = ResidualBlock(128)
        self.residual_3 = ResidualBlock(128)
        self.residual_4 = ResidualBlock(128)

        ### decoder
        self.deconv_1 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=128 * 2, out_channel=64, kernel_size=(6, 5), stride=(2, 1), padding=(2, 8), dilation=(1, 4)),
            ComplexBatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.deconv_2 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=64 * 2, out_channel=32, kernel_size=(6, 5), stride=(2, 1), padding=(2, 4), dilation=(1, 2)),
            ComplexBatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.deconv_3 = nn.Sequential(
            ComplexConvTranspose2d(in_channel=32 * 2, out_channel=1, kernel_size=(7, 5), stride=(2, 1), padding=(2, 2), dilation=(1, 1)),
        )


    def select(self, ref, idx):
        B, C, Fr, T_, _ = ref.shape
        B, T, K = idx.shape

        ref_ = torch.stack([ref[i, :, :, idx[i], :] for i in range(B)], dim=0)
        ref_ = ref_.permute(0, 1, 4, 2, 3, 5).reshape(B, K, C, Fr, T, 2)

        return ref_


    def forward(self, x, x_, ref, idx):
        input = x

        B, Fr, T, _ = x.shape
        _, _, T_, _ = ref.shape

        x = x.reshape(B, 1, Fr, T, 2)
        x_ = x_.reshape(B, 1, Fr, T, 2)
        ref = ref.reshape(B, 1, Fr, T_, 2)

        skips = []

        ### encoder_1
        x = self.conv_1(x)
        x_ = self.conv_1(x_)
        ref = self.conv_1(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_1(x_, ref_)
        skip = torch.cat([x, x_, ref_], dim=1)
        skip = self.cbam_1(skip)
        skips.append(skip)

        ### encoder_2
        x = self.conv_2(x)
        x_ = self.conv_2(x_)
        ref = self.conv_2(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_2(x_, ref_)
        skip = torch.cat([x, x_, ref_], dim=1)
        skip = self.cbam_2(skip)
        skips.append(skip)

        ### encoder_3
        x = self.conv_3(x)
        x_ = self.conv_3(x_)
        ref = self.conv_3(ref)
        ref_ = self.select(ref, idx)
        ref_ = self.qkv_3(x_, ref_)
        skip = torch.cat([x, x_, ref_], dim=1)
        skip = self.cbam_3(skip)
        skips.append(skip)

        ### res block
        x = skip
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)

        ### decoder
        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_1(x)

        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_2(x)

        skip = skips.pop()
        x = torch.cat((x, skip), dim=1)
        x = self.deconv_3(x)

        x = x.squeeze(1)

        mag_mask = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        phase_mask = x / (mag_mask.unsqueeze(-1) + 1e-12)
        mag_mask = torch.tanh(mag_mask).unsqueeze(-1)

        return input * mag_mask * phase_mask


# if __name__ == '__main__':
#     pass