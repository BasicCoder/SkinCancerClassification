import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter as P

from models.resnet import resnet34, resnet50

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B x (N) x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B x (N) x (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=nn.Conv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


class ResNet50_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Attn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)
        self.attn = Attention(2048)

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature = self.attn(x)

        attn_feature = self.attn_avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x


class ResNet50_Self_Attn(nn.Module):
    def __init__(self, pretrained=False, out_features=125):
        super(ResNet50_Self_Attn, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)
        self.attn = Self_Attn(2048, 'relu')

    def forward(self, x):
        # shape [N, C, H, W]
        x = self.model(x)

        attn_feature, p = self.attn(x)
        attn_feature = self.avgpool(attn_feature)
        attn_feature = attn_feature.view(attn_feature.size(0), -1)

        x = self.fc(attn_feature)
        return x

