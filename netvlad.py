import torch
import torch.nn as nn
import torch.nn.functional as F
from inceptionresnetv2 import InceptionResNetV2

class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=1536, alpha=100.0, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.fc = nn.Linear(98304, 1024)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) -  self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = self.fc(vlad)
        return vlad

if __name__ == '__main__':
    model1 = InceptionResNetV2()
    pretrained_dict = torch.load('inceptionresnetv2-520b38e4.pth')
    model_dict = model1.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model1.load_state_dict(model_dict)
    x = torch.randn(1, 3, 299, 299)
    model2 = NetVLAD()
    out = model1(x)
    out = model2(out)
    print(out.shape) # torch.Size([1, 1024])
