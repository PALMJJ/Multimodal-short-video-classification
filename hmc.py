import torch
import torch.nn as nn
import torch.nn.functional as F

class HMC(nn.Module):
    def __init__(self, feature_size, L1_labels_num, L2_labels_num, L12_table, mlp_hidden_size=1024, mask_value=-100):
        super(HMC, self).__init__()
        self.mlp_hidden_size = mlp_hidden_size
        self.mask_value = mask_value
        self.feature_size = feature_size
        self.L1_labels_num = L1_labels_num
        self.L2_labels_num = L2_labels_num
        self.L12_table = L12_table

        assert len(L12_table) == L1_labels_num
        assert self.check_L12_table(L12_table)

        self.fc_L1_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L1_2 = nn.Linear(self.mlp_hidden_size, self.L1_labels_num)
        self.fc_L2_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L2_2 = nn.Linear(2 * self.mlp_hidden_size, self.L2_labels_num)

    def check_L12_table(self, L12_table):
        L2_labels = [num for lst in L12_table for num in lst]
        assert len(L2_labels) == self.L2_labels_num
        for i in range(self.L2_labels_num):
            if i not in L2_labels:
                return False
        return True

    def forward(self, x):
        assert len(x.shape) == 2

        L1 = F.relu(self.fc_L1_1(x))
        L2 = F.relu(self.fc_L2_1(x))
        L2 = torch.cat((L1, L2), dim=1)
        L1 = F.relu(self.fc_L1_2(L1))
        L1 = F.softmax(L1, dim=1)
        L2 = F.relu(self.fc_L2_2(L2))

        L1_label = L1.argmax(dim=1)
        mask = torch.ones_like(L2) * self.mask_value

        for i, element in enumerate(L1_label):
            idx = element.item()
            mask[i, self.L12_table[idx]] = 0

        L2 += mask
        L2 = F.softmax(L2, dim=1)

        return L1, L2

def hmc_loss(L1, L2, L1_gt, L2_gt, Lambda=0.5, Beta=0.5):
    Lambda = Lambda
    Beta = Beta

    batch_num = L1.shape[0]
    Y1 = L1[torch.arange(batch_num), L1_gt]
    Y2 = L2[torch.arange(batch_num), L2_gt]

    L1_loss = - Y1.log().mean()
    L2_loss = - Y2.log().mean()
    LH_loss = torch.max(Y2 - Y1, torch.zeros_like(Y1)).mean()

    return L1_loss + Lambda * L2_loss + Beta * LH_loss

if __name__ == '__main__':
    x = torch.randn(1, 1024)
    model = HMC(1024, 5, 7, [[0], [2, 3, 4], [5], [6], [1]])
    L1, L2 = model(x)
    L1_gt = torch.Tensor([0, 1, 1]).long()
    L2_gt = torch.Tensor([0, 2, 4]).long()
    loss = hmc_loss(L1, L2, L1_gt, L2_gt)
    print(L1.shape)
    print(L2.shape)
    print(loss)
    """
    torch.Size([1, 5])
    torch.Size([1, 7])
    tensor(51.7262, grad_fn=<AddBackward0>)
    """