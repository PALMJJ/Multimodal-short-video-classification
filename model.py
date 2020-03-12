import torch
from lmf import LMF
from hmc import HMC
import torch.nn as nn
from pytorch_transformers import BertTokenizer

class VideoClassificationModel(nn.Module):
    def __init__(self, L1_labels_num, L2_labels_num, L12_table):
        super(VideoClassificationModel, self).__init__()
        self.L1_labels_num = L1_labels_num
        self.L2_labels_num = L2_labels_num
        self.L12_table = L12_table
        self.lmf = LMF()
        self.hmc = HMC(feature_size=1024, L1_labels_num=self.L1_labels_num, L2_labels_num=L2_labels_num, L12_table=self.L12_table)

    def forward(self, images, texts, tokenizer):
        out_lmf = self.lmf(images, texts, tokenizer)
        L1, L2 = self.hmc(out_lmf)
        return L1, L2

if __name__ == '__main__':
    images = torch.randn(1, 3, 299, 299)
    texts = ['[CLS] 你是谁? 你是好人. [SEP]']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = VideoClassificationModel(5, 7, [[0], [2, 3, 4], [5], [6], [1]])
    model_image = model.lmf.inceptionresnetv2
    pretrained_dict = torch.load('inceptionresnetv2-520b38e4.pth')
    model_dict = model_image.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_image.load_state_dict(model_dict)

    L1, L2 = model(images, texts, tokenizer)
    print(L1.shape)
    print(L2.shape)
