import torch
import torch.nn as nn
from netvlad import NetVLAD
import torch.nn.functional as F
from bert import get_tokens, TextNet
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from inceptionresnetv2 import InceptionResNetV2
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LMF(nn.Module):
    def __init__(self, output_dim, rank, hidden_dims=32, use_softmax=False):
        super(LMF, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        self.hidden_dims = hidden_dims
        self.use_softmax = use_softmax
        self.inceptionresnetv2 = InceptionResNetV2()
        self.netvlad = NetVLAD()
        self.textnet = TextNet()

        self.image_factor = Parameter(torch.Tensor(self.rank, self.hidden_dims, self.output_dim).to(device))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.hidden_dims, self.output_dim).to(device))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank).to(device))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim).to(device))

        xavier_normal_(self.image_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, images, texts, tokenizer):
        images_out = self.inceptionresnetv2(images)
        images_out = self.netvlad(images_out)

        tokens, segments, input_masks = get_tokens(texts, tokenizer)
        texts_out = self.textnet(tokens, segments, input_masks)

        fusion_image = torch.matmul(images_out, self.image_factor)
        fusion_text = torch.matmul(texts_out, self.text_factor)
        fusion_zy = fusion_image * fusion_text

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output

if __name__ == '__main__':
    images = torch.randn(1, 3, 299, 299)
    texts = ['[CLS] 你好吗? [SEP]']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    lmf = LMF(output_dim=32, rank=4, use_softmax=False)
    output = lmf(images, texts, tokenizer)
    print(output.shape)