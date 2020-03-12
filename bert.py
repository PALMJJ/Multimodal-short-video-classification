import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_tokens(texts, tokenizer):
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))
    max_len = max([len(single) for single in tokens])
    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    tokens = torch.tensor(tokens)
    segments = torch.tensor(segments)
    input_masks = torch.tensor(input_masks)
    return tokens.to(device), segments.to(device), input_masks.to(device)

class TextNet(nn.Module):
    def __init__(self,  code_length=1024):
        super(TextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained('/home/hengyuli/cross-modal/bert_config.json')
        self.textExtractor = BertModel.from_pretrained('/home/hengyuli/cross-modal/pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        hash_features = self.fc(text_embeddings)
        hash_features = self.tanh(hash_features)
        return hash_features

if __name__ == '__main__':
    model_text = TextNet(code_length=1024)
    texts = ['[CLS] 你是谁? [SEP]', '[CLS] 你是好人. [SEP]']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokens, segments, input_masks = get_tokens(texts, tokenizer)
    out_text = model_text(tokens, segments, input_masks)
    print(out_text.shape) # torch.Size([2, 1024])
