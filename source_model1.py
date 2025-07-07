import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x,  mask):
        "Pass the input (and mask) through each layer in turn."
        #loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x,  mask)
        return self.norm(x)
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, size2, self_attn1, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn1 = self_attn1
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer2 = clones(SublayerConnection(size2, dropout), 1)
        self.size = size
    def forward(self, x,mask):
        x = self.sublayer[0](x,lambda x: self.self_attn1(x, x, x,  mask))
        return self.sublayer[1](x,  self.feed_forward)
def attention(query, key, value,device,mask=None):
    "Compute 'Scaled Dot Product Attention'"
    padding_num = torch.ones_like(mask)
    padding_num = -2 ** 31 * padding_num.float()  # -inf
    scores = torch.matmul(query, key.transpose(-2, -1))
    # print(scores.shape)
    # print(mask.shape)
    # print(padding_num.shape)
    scores = torch.where(mask.to(device),scores .to(device), padding_num.to(device))
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model,device, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.linears = clones(nn.Linear(d_model, 128 * h), 3)
        self.W_o = nn.Linear(128 * h, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.ELU = nn.ELU()
        self.device=device
    def forward(self, query, key, value,mask=None):
        if mask is not None:
            #Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(-1,self.h, -1,-1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(nbatches, -1, self.h, 128 * self.h).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]
        queries = self.linears[0](query)
        keys = self.linears[1](key)
        values = self.linears[2](value)
        queries = queries.reshape([nbatches, self.h, -1, 128])
        keys = keys.reshape([nbatches, self.h, -1, 128])
        values = values.reshape([nbatches, self.h, -1, 128])
        # 2) Apply attention on all the projected vectors in batch.
        x = attention(queries, keys, values,self.device, mask=mask)
        # 3) "Concat" using a view and apply a final linear.
        x = x.contiguous().view(nbatches, -1, self.h * 128)
        return self.W_o(x)
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, d_model1, d_model2):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc = nn.Sequential(
            nn.Linear(6670, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
              # 512，2
        )
        self.fc1=nn.Linear(64, 2)

        self.w = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w, a=0, b=1)
        self.bn2 = nn.BatchNorm1d(116, affine=True)
    def encode(self, src,  src_mask):
        return self.encoder(self.src_embed(src),  src_mask)

    def w_upper_triangle_values(self,w):
        # 获取上三角部分的索引
        row_indices, col_indices = torch.triu_indices(w.size(0), w.size(1), offset=1)
        # 提取上三角部分的值
        upper_triangle_values = w[row_indices, col_indices]
        # 将上三角部分的值转换为一个向量
        vector = upper_triangle_values.view(-1)
        # 将向量转换为大小为 [1, 6670] 的矩阵
        result_matrix = vector.view(1, -1)
        return result_matrix

    def forward(self, src,src_mask):
        "Take in and process masked src and target sequences."

        w = 10 * (self.w)
        w = F.relu(self.bn2(w))
        w = (w + w.T) / 2
        l1 = torch.norm(w, p=1, dim=1).mean()
        result_matrix = self.w_upper_triangle_values(w)

        src = result_matrix * src

        src = self.encode(src, src_mask)
        src = (torch.sum(src, dim=1) / src.size(1)).squeeze()  # torch.Size([16, 256])
        src = self.fc(src)  # 32, 2
        src = self.fc1(src)

        return src,0.3*l1


