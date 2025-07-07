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
    def forward(self, x,  mask,W_A,W_B,W_C,W_D,W_E,W_F):
        "Pass the input (and mask) through each layer in turn."
        #loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x,  mask,W_A,W_B,W_C,W_D,W_E,W_F)
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
    def forward(self, x,mask,W_A,W_B,W_C,W_D,W_E,W_F):
        x = self.sublayer[0](x,lambda x: self.self_attn1(x, x, x,  mask,W_A,W_B,W_C,W_D,W_E,W_F))
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
    def forward(self, query, key, value,mask, W_A,W_B,W_C,W_D,W_E,W_F):
        if mask is not None:
            #Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(-1,self.h, -1,-1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(nbatches, -1, self.h, 128 * self.h).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]
        queries = self.linears[0](query) + query @ (W_A @ W_B)
        keys = self.linears[1](key) + key @ (W_C @ W_D)
        values = self.linears[2](value) + value @ (W_E @ W_F)
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
        )
        self.fc1=nn.Linear(64, 2)

        self.W_A = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_B = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        nn.init.zeros_(self.W_B)

        self.W_C = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_D = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        nn.init.zeros_(self.W_B)

        self.W_E = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_F = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        nn.init.zeros_(self.W_B)


    def encode(self, src,  src_mask,W_A,W_B,W_C,W_D,W_E,W_F):
        return self.encoder(self.src_embed(src),  src_mask,W_A,W_B,W_C,W_D,W_E,W_F)


    def forward(self, src,src_mask):
        # src = prompt.add(src)
        #"Take in and process masked src and target sequences."
        src = self.encode(src, src_mask,self.W_A,self.W_B,self.W_C,self.W_D,self.W_E,self.W_F)

        src = (torch.sum(src, dim=1) / src.size(1)).squeeze()  # torch.Size([16,6670])
        src0 = self.fc(src)
        src = self.fc1(src0)

        return src

