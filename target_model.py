
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
            x = layer(x,mask,W_A,W_B,W_C,W_D,W_E,W_F)
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
        x = self.sublayer[0](x,lambda x: self.self_attn1(x, x, x,mask,W_A,W_B,W_C,W_D,W_E,W_F))
        return self.sublayer[1](x,  self.feed_forward)
def attention(query, key, value,device,mask=None):
    "Compute 'Scaled Dot Product Attention'"
    padding_num = torch.ones_like(mask)
    padding_num = -2 ** 31 * padding_num.float()  # -inf
    scores = torch.matmul(query, key.transpose(-2, -1))
    # print(scores.shape)
    # print(mask.shape)
    # print(padding_num.shape)
    scores = torch.where(mask.to(device),scores.to(device), padding_num.to(device))
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
        self.device = device
    def forward(self, query, key, value,mask, W_A,W_B,W_C,W_D,W_E,W_F):
        if mask is not None:
            #Same mask applied to all h heads.
            mask = mask.unsqueeze(1).expand(-1,self.h, -1,-1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [l(x).view(nbatches, -1, self.h, 128 * self.h).transpose(1, 2) for l, x in
        #                      zip(self.linears, (query, key, value))]

        # W_A:torch.Size([16, 3, 256])
        # W_B:torch.Size([16, 6670, 3])
        #(W_A @ W_B):torch.Size([16, 6670, 256])
        #query:torch.Size([16, 25, 6670])
        #query*(W_A @ W_B)-->query @ (W_A @ W_B)
        # torch.Size([16, 6670, 256])*torch.Size([16, 25, 6670])-->  torch.Size([16, 25, 256])
        queries = self.linears[0](query) + query @ (W_A @ W_B)#torch.Size([16, 6670, 256])*torch.Size([16, 25, 6670])-->  torch.Size([16, 25, 256])
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
class Prompt(nn.Module):

    def __init__(self,
                  model_dim=6670,temperature=8):
        super().__init__()
        self.model_dim = model_dim
        self.attn_W_down = nn.Linear(self.model_dim, 100, bias=False)
        self.attn_W_up = nn.Linear(100, self.model_dim, bias=False)
        self.attn_non_linear = nn.SiLU()
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.temperature = temperature

        self.attn_W_down1 = nn.Linear(6670, 100, bias=False)
        self.attn_W_up1 = nn.Linear(100, 256, bias=False)
        self.attn_non_linear1 = nn.SiLU()
        self.layer_norm1 = nn.LayerNorm(256)

    def forward(self,inputs_embeds,W_A,G_A ,W_C, G_C,W_E,G_E):
        # top k源站点
        top_k = len(G_A) + 1
        if W_A.shape[0]==6670:
            q_mul_prefix_emb_added = torch.cat((G_A + [W_A]), dim=1)  # torch.Size([6670, 12])
            k_mul_prefix_emb_added = torch.cat((G_C + [W_C]), dim=1)  # torch.Size([6670, 12])
            v_mul_prefix_emb_added = torch.cat((G_E + [W_E]), dim=1)  # torch.Size([6670, 12])
            mul_prefix_emb_added= torch.stack((q_mul_prefix_emb_added,k_mul_prefix_emb_added,v_mul_prefix_emb_added))#torch.Size([3, 6670, 12])

            x = self.attn_W_down(inputs_embeds)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)  # torch.Size([32, 25, 6670])
            x, _ = torch.max(x, 1)  # torch.Size([32, 6670])

            attn_scores = (x @ mul_prefix_emb_added) / self.temperature#torch.Size([3, 32, 12])
            # 调整形状,进行求和
            summed_matrix = attn_scores.view(3,-1, top_k, 3).sum(dim=3)#torch.Size([3, 16, 4])
            # softmax操作
            softmax_matrix = F.softmax(summed_matrix, dim=2)  # torch.Size([16,4])
            w=[]
            for i in range(3):
               if i == 0:
                    W_sum = torch.zeros_like(G_A[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                    # 使用循环处理 G_A 中的每个权重
                    for j, weight in enumerate(G_A):
                           W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                           W_sum += W  # 将结果累加到 W_sum
                    W4 = W_A
                    W = W_sum + W4  # torch.Size([16, 6670, 3])
               if i == 1:
                   W_sum = torch.zeros_like(G_A[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                   # 使用循环处理 G_C 中的每个权重
                   for j, weight in enumerate(G_C):
                       W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                       W_sum += W  # 将结果累加到 W_sum
                   W4 = W_C
                   W = W_sum + W4  # torch.Size([16, 6670, 3])
               if i == 2:
                   W_sum = torch.zeros_like(G_E[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                   # 使用循环处理 G_A 中的每个权重
                   for j, weight in enumerate(G_E):
                       W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                       W_sum += W  # 将结果累加到 W_sum
                   W4 = W_E
                   W = W_sum + W4  # torch.Size([16, 6670, 3])
               w.append(W)
        else:
            q_mul_prefix_emb_added = torch.cat((G_A + [W_A]), dim=0).transpose(0, 1)  # torch.Size([256,12])
            k_mul_prefix_emb_added = torch.cat((G_C + [W_C]), dim=0).transpose(0, 1)  # torch.Size([256,12])
            v_mul_prefix_emb_added = torch.cat((G_E + [W_E]), dim=0).transpose(0, 1)  # torch.Size([256,12])
            mul_prefix_emb_added = torch.stack(
                (q_mul_prefix_emb_added, k_mul_prefix_emb_added, v_mul_prefix_emb_added))  # torch.Size([3, 256, 12])


            x = self.attn_W_down1(inputs_embeds)
            x = self.attn_non_linear1(x)
            x = self.attn_W_up1(x)
            x = self.layer_norm1(x)  # torch.Size([32, 25, 256])
            x, _ = torch.max(x, 1)  # torch.Size([32, 256])

            attn_scores = (x @ mul_prefix_emb_added) / self.temperature
            # 调整形状,进行求和
            summed_matrix = attn_scores.view(3,-1, top_k, 3).sum(dim=3)#torch.Size([3, 32, 4])
            # softmax操作
            softmax_matrix = F.softmax(summed_matrix, dim=2)  # torch.Size([32, 4])

            w=[]
            for i in range(3):
                if i == 0:
                    W_sum = torch.zeros_like(G_A[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                    # 使用循环处理 G_A 中的每个权重
                    for j, weight in enumerate(G_A):
                        W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                        W_sum += W  # 将结果累加到 W_sum
                    W4 = W_A
                    W = W_sum + W4  # torch.Size([16, 6670, 3])
                if i == 1:
                    W_sum = torch.zeros_like(G_A[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                    # 使用循环处理 G_C 中的每个权重
                    for j, weight in enumerate(G_C):
                        W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                        W_sum += W  # 将结果累加到 W_sum
                    W4 = W_C
                    W = W_sum + W4  # torch.Size([16, 6670, 3])
                if i == 2:
                    W_sum = torch.zeros_like(G_E[0]).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1).clone()
                    # 使用循环处理 G_A 中的每个权重
                    for j, weight in enumerate(G_E):
                        W = softmax_matrix[i, :, j:j + 1].unsqueeze(-1) * weight  # 使用 softmax 和权重进行相乘
                        W_sum += W  # 将结果累加到 W_sum
                    W4 = W_E
                    W = W_sum + W4  # torch.Size([16, 6670, 3])
                w.append(W)
        return w
class Prompt2(nn.Module):

    def __init__(self,
                  model_dim=6670,temperature=8):
        super().__init__()
        self.model_dim = model_dim
        self.attn_W_down = nn.Linear(self.model_dim, 100, bias=False)
        self.attn_W_up = nn.Linear(100, self.model_dim, bias=False)
        self.attn_non_linear = nn.SiLU()
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.temperature = temperature


    def forward(self,inputs_embeds):
            x = self.attn_W_down(inputs_embeds)
            x = self.attn_non_linear(x)
            x = self.attn_W_up(x)
            x = self.layer_norm(x)  #torch.Size([32, 25, 6670])

            return (x+inputs_embeds)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, src_embed, d_model1, d_model2,top_k):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc = nn.Sequential(
            nn.Linear(6670, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc1=nn.Linear(64, 2)

        self.top_k = top_k

        for i in range(1, top_k + 1):
            setattr(self, f'W_A_{i}', nn.Parameter(torch.empty(6670, 3)))
            setattr(self, f'W_B_{i}', nn.Parameter(torch.empty(3, 128 * 2)))
            setattr(self, f'W_C_{i}', nn.Parameter(torch.empty(6670, 3)))
            setattr(self, f'W_D_{i}', nn.Parameter(torch.empty(3, 128 * 2)))
            setattr(self, f'W_E_{i}', nn.Parameter(torch.empty(6670, 3)))
            setattr(self, f'W_F_{i}', nn.Parameter(torch.empty(3, 128 * 2)))

        self.W_A1 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_B1 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_A1, a=math.sqrt(5))
        nn.init.zeros_(self.W_B1)

        self.W_C1 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_D1 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_C1, a=math.sqrt(5))
        nn.init.zeros_(self.W_D1)

        self.W_E1 = nn.Parameter(torch.empty(6670, 3))  # LoRA权重A
        self.W_F1 = nn.Parameter(torch.empty(3, 128 * 2))  # LoRA权重B初始化LoRA权重
        nn.init.kaiming_uniform_(self.W_E1, a=math.sqrt(5))
        nn.init.zeros_(self.W_F1)

        self.w = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w, a=0, b=1)

        self.bn2 = nn.BatchNorm1d(116, affine=True)

        self.w1 = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w1, a=0, b=1)

    def encode(self, src, src_mask, W_A, W_B, W_C, W_D, W_E, W_F):
        return self.encoder(self.src_embed(src), src_mask, W_A, W_B, W_C, W_D, W_E, W_F)

    def w_upper_triangle_values(self, w):
        # 获取上三角部分的索引
        row_indices, col_indices = torch.triu_indices(w.size(0), w.size(1), offset=1)
        # 提取上三角部分的值
        upper_triangle_values = w[row_indices, col_indices]
        # print(row_indices, col_indices)
        # 将上三角部分的值转换为一个向量
        vector = upper_triangle_values.view(-1)
        # 将向量转换为大小为 [1, 6670] 的矩阵
        result_matrix = vector.view(1, -1)
        # print(result_matrix)
        return result_matrix

    def forward(self, src,src_mask,prompt1):

        w=10*(0.4*self.w+0.6*self.w1)
        w = F.relu(self.bn2(w))
        w = (w + w.T) / 2
        #print(w)
        l1 = torch.norm(w, p=1, dim=1).mean()
        result_matrix=self.w_upper_triangle_values(w)

        src = result_matrix * src

        # 在 forward 中创建包含所有 W_A_1, W_A_2, ... 的列表
        G_A = [getattr(self, f'W_A_{i}') for i in range(1, self.top_k + 1)]
        G_B = [getattr(self, f'W_B_{i}') for i in range(1, self.top_k + 1)]
        G_C = [getattr(self, f'W_C_{i}') for i in range(1, self.top_k + 1)]
        G_D = [getattr(self, f'W_D_{i}') for i in range(1, self.top_k + 1)]
        G_E = [getattr(self, f'W_E_{i}') for i in range(1, self.top_k + 1)]
        G_F = [getattr(self, f'W_F_{i}') for i in range(1, self.top_k + 1)]
        # q,k,v
        w1 = prompt1(src, self.W_A1, G_A, self.W_C1, G_C, self.W_E1, G_E)
        w2 = prompt1(src, self.W_B1, G_B, self.W_D1, G_D, self.W_F1, G_F)

        src = self.encode(src, src_mask, w1[0], w2[0], w1[1], w2[1], w1[2], w2[2])
        src = (torch.sum(src, dim=1) / src.size(1)).squeeze()  # torch.Size([16,6670])
        src = self.fc(src)
        src = self.fc1(src)
        return src,0.3*l1

