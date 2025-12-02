import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, x):
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)  # normalizza lungo seq_len
        out = torch.matmul(attn, V)  # (batch, seq_len, embed_dim)
        
        return out
