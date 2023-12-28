import torch
import torchviz
from torch.nn.functional import softmax


class MMTs(torch.nn.Module):

    def __init__(self, hidden_nums: int, multi_nums: int, heads: int, dropout: float):
        super().__init__()
        self.n_heads = heads
        self.multi_nums = multi_nums
        self.hidden_nums = hidden_nums
        self.Wq = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.Wk = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.Wv = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v):
        residual = q
        B, M, S, D = q.shape
        q = torch.matmul(q, self.Wq).view(B, M, S, self.n_heads, self.hidden_nums)
        k = torch.matmul(k, self.Wk).view(B, M, S, self.n_heads, self.hidden_nums)
        v = torch.matmul(v, self.Wv).view(B, M, S, self.n_heads, self.hidden_nums)
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)  # B M H S D
        attn = torch.matmul(q, k.transpose(-1, -2)) / self.hidden_nums ** 0.5
        attn = self.dropout(torch.softmax(attn, dim=-1))
        # attn = torch.where(torch.isnan(attn), 1, attn)
        out = torch.matmul(attn, v).nanmean(2) + residual
        return out


if __name__ == '__main__':
    m = MMTs(hidden_nums=128, multi_nums=9, dropout=0.1, heads=5)
    x = torch.randn(size=(2, 9, 3, 128))
    loss12 = m(x, x, x)
    torchviz.make_dot(loss12, params=dict(list(m.named_parameters()))).render()

