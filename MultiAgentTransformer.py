import torch
import torchviz
from torch.nn.functional import softmax


class MMTAs(torch.nn.Module):

    def __init__(self, hidden_nums: int, multi_nums: int, heads: int, dropout: float):
        super().__init__()
        self.n_heads = heads
        self.multi_nums = multi_nums
        self.hidden_nums = hidden_nums
        self.Wq = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.Wk = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.Wv = torch.nn.Parameter(torch.randn(size=(multi_nums, hidden_nums, heads * hidden_nums)))
        self.layer_norm_q = torch.nn.LayerNorm(normalized_shape=[3, 1])
        self.layer_norm_k = torch.nn.LayerNorm(normalized_shape=[1, 3])
        self.WAgent_bias1 = torch.nn.Parameter(torch.randn(size=(multi_nums, 1, 3)))
        self.WAgent_bias2 = torch.nn.Parameter(torch.randn(size=(multi_nums, 3, 1)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v):
        residual = q
        B, M, S, D = q.shape

        q = torch.matmul(q, self.Wq)
        k = torch.matmul(k, self.Wk)
        v = torch.matmul(v, self.Wv)
        A = torch.mean(q, dim=-2, keepdim=True)
        q_agent = self.layer_norm_q(torch.matmul(q, A.transpose(-1, -2)))
        q_agent = torch.softmax(q_agent + self.WAgent_bias2, dim=2)
        k_agent = self.layer_norm_k(torch.matmul(A, k.transpose(-1, -2))) + self.WAgent_bias1
        k_agent = torch.softmax(k_agent, dim=3)
        qkv = q_agent @ k_agent @ v
        qkv = qkv.view(B, M, S, self.n_heads, self.hidden_nums).mean(-2) + residual
        return qkv


if __name__ == '__main__':
    m = MMTs(hidden_nums=128, multi_nums=9, dropout=0.1, heads=5)
    x = torch.randn(size=(2, 9, 3, 128))
    loss12 = m(x, x, x)
    torchviz.make_dot(loss12, params=dict(list(m.named_parameters()))).render()

