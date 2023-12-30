import torch
import numpy as np
from math import ceil


class Choquet_Integral(torch.nn.Module):
    def __init__(self, heads: int, neighbor_node_nums: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.neighbors_node_nums = neighbor_node_nums
        self.var_num = 2 ** self.neighbors_node_nums - 1
        self.FM = torch.ones(size=(self.var_num, heads), requires_grad=True) / self.neighbors_node_nums
        # self.norm = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=heads),
        #                                 torch.nn.PReLU(),
        #                                 torch.nn.Dropout(dropout))
        self.act = torch.nn.PReLU()
        self.mobius_index = []
        self.source_index, self.source_index_mobius, self.sub_source_for_draw = self.get_source_index(
            self.neighbors_node_nums)
        self.M = torch.nn.Parameter(self.generate_mobius_transformer(), requires_grad=False)
        self.S = torch.nn.Parameter(self.generate_shapley_transformer(), requires_grad=False)
        self.mobius_index = torch.tensor(self.mobius_index)
        self.standardize()
        FM_noise = torch.randn(size=(self.var_num, heads)) + 5e-2
        self.FM = torch.nn.Parameter(self.FM + FM_noise)
        # self.draw_hasse_diagram()

    def get_source_index(self, N):
        source_index = []
        sub_source_for_draw = []
        source_index_mobius = []
        for bcode in range(1, 2 ** N):
            sourceNum = [j for j, i in enumerate(bin(bcode)[2:][::-1]) if i == "1"]
            source_index.append([(bcode - 1) - 2 ** i for i in sourceNum])
            sub_source_for_draw.append([i + 1 for i in sourceNum])
            source_index_mobius.append([0] * N)
            for k in sub_source_for_draw[-1]:
                source_index_mobius[-1][k - 1] = 1
            self.mobius_index.append(sourceNum + [sourceNum[0]] * (N - len(sourceNum)))
        sub_source_for_draw.insert(0, [0])
        source_index_mobius = torch.tensor(source_index_mobius)
        return source_index, source_index_mobius, sub_source_for_draw

    def generate_mobius_transformer(self):

        def count_ones(n):

            count = 0
            while n:
                n &= (n - 1)  # Clears the rightmost 1 bit
                count += 1
            return count

        M = torch.zeros(size=(2 ** self.neighbors_node_nums - 1, 2 ** self.neighbors_node_nums - 1),
                        requires_grad=False)
        for i in range(1, 2 ** self.neighbors_node_nums):
            for j in range(1, i + 1):
                if i & j == j:  # j ⊆ i
                    M[i - 1][j - 1] = pow(-1, (count_ones(i) - count_ones(j)))
        return M

    def generate_shapley_transformer(self):
        from scipy.special import factorial
        def count_ones(n):

            count = 0
            while n:
                n &= (n - 1)  # Clears the rightmost 1 bit
                count += 1
            return count

        S = torch.zeros(size=(self.neighbors_node_nums, 2 ** self.neighbors_node_nums - 1),
                        requires_grad=False)

        const = factorial(self.neighbors_node_nums)
        for idx, i in enumerate(2 ** n for n in range(self.neighbors_node_nums)):
            for j in range(0, 2 ** self.neighbors_node_nums):
                if j == 0:
                    coff = (factorial(self.neighbors_node_nums - 1)) / const
                    S[idx][i - 1] = coff
                elif i | j != j:  # i !⊆ j
                    size = count_ones(j)
                    coff = (factorial(self.neighbors_node_nums - size - 1) * factorial(size)) / const
                    S[idx][(i | j) - 1] = coff
                    S[idx][j - 1] = -coff
        return S

    def generate_shapley_value(self):
        return torch.einsum("np,ph->nh", self.S, self.FM)

    @torch.no_grad()
    def standardize(self):
        for idx, subsetIdx in enumerate(self.source_index):
            if len(subsetIdx) > 1:
                maxVal, _ = torch.max(self.FM[subsetIdx, :], dim=0)
                torch.where(self.FM[idx, :] <= maxVal,
                            torch.add(self.FM[idx, :], maxVal),
                            self.FM[idx, :],
                            out=self.FM[idx, :])

    def forward(self, x):
        N, S, D = x.shape
        FM = self.act(self.FM)
        x_sort, sortIdx = torch.sort(x, dim=1, descending=True)
        x_sort = torch.cat((x_sort, torch.zeros(N, 1, D, device=x.device)), -2)
        x_sort = (x_sort[:, :-1, :] - x_sort[:, 1:, :])
        sortIdx = torch.cumsum(torch.pow(2, sortIdx), dim=1) - 1
        FM = FM[sortIdx, :]
        x = (FM * x_sort.unsqueeze(-1)).sum(1).squeeze(-1)
        return x

    def forward_mobius(self, x):
        # FM = torch.clamp(input=self.FM, min=0)
        FM = self.act(self.FM)
        mobius_FM = torch.matmul(self.M, FM)
        xf = torch.min(x[:, self.mobius_index, :], dim=-2)[0]
        xf = (xf * mobius_FM).sum(1)
        # xf = torch.einsum("bsd,sh->bhd", xf, mobius_FM).view(B, self.heads)
        return xf

    def shapley_value(self):
        from scipy.special import comb
        from einops import repeat
        import matplotlib.pyplot as plt
        import seaborn as sns
        self.standardize()

        def f(x):
            count = 0
            while x:
                x = x & (x - 1)
                count += 1
            return count

        total_source = self.FM.detach().cpu().unsqueeze(0).expand(self.neighbors_node_nums,
                                                                  2 ** self.neighbors_node_nums - 1, self.heads)
        total_source = torch.cat([torch.zeros(size=(self.neighbors_node_nums, 1, self.heads)), total_source], dim=1)
        source_index = repeat(torch.tensor([2 ** i for i in range(self.neighbors_node_nums)]), "S -> S S2 H",
                              S2=2 ** self.neighbors_node_nums,
                              H=self.heads)
        all_source_index = repeat(torch.tensor([i for i in range(2 ** self.neighbors_node_nums)]), "S2 -> S S2 H",
                                  S=self.neighbors_node_nums,
                                  H=self.heads)
        inv_combine_num = repeat(torch.tensor(
            [1 / comb(self.neighbors_node_nums - 1, f(i) - 1) for i in range(2 ** self.neighbors_node_nums)]),
            "S2 -> S S2 H", S=self.neighbors_node_nums,
            H=self.heads)
        inv_combine_num = torch.where(inv_combine_num == torch.inf, 1, inv_combine_num)
        mask = (all_source_index & source_index) == source_index
        div_mask = all_source_index & (all_source_index ^ source_index)
        margin_source = total_source - torch.gather(total_source, dim=1, index=div_mask)
        margin_source = (
                torch.sum(torch.where(mask, margin_source, 0) * inv_combine_num, dim=1) * (
                1 / self.neighbors_node_nums)).mean(
            -1)

        sns.barplot(x=self.meta_path[1:], y=margin_source.numpy())
        plt.title("Shapely Value")
        plt.show()

        return margin_source

    def reset(self):
        torch.nn.init.constant_(self.FM.data, 1 / self.neighbors_node_nums)
        self.standardize()


if __name__ == '__main__':

    device = "cpu"
    Neighbor_node_nums = 3
    N_hidden = 1
    m = Choquet_Integral(neighbor_node_nums=Neighbor_node_nums, dropout=0.3, heads=1).to(device)
    m.FM = torch.nn.Parameter(torch.tensor([[0.45,0.45,0.6,0.3,0.9,0.9,1]]).T)
    for i in range(500):
        x = torch.tensor([[[18],[16],[10]], [[10],[12],[18]]]).float()
        x = x.to(device)
        y1 = m.forward_mobius(x)
        y2 = m(x)
        p = 0
