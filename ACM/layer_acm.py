import math
import time
import numpy as np
import torch
import einops
from torch_geometric.utils import scatter, add_self_loops
from math import ceil

device = "cuda"


def zero_one(x, interval=None):
    if interval is None:
        interval = [0, 1]
    start, end = interval
    length = end - start
    maxm, _ = torch.max(x, dim=0, keepdim=True)
    minm, _ = torch.min(x, dim=0, keepdim=True)
    res = ((x - minm) / (maxm - minm)) * length + start
    res[res == torch.inf] = end
    return res


class FeatureProjector(torch.nn.Module):
    def __init__(self, c_in, n_embed, n_out, dropout, bias=True):
        super().__init__()
        self.feature_project_weight = torch.nn.Parameter(torch.randn((c_in, n_embed, n_out)))
        self.feature_project_bias = torch.nn.Parameter(torch.zeros((c_in, n_out))) if bias else 0
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm([c_in, n_out]),
                                        torch.nn.PReLU(),
                                        torch.nn.Dropout(dropout))

    def forward(self, x):
        return self.norm(torch.einsum("bcm,cmn->bcn", x, self.feature_project_weight) + self.feature_project_bias)


class FPs(torch.nn.Module):
    def __init__(self, c_in, n_embed, n_out, dropout, layers=2, bias=True):
        super().__init__()
        self.nn = torch.nn.ModuleList(
            [FeatureProjector(c_in=c_in, n_embed=n_embed, n_out=n_out, bias=bias, dropout=dropout)] +
            [FeatureProjector(c_in=n_out, n_embed=n_embed, n_out=n_out, bias=bias, dropout=dropout) for _ in
             range(layers - 1)]
        )

    def forward(self, x):
        for m in self.nn:
            x = m(x)
        return x


class Choquet_Integral(torch.nn.Module):
    def __init__(self, source_num: int, heads: int, concat: bool, hidden: int, dropout: float, meta_path:list):
        super().__init__()
        self.src_num = source_num
        self.heads = heads
        self.concat = concat
        self.var_num = 2 ** self.src_num - 1
        self.FM = torch.ones(size=(self.var_num, heads), requires_grad=True) / self.src_num
        # self.FM = torch.nn.Parameter(
        #     torch.ones(size=(self.var_num, heads), requires_grad=True) / self.src_num)

        # self.FM = torch.nn.Parameter(torch.ones(size=(self.var_num, heads), requires_grad=True) / self.src_num + torch.randn(size=(self.var_num, heads)) * 1e-1)
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm([heads if concat else 1, hidden]),
                                        torch.nn.PReLU(),
                                        torch.nn.Dropout(dropout))
        self.act = torch.nn.PReLU()
        self.meta_path = np.array(["∅"] + meta_path)
        self.mobius_index = []
        self.source_index, self.source_index_mobius, self.sub_source_for_draw = self.get_source_index(self.src_num)
        self.M = torch.nn.Parameter(self.generate_mobius_transformer(), requires_grad=False)
        self.mobius_index = torch.tensor(self.mobius_index)
        self.standardize()
        FM_noise = torch.randn(size=(self.var_num, heads)) * 1e-1
        with torch.no_grad():
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
        from math import log2
        def f(a, b):
            tmp_a = 0
            while a:
                tmp_a += a & 1
                a >>= 1
            tmp_b = 0
            while b:
                tmp_b += b & 1
                b >>= 1
            return -1 ** (a - b)

        M = torch.zeros(size=(2 ** self.src_num - 1, 2 ** self.src_num - 1), requires_grad=False)
        for i in range(1, 2 ** self.src_num):
            for j in range(1, i + 1):
                if i & j == j:  # j ⊆ i
                    M[i - 1][j - 1] = f(i, j)
        return M

    @torch.no_grad()
    def standardize(self):
        self.FM = torch.nn.Parameter(torch.where(self.FM < 0, 0, self.FM))
        for idx, subsetIdx in enumerate(self.source_index):
            if len(subsetIdx) > 1:
                maxVal, _ = torch.max(self.FM[subsetIdx, :], dim=0)
                torch.where(self.FM[idx, :] <= maxVal,
                            torch.add(self.FM[idx, :], maxVal),
                            self.FM[idx, :],
                            out=self.FM[idx, :])
        self.FM = torch.nn.Parameter(torch.where(self.FM > 1, 1, self.FM))

    def forward(self, x):
        N, S, D = x.shape
        x_sort, sortIdx = torch.sort(x, dim=1, descending=True)
        x_sort = torch.cat((x_sort, torch.zeros(N, 1, D, device=device)), -2)
        x_sort = (x_sort[:, :-1, :] - x_sort[:, 1:, :])
        sortIdx = torch.cumsum(torch.pow(2, sortIdx), dim=1) - 1
        FM_ = self.FM[sortIdx.view(-1), :].view(N, S, D, self.heads)
        x = torch.einsum("bsn,bsnh->bhn", x_sort, FM_)
        if not self.concat:
            x = torch.sum(x, dim=-2, keepdim=True)
        return self.norm(x)

    def forward_mobius(self, x):
        # FM = torch.clamp(input=self.FM, min=0)
        FM = self.act(self.FM)
        mobius_FM = torch.einsum("sh,bs->sh", self.FM, self.M)
        xf, _ = torch.min(x[:, self.mobius_index, :], dim=-2)
        xf = torch.einsum("bsd,sh->bhd", xf, mobius_FM)
        if not self.concat:
            xf = xf.sum(1, keepdim=True)
        return self.norm(xf)

    def draw_hasse_diagram(self):
        import matplotlib.pyplot as plt
        from scipy.special import comb
        from copy import deepcopy

        # self.standardize()
        height = self.src_num + 1
        vertical = 100 / (self.src_num + 2)
        horizontal = 100
        size_scaler = lambda x: x * 50 + 1

        vertical_margin = [(i + 1) * vertical for i in range(self.src_num + 2)]
        horizontal_margin = [horizontal / (comb(self.src_num, i) + 1) for i in range(self.src_num + 1)]

        col_num = 2
        row_num = int(ceil(self.heads / col_num))
        fig, ax = plt.subplots(row_num, col_num)
        if row_num == 1:
            ax = [ax]
        c = 0
        r = 0
        for head_idx in range(self.heads):
            FM = np.concatenate([[0], self.FM[:, head_idx].detach().cpu().numpy()], axis=0)
            coordinates = []
            offset = [1 for _ in range(self.src_num + 1)]
            for idx, fm in enumerate(FM):
                idx_ = idx
                h = 0
                while idx_:
                    h += idx_ & 1
                    idx_ >>= 1
                c_coordinate = offset[h] * horizontal_margin[h]
                r_coordinate = vertical_margin[h]
                coordinates.append([c_coordinate, r_coordinate])
                ax[r][c].scatter([c_coordinate], [r_coordinate], c="blue", s=size_scaler(abs(fm)))
                offset[h] += 1
            coordinates = np.array(coordinates)

            si = deepcopy(self.source_index)
            si.insert(0, [(1 << i) - 1 for i in range(self.src_num)])
            for i in range(2 ** self.src_num):
                target_co = coordinates[i]
                sub_co = si[i]
                ax[r][c].text(target_co[0] - 1, target_co[1] + 1, f"{self.meta_path[self.sub_source_for_draw[i]]}: {round(FM[i], 3)}", fontsize=6)
                if len(sub_co) == 1:
                    continue
                else:
                    sub_co = [i + 1 for i in sub_co]
                    sub_co = coordinates[sub_co]
                for j in range(sub_co.shape[0]):
                    ax[r][c].plot([target_co[0], sub_co[j][0]], [target_co[1], sub_co[j][1]], c="b", alpha=0.1)
            c += 1
            if c == col_num:
                c = 0
                r += 1
        plt.show()

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

        total_source = self.FM.detach().cpu().unsqueeze(0).expand(self.src_num, 2 ** self.src_num - 1, self.heads)
        total_source = torch.cat([torch.zeros(size=(self.src_num, 1, self.heads)), total_source], dim=1)
        source_index = repeat(torch.tensor([2 ** i for i in range(self.src_num)]), "S -> S S2 H", S2=2 ** self.src_num,
                              H=self.heads)
        all_source_index = repeat(torch.tensor([i for i in range(2 ** self.src_num)]), "S2 -> S S2 H", S=self.src_num,
                                  H=self.heads)
        inv_combine_num = repeat(torch.tensor([1 / comb(self.src_num - 1, f(i)-1) for i in range(2 ** self.src_num)]),
                                 "S2 -> S S2 H", S=self.src_num,
                                 H=self.heads)
        inv_combine_num = torch.where(inv_combine_num == torch.inf, 1, inv_combine_num)
        mask = (all_source_index & source_index) == source_index
        div_mask = all_source_index & (all_source_index ^ source_index)
        margin_source = total_source - torch.gather(total_source, dim=1, index=div_mask)
        margin_source = (torch.sum(torch.where(mask, margin_source, 0) * inv_combine_num, dim=1) * (1 / self.src_num)).mean(-1)

        sns.barplot(x=self.meta_path[1:], y=margin_source.numpy())
        plt.title("Shapely Value")
        plt.show()

        return margin_source

    def reset(self):
        torch.nn.init.constant_(self.FM.data, 1 / self.src_num)
        self.standardize()


class CIE(torch.nn.Module):
    def __init__(self, source_num: int, heads: int, standard_train: bool, concat: bool):
        super().__init__()
        self.src_num = source_num
        self.heads = heads
        self.concat = concat
        self.var_num = self.src_num - 1
        self.FM = torch.nn.Parameter(torch.cat([torch.zeros(size=(1, heads), requires_grad=False),
                                                torch.randn(size=(self.var_num, heads),
                                                            requires_grad=True)], dim=0))
        self.Agg = torch.nn.Parameter(torch.randn(size=(1, N_source, heads)))
        self.source_index = torch.nn.Parameter(torch.tensor(self.get_source_index(source_num)), requires_grad=False)
        self.norm = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=source_num),
                                        torch.nn.PReLU(),
                                        torch.nn.Dropout(0))

    def get_source_index(self, N):
        sourceIndex = []
        for bcode in range(1, 2 ** N):
            sourceNum = [j + 1 if i == "1" else 0 for j, i in enumerate(bin(bcode)[2:][::-1].ljust(N, "0"))]
            sourceIndex.append(sourceNum)
        return sourceIndex

    def forward(self, x):
        N, S, D = x.shape
        # x = self.norm(x)
        x_sort, sortIdx = torch.sort(x, dim=1, descending=True)
        x_sort = torch.cat((x_sort, torch.zeros(N, 1, D, device=device)), -2)
        x_sort = (x_sort[:, :-1, :] - x_sort[:, 1:, :])
        sortIdx = torch.cumsum(torch.pow(2, sortIdx), dim=1) - 1  # N S D
        sortIdx = self.source_index[sortIdx.view(-1)]  # (2**S-1) S -> (B S D) S
        FM_ = self.FM[sortIdx]  # S H -> (B S D) S H
        FM_ = (FM_ * self.Agg).sum(1).view(N, S, D, self.heads)  # (B S D) S H, S H -> B S D H
        # FM_ = einops.reduce(FM_ * self.Agg, "(n s1 d) s2 h -> n s1 d h ", n=N, s1=S, s2=S, d=D, h=self.heads,
        #                     reduction="sum")
        # x = torch.einsum("nsdh,nsd->nhd", FM_, x_sort)
        x = (x_sort.unsqueeze(-1) * FM_).sum(1)
        if self.concat:
            return torch.transpose(x, -1, -2)
            # x = x.view(N, self.heads * D)
        else:
            x = torch.sum(x, dim=-1, keepdim=True)
        return x


class CII(torch.nn.Module):
    def __init__(self, source_num: int, out_hid: int, heads: int, concat: bool, dropout):
        super().__init__()
        self.src_num = source_num
        self.heads = heads
        self.out = out_hid
        self.Wj = torch.nn.Parameter(torch.randn(size=(heads, source_num, 2 * out_hid, out_hid), requires_grad=True))
        self.Wi = torch.nn.Parameter(torch.randn(size=(heads, source_num, out_hid, out_hid)), requires_grad=True)
        self.norm1 = torch.nn.Sequential(
            torch.nn.LayerNorm([source_num, out_hid]),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.norm2 = torch.nn.Sequential(
            torch.nn.LayerNorm([heads, out_hid]),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x):
        N, S, D = x.shape
        x_sort, sortIdx = torch.sort(x, dim=1, descending=True)
        x_sort = torch.cat((x_sort, torch.zeros(N, 1, D, device=device)), -2)
        x_sort = (x_sort[:, :-1, :] - x_sort[:, 1:, :])
        sortIdx = torch.cumsum(torch.pow(2, sortIdx), dim=1) / 2 ** self.src_num
        sortEmbed = self.norm1(torch.einsum("bsd,hsdo->bhso", sortIdx, self.Wi))
        x_sort = torch.cat([x_sort.unsqueeze(1).expand(N, self.heads, S, self.out), sortEmbed], dim=-1)
        # x_sort = x_sort.unsqueeze(1).expand(N, self.heads, S, self.out) + sortEmbed
        return self.norm2(torch.einsum("bhsd,hsdo->bho", x_sort, self.Wj))


class CII_(torch.nn.Module):
    def __init__(self, source_num: int, head: int, out_hid: int, hidden: int, standard_train: bool, concat: bool):
        super().__init__()
        self.src_num = source_num
        self.hidden = hidden
        self.out = out_hid
        self.head = head
        self.Wj = torch.nn.Parameter(torch.randn(size=(head, source_num, 2 * out_hid, out_hid), requires_grad=True))
        self.Wi = torch.nn.Parameter(torch.randn(size=(head, source_num, out_hid, out_hid)), requires_grad=True)
        self.norm1 = torch.nn.Sequential(
            torch.nn.LayerNorm([source_num, out_hid]),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.1))
        self.norm2 = torch.nn.Sequential(
            torch.nn.LayerNorm([head, out_hid]),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.1))

    def forward(self, x):
        N, S, D = x.shape
        x_sort, sortIdx = torch.sort(x, dim=1, descending=True)
        x_sort = torch.cat((x_sort, torch.zeros(N, 1, D, device=device)), -2)
        x_sort = (x_sort[:, :-1, :] - x_sort[:, 1:, :])
        sortIdx = (torch.cumsum(torch.pow(2, sortIdx), dim=1) / (2 ** self.src_num)) * 2 - 1
        sortEmbed = self.norm1(torch.einsum("bsd,hsdo->bhso", sortIdx, self.Wi))
        # x_sort = torch.cat([x_sort.unsqueeze(1).expand(N, S, self.out), sortEmbed], dim=-1)
        x_sort = torch.cat([x_sort.unsqueeze(1).expand(N, self.head, S, self.out) + sortEmbed,
                            x.unsqueeze(1).expand(N, self.head, S, self.out)], dim=-1)

        # (x_sort.unsqueeze(-1) * self.Wj.unsqueeze(0)).sum(-2).sum(2)
        return self.norm2(torch.einsum("bhsd,hsdo->bho", x_sort, self.Wj))


class GaussianMF(torch.nn.Module):

    def __init__(self,
                 semantic_num: int,
                 in_channel: int,
                 num_mf: int,
                 values_intervals: list,
                 fix: bool = False,
                 close: bool = True,
                 cross: float = 0.7):
        """
        Initiate Gaussian-Type membership function
        :param semantic_num: number of semantics
        :param in_channel: Input dimension
        :param num_mf: Number of membership function
        :param values_intervals: Domain of definition of membership functions
        :param fix: Whether to make the parameters of the membership function trainable
        :param close: Let the peak of the membership function fall on the boundary of the defined domain
        :param cross: The cross point of two membership function
        """

        super().__init__()
        self.num_mf = num_mf
        self.intervals = values_intervals
        self.in_channel = in_channel
        self.close = close
        self.cross = torch.tensor(cross)
        self.fix = fix

        start, end = values_intervals
        if close:
            intervals = (end - start) / (num_mf + 1)
            start += intervals
            end -= intervals
        C = torch.linspace(start=start, end=end, steps=num_mf, requires_grad=False)
        if num_mf > 1:
            sigma = torch.pow(C[1] - C[0], 2) * (1 / (-8 * torch.log(torch.tensor(cross))))
        else:
            sigma = 0.1225
        C = torch.tile(torch.linspace(start=start, end=end, steps=num_mf).view(1, 1, 1, -1),
                       (semantic_num, in_channel, 1))
        Sigma = torch.tile((sigma * torch.ones((num_mf,))).view(1, 1, 1, -1), (semantic_num, in_channel, 1))
        if not fix:
            self.register_parameter("C", torch.nn.Parameter(C))
            self.register_parameter("Sigma", torch.nn.Parameter(Sigma))
        else:
            self.register_buffer("C", C)
            self.register_buffer("Sigma", Sigma)

    def forward(self, x):
        return torch.exp(- torch.pow((x.unsqueeze(-1) - self.C), 2) / (2 * self.Sigma))

    @torch.no_grad()
    def crop(self):
        torch.clamp(self.C, self.intervals[0], self.intervals[1], out=self.C)


class FNNConv(torch.nn.Module):
    def __init__(self,
                 num_mf: int,
                 val_interval: [int, int],
                 cross: float,
                 n_hidden: int,
                 window_size: int,
                 stride_size: int,
                 dropout: float,
                 refine_ratio: float,
                 norm: bool,
                 fix: bool,
                 extract_dim: int
                 ):
        super().__init__()
        self.num_mf = num_mf
        self.val_interval = val_interval
        self.cross = cross
        self.fix = fix
        self.norm = norm
        self.refine_ratio = refine_ratio
        self.window = window_size
        self.stride = stride_size

        self.num_blocks = torch.empty(size=(1, n_hidden, num_mf)).unfold(1, window_size, stride_size).shape[1]
        self.rules_size = self.num_blocks * pow(num_mf, window_size)
        self.rules_size_refine = int(ceil(refine_ratio * self.rules_size))
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm([extract_dim, n_hidden]),
                                        torch.nn.PReLU(),
                                        torch.nn.Dropout(dropout))

        if 0 < self.refine_ratio < 1:
            self.refiner = torch.nn.AdaptiveMaxPool1d(output_size=self.rules_size_refine)
        elif self.refine_ratio > 1 or self.refine_ratio <= 0:
            raise "ration must between [0, 1]"
        else:
            self.refiner = None
        self.coe_defuzzifize = torch.nn.Parameter(
            torch.randn(size=(1, extract_dim, self.rules_size_refine, n_hidden)))

        self.fuzzier = GaussianMF(in_channel=n_hidden,
                                  semantic_num=extract_dim,
                                  num_mf=num_mf,
                                  fix=fix,
                                  cross=cross,
                                  values_intervals=val_interval)

    @staticmethod
    def full_expand(x):
        N, S, B, W, M = x.shape
        tmp = x[:, :, :, 0, :]
        for w in range(1, W):
            tmp = torch.mul(tmp.unsqueeze(-2), x[:, :, :, w, :].unsqueeze(-1))
            tmp = tmp.view(N, S, B, -1)
        return tmp

    def forward(self, xf):
        xf = zero_one(xf, interval=self.val_interval)
        xf = self.fuzzier(xf)
        xf = torch.transpose(xf.unfold(dimension=2, size=self.window, step=self.stride), -1, -2)
        xf = self.full_expand(xf)
        norm_fac = torch.sum(xf, dim=-1, keepdim=True) if self.norm else 1  # (N, R) -> (N, 1)
        xf = (xf / norm_fac).reshape(xf.size(0), xf.size(1), -1)
        if self.refiner is not None:
            xf = self.refiner(xf)
        return self.norm((xf.unsqueeze(-1) * self.coe_defuzzifize).sum(-2))


class DjFNNConv(torch.nn.Module):
    def __init__(self,
                 num_mf: int,
                 val_interval: [int, int],
                 cross: float,
                 n_hidden: int,
                 n_node: int,
                 dropout: float,
                 concat: bool,
                 norm: bool,
                 fix: bool,
                 out_chn: int,
                 extract_dim: int):
        super().__init__()
        self.fuzzier = GaussianMF(in_channel=n_hidden,
                                  semantic_num=extract_dim,
                                  num_mf=num_mf,
                                  fix=fix,
                                  cross=cross,
                                  values_intervals=val_interval)
        self.norm = norm
        self.concat = concat
        self.val_interval = val_interval
        self.coe_OR = torch.nn.Parameter(
            torch.randn(
                size=(1, extract_dim, n_hidden, num_mf, n_node)))  # (N, S, D, M),  (N, S, D, N_NODE) -> N, S, D, N_NODE
        self.coe_AND = torch.nn.Parameter(torch.randn(
            size=(1, extract_dim, n_hidden, n_node)))  # (N, S, D, N_NODE), (N, S, D, N_NODE) -> N, S, N_NODE
        self.coe_DF = torch.nn.Parameter(
            torch.randn(size=(1, extract_dim, n_node, out_chn)))  # (N, S, N_NODE), (N, S, N_NODE, O) -> N, S, O

    def forward(self, x, *args):
        x = zero_one(x, self.val_interval)
        xf = self.fuzzier(x).unsqueeze(-1)
        # OR
        xf = torch.mul(xf, self.coe_OR).mean(-2)
        # AND
        xf = torch.mean((xf + self.coe_AND) - torch.mul(xf, self.coe_AND), dim=-2).unsqueeze(-1)
        if self.norm:
            norm = torch.sum(xf, dim=-1)
        else:
            norm = 1

        # defuzzification
        # xf = torch.einsum("bsn,sno->bso", xf, self.coe_DF) / norm
        xf = (xf * self.coe_DF).sum(2) / norm

        return torch.cat([x, xf], dim=-1) if self.concat else x + xf


if __name__ == '__main__':
    N_source = 3
    N_hidden = 128
    # x = torch.tensor([[[1, 3],
    #                    [4, 2],
    #                    [1, 9]],
    #                   [[1, 3],
    #                    [4, 2],
    #                    [7, 5]],
    #                   ])
    # m = CII_(source_num=N_source, head=4, hidden=256, out_hid=N_hidden, concat=True, standard_train=True).to(device)
    # m = CIE(heads=5, source_num=N_source, concat=True, standard_train=True).to(device)
    m = Choquet_Integral(heads=5, source_num=N_source, concat=True, dropout=0.5, hidden=128).to(device)
    start = time.time()
    for i in range(500):
        x = torch.randn(64, N_source, N_hidden)
        x = x.to(device)
        m.forward_mobius(x)
    print(time.time() - start)
