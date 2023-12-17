import torch
from layer_acm import FPs, DjFNNConv, Choquet_Integral, FNNConv, CII_, CII


class FLGnnH(torch.nn.Module):

    def __init__(self,
                 semantic_num: int,
                 hidden: int,
                 out_channels: int,
                 num_mf: int,
                 feature_projector_layers: int,
                 val_interval: [int, int],
                 cross: float,
                 window_size: int,
                 stride_size: int,
                 dropout: float,
                 refine_ratio: float,
                 choquet: bool,
                 choquet_heads: int,
                 choquet_concat: bool,
                 norm: bool,
                 concat: bool,
                 fix: bool,
                 fuzzy: bool,
                 data_size: dict,
                 target:str,
                 residual:bool,
                 **kwargs):
        super().__init__()
        self.sem_num = semantic_num
        self.feature_projector_layers = feature_projector_layers
        self.hidden = hidden
        self.out_chs = out_channels
        self.concat = concat
        self.choquet = choquet
        self.chi_concat = choquet_concat
        self.chi_heads = choquet_heads
        self.fuzzy = fuzzy

        extract_dim = self.sem_num if not self.choquet else (self.chi_heads if self.chi_concat else 1)
        self.Fps = FPs(c_in=extract_dim, n_embed=self.hidden, n_out=self.hidden, layers=1, dropout=dropout)

        if fuzzy:
            self.FNNConvS = FNNConv(num_mf=num_mf,
                                    val_interval=val_interval,
                                    cross=cross,
                                    window_size=window_size,
                                    stride_size=stride_size,
                                    dropout=dropout,
                                    refine_ratio=refine_ratio,
                                    extract_dim=extract_dim,
                                    fix=fix,
                                    n_hidden=hidden,
                                    norm=norm)
            # self.FNNConvS = DjFNNConv(
            #     num_mf=num_mf,
            #     val_interval=val_interval,
            #     cross=cross,
            #     dropout=dropout,
            #     extract_dim=extract_dim,
            #     fix=fix,
            #     n_hidden=hidden,
            #     norm=norm,
            #     n_node=hidden,
            #     concat=concat,
            #     out_chn=hidden,
            # )

        else:
            self.FNNConvS = None

        if self.choquet:
            # self.ChI = CII(source_num=semantic_num,
            #                concat=choquet_concat,
            #                heads=choquet_heads,
            #                dropout=dropout,
            #                out_hid=hidden)
            self.ChI = Choquet_Integral(source_num=semantic_num,
                                        concat=choquet_concat,
                                        heads=choquet_heads,
                                        dropout=dropout,
                                        hidden=hidden,
                                        meta_path=list(data_size.keys()))
        else:
            self.ChI = None

        outer_hid = self.hidden * ((semantic_num if self.concat else 0) + extract_dim)
        self.concat_layer = torch.nn.Linear(in_features=outer_hid, out_features=hidden)
        self.act = torch.nn.PReLU()

        self.target = target
        if residual:
            self.residual = torch.nn.Linear(in_features=hidden, out_features=hidden)

        else:
            self.residual = None
        self.outer = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.BatchNorm1d(num_features=hidden),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Dropout(p=dropout),
            torch.nn.BatchNorm1d(num_features=hidden),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden, self.out_chs)
        )

        self.input_drop = torch.nn.Dropout(dropout)
        self.embedings = torch.nn.ParameterDict({})
        for k, v in data_size.items():
            if v != hidden:
                self.embedings[k] = torch.nn.Parameter(
                    torch.Tensor(v, hidden).uniform_(-0.5, 0.5))
        # self.attention = MultiheadAttention(embed_dim=self.hidden, num_heads=choquet_heads)


    def forward(self, feats_dict, *args):

        if self.embedings is not None:
            for k, v in feats_dict.items():
                if k in self.embedings:
                    feats_dict[k] = v @ self.embedings[k]

        x = self.input_drop(torch.stack(list(feats_dict.values()), dim=1))

        if self.choquet:
            xf = self.ChI.forward_mobius(x)
        else:
            xf = x.clone()

        xf = self.Fps(xf)

        if self.FNNConvS:
            xf = self.FNNConvS(xf)
        if self.concat:
            xf = torch.concat([x.contiguous().view(x.size(0), -1), xf.contiguous().reshape(x.size(0), -1)], dim=-1)
        else:
            # xf, _ = self.attention(xf, xf, xf)
            xf = xf.contiguous().view(xf.size(0), -1)
        xf = self.concat_layer(xf)
        if self.residual:
            xf = xf + self.residual(feats_dict[self.target])

        xf = self.act(xf)
        xf = self.outer(xf)
        return xf
