import torch


class MMLP(torch.nn.Module):

    def __init__(self, input_num: int, out_num: int, multi_num: int, dropout: float = 0.1, ban: bool = True,
                 act: bool = True, bias: bool = True):
        super().__init__()
        if act:
            self.act = torch.nn.PReLU()
        else:
            self.act = None
        self.dropout = torch.nn.Dropout(dropout)
        self.multi_num = multi_num
        self.ban = ban
        self.bias = bias

        # parameter of linear transformer
        self.register_parameter("W", torch.nn.Parameter(torch.randn(size=(multi_num, input_num, out_num))))
        if bias:
            self.register_parameter("Bias", torch.nn.Parameter(torch.randn(size=(multi_num, out_num))))
        else:
            self.register_parameter("Bias", None)

        # parameter of batch normalization
        self.ban = ban
        if ban:
            self.register_parameter("Gamma", torch.nn.Parameter(torch.randn(size=(multi_num, 1, 1))))
            self.register_parameter("Beta", torch.nn.Parameter(torch.randn(size=(multi_num, 1, 1))))
        else:
            self.register_parameter("Gamma", None)
            self.register_parameter("Beta", None)

        self.init()

    def init(self):
        torch.nn.init.kaiming_uniform_(self.W)
        if self.bias:
            torch.nn.init.constant_(self.Bias, 0)
        if self.ban:
            torch.nn.init.constant_(self.Gamma, 1)
            torch.nn.init.constant_(self.Beta, 0)

    def forward(self, x, part_nums: int = None):
        x = self.dropout(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(x.shape) > 3:
            op = "bmsi,mio->bmso"
        else:
            op = "bmi,mio->bmo"
        if part_nums is None:
            bias = self.Bias if self.bias else 0
            x = torch.einsum(op, x, self.W)
            # x = torch.where(torch.isnan(x), 0, x)
        else:
            tmp = []
            count = self.multi_num
            start = 0
            while count:
                select_num = min(part_nums, count)
                x_part = x[:, start: start + select_num, :]
                bias = self.Bias[start: start + select_num] if self.bias else 0
                x_part = self.act(
                    torch.einsum(op, x_part, self.W[start: start + select_num, :, :]) + bias)
                tmp.append(x_part)
                start += select_num
                count -= select_num
            x = torch.cat(tmp, dim=1)

        if self.act is not None:
            x = self.act(x)

        if x.size(0) > 1 and self.ban:  # if enable batch normalization
            if len(x.shape) <= 3:
                x = torch.transpose(x, 0, 1)  # B M I -> M B I
                x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
                x = x * self.Gamma + self.Beta
                x = torch.transpose(x, 0, 1)
            else:
                x = torch.einsum("bsmi->mbsi")  # B S M I -> M B S I
                x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)
                x = x * self.Gamma.unsqueeze(-1) + self.Beta.unsqueeze(-1)
                x = torch.einsum("mbsi->bsmi")
        return x


if __name__ == '__main__':
    input = torch.tensor([[
        [1, 2, 5],
        [2, 4, 19],
        [33, 44, 55],
        [12, 44, 32],
    ],
        [
            [23, 2, 14],
            [2, 42, 129],
            [12, 123, 123],
            [4, 123, 32],
        ],
    ]).float()
    import torchviz

    m = MMLP(input_num=3, multi_num=4, out_num=1, ban=False)
    opt = torch.optim.Adam(params=m.parameters())

    opt.zero_grad()

    x1 = m(input, part_nums=1)
    loss_1 = x1[:, 0, :].mean(0).sum()
    loss_1.backward()
    for x in m.parameters():
        print(x.grad)

    opt.zero_grad()

    x2 = m(input, part_nums=1)
    loss_2 = x2[:, 1, :].mean(0).sum()
    loss_2.backward()
    for x in m.parameters():
        print(x.grad)

    opt.zero_grad()

    x12 = m(input, part_nums=1)
    opt = torch.optim.Adam(params=m.parameters())
    loss_12 = x12.mean(1).sum()
    torchviz.make_dot(x12, params=dict(list(m.named_parameters()))).render()
    loss_12.backward()
    for x in m.parameters():
        print(x.grad)
