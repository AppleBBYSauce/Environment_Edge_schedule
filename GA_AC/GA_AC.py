# Item's name: GA_AC
# Autor: bby
# DateL 2023/12/6 20:27

import torch
from torch_scatter import scatter
from torch.nn.functional import gumbel_softmax, softmax
from torch_geometric.utils import degree
from GCN import GCN
import numpy as np

class TwoStageAttention(torch.nn.Module):

    def __init__(self, input_nums: int, hidden_nums: int, node_nums: int = 0, dropout: float = 0.1):
        super().__init__()

        self.rnn = torch.nn.GRU(input_size=hidden_nums, hidden_size=hidden_nums, batch_first=False, dropout=dropout,
                                bidirectional=True, num_layers=2)

        self.drop_out = torch.nn.Dropout(dropout)

        self.rnn_connect = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=2 * hidden_nums, out_features=1),
            torch.nn.GELU())

        self.q = torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums)
        self.k = torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums)

    def forward(self, h, edge_index):
        h = self.drop_out(h)
        h_source = h[:, edge_index[1], :]
        h_target = h[:, edge_index[0], :]
        B, N, H = h.shape
        buffer = [[None for _ in range(N)] for __ in range(N)]

        edge_hard_weight = torch.zeros(size=(B, edge_index.shape[1]), device=edge_index.device)
        edge_soft_weight = torch.zeros(size=(B, edge_index.shape[1]), device=edge_index.device)

        # hard attention
        h_hard = torch.stack([h_target, h_source], dim=0).view(2, -1, H)
        h_hard = self.rnn(h_hard)[0][0]
        h_hard = self.rnn_connect(h_hard.view(B, -1, 2 * H)).squeeze(-1)

        for i in range(N):
            idx = edge_index[1] == i
            edge_hard_weight[:, idx] = gumbel_softmax(h_hard[:, idx], dim=1, tau=0.1)
            edge_soft_weight[:, idx] = softmax(h_hard[:, idx], dim=1)

        # soft attention
        q = self.q(h_source)
        k = self.k(h_target)
        h_soft = torch.einsum("bnh,bnh->bn", q, k)

        edge_hard_weight = edge_hard_weight.unsqueeze(-1)
        edge_soft_weight = edge_soft_weight.unsqueeze(-1)

        return scatter(src=h_source * edge_soft_weight * edge_hard_weight, index=edge_index[0], dim=1, dim_size=N)


class Critic(torch.nn.Module):

    def __init__(self, hidden_nums: int, output_nums: int, action_nums: int,
                 state_nums: int, node_nums: int, edge_index, drop_out: float = 0.1):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=state_nums + action_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=node_nums),
            torch.nn.GELU(),
        )
        self.twoStageAttention = TwoStageAttention(input_nums=hidden_nums, hidden_nums=hidden_nums)
        self.GNN = GCN(input_num=hidden_nums, hidden_num=hidden_nums, out_num=output_nums, layer=2,
                       edge_index=edge_index, dropout=drop_out, node_num=node_nums)

    def forward(self, observations, actions, edge_index):
        B, N, H = observations.shape
        h = self.encoder(torch.concat([observations, actions.view(B, N, -1)], dim=2))
        h = self.twoStageAttention(h, edge_index)
        h = self.GNN(h, edge_index)
        h = torch.sum(h, dim=1).squeeze()
        return h


class Actor(torch.nn.Module):

    def __init__(self, hidden_nums: int, freedom_degree: list, state_nums: int):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
        )

        self.decide_mu = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_nums, out_features=num) for num in freedom_degree]
        )

        self.decide_sigma = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_nums, out_features=num) for num in freedom_degree]
        )

        self.action_nums = len(freedom_degree)
        self.freedom_nums = freedom_degree[0]
        # self.action_nums = [[None, None] for _ in range(freedom_degree)]

    def forward(self, x):
        x = self.nn(x)
        action_parameters_mu = []
        action_parameters_lnsigma = []
        for idx, (mu, sigma) in enumerate(zip(self.decide_mu, self.decide_sigma)):
            action_parameters_mu.append(torch.sigmoid_(mu(x)))
            action_parameters_lnsigma.append(torch.tanh_(sigma(x)))

        action_parameters_mu = torch.stack(action_parameters_mu, dim=1)
        action_parameters_lnsigma = torch.stack(action_parameters_lnsigma, dim=1)
        normal_random = torch.randn(size=action_parameters_mu.shape, device=action_parameters_mu.device)
        randn_action = torch.clip_((action_parameters_mu + normal_random) * torch.sqrt(torch.exp(action_parameters_lnsigma)).detach(), min=0, max=1)
        action_log_probability = -(action_parameters_lnsigma / 2) - ((randn_action - action_parameters_mu)**2 / (2 * torch.exp(action_parameters_lnsigma))) + 5
        return randn_action, torch.einsum("ijk->i", action_log_probability)

    def constrain_action(self, task_nums, neighbor_nums, local_egde, actions):
        if isinstance(task_nums, np.ndarray):
            task_nums = torch.tensor(task_nums).to(local_egde.device)

        I_loc = (task_nums * actions[:, 0, :]).floor().int()
        p = torch.softmax(actions[:, 1, :], dim=1)
        if local_egde.shape[0] != actions.shape[0]:
            local_egde = local_egde.expand(actions.shape[0], neighbor_nums)
        I_mig = local_egde.gather(1, ((neighbor_nums-1) * actions[:, 2, :]).floor().long())
        return p, I_loc, I_mig


class GA_AC(torch.nn.Module):

    def __init__(self, state_nums: int, freedom_nums: list, hidden_nums: int, agents_nums: int, edge_index, local_edge,
                 gamma: float, step: int):
        super().__init__()

        self.critic = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                             node_nums=local_edge.shape[1], state_nums=state_nums, edge_index=edge_index)
        self.critic_update = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                                    node_nums=local_edge.shape[1], state_nums=state_nums, edge_index=edge_index)

        self.actor = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums)
        self.actor_update = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums)

        gamma = torch.tensor([gamma ** i for i in range(step - 1)])
        self.register_buffer("gamma", gamma)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("local_edge", local_edge)

    def generate_act(self, observation, task_nums, update:bool=False):

        if (not isinstance(task_nums, np.ndarray)) and observation.device != self.local_edge.device:
            observation = observation.to(self.local_edge.device)
            task_nums = task_nums.to(self.local_edge.device)

        if not update:
            actions, a_prob = self.actor(observation)
            actions = self.actor.constrain_action(actions=actions, task_nums=task_nums, neighbor_nums=self.local_edge.shape[1],
                                                  local_egde=self.local_edge)
        else:
            actions, a_prob = self.actor_update(observation)
            actions = self.actor_update.constrain_action(actions=actions,
                                                         task_nums=task_nums,
                                                         neighbor_nums=self.local_edge.shape[1],
                                                         local_egde=self.local_edge)
        return actions, a_prob

    def forward(self, observation, actions):
        q_values = self.critic(observation, torch.stack(actions, dim=1), self.edge_index)
        return q_values

    def get_pre(self, observation, actions):
        return self.critic_update(observation, actions, self.edge_index)

    def get_target(self, observation, actions):
        return self.critic(observation, actions, self.edge_index)


if __name__ == '__main__':
    observe = torch.randn(size=(32, 4, 27))  # Batch_size ,node_nums, hidden_nums
    edge_index = torch.LongTensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 2, 3, 0, 1]])
    m = GA_AC(state_nums=33, freedom_nums=[4, 4, 4], hidden_nums=64, agents_nums=4, edge_index=edge_index)
    m(observe, edge_index)
    # actor = Actor(hidden_nums=64, freedom_degree=[4, 4, 4], state_nums=27)
    # critic = Critic(state_nums=27, action_nums=12, hidden_nums=64, output_nums=1)
    # a, p = actor(observe[:, 0, :])
    # a_constrain = actor.constrain_action([1, 2, 3, 4], 5, a)
    # q = critic(observe, a, edge_index)

    # x = torch.randn(size=(4, 128)) # N, D
    # edge_index = torch.LongTensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 2, 3, 0, 1]]) # 2, M
    # edge_features = torch.concat([x[edge_index[0]], x[edge_index[1]]], dim=1)
    # p = 1
