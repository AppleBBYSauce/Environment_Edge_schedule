# Item's name: GA_AC
# Autor: bby
# DateL 2023/12/6 20:27

import torch
from torch_scatter import scatter
from torch.nn.functional import gumbel_softmax, softmax
from torch_geometric.utils import degree
import numpy as np


class Critic(torch.nn.Module):

    def __init__(self, hidden_nums: int, output_nums: int, action_nums: int,
                 state_nums: int, node_nums: int, edge_index, drop_out: float = 0.1):
        super().__init__()

        self.mlp_state = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=state_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=node_nums),
            torch.nn.GELU(),
        )

        self.mlp_action = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features= action_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=node_nums),
            torch.nn.GELU(),
        )

        self.outer = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=2 * hidden_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=output_nums),
        )

    def forward(self, observations, actions, edge_index):
        B, N, H = observations.shape
        h1 = self.mlp_action(actions.view(B, N, -1))
        h2 = self.mlp_state(observations)
        h = torch.sum(torch.concat([h1, h2], dim=2), dim=1).squeeze()
        return self.outer(h)


class Actor(torch.nn.Module):

    def __init__(self, hidden_nums: int, freedom_degree: list, state_nums: int):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
        )

        self.decide_mu = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_nums, out_features=num) for num in freedom_degree]
        )

        self.action_nums = len(freedom_degree)
        self.freedom_nums = freedom_degree[0]
        # self.action_nums = [[None, None] for _ in range(freedom_degree)]

    def forward(self, x, VAR=None):
        x = self.nn(x)
        action_parameters_mu = []
        for idx, (mu, sigma) in enumerate(zip(self.decide_mu, self.decide_sigma)):
            action_parameters_mu.append(torch.sigmoid_(mu(x)))
        action = torch.stack(action_parameters_mu, dim=1)
        if VAR is not None:
            action = torch.distributions.Normal(loc=action, scale=VAR)
            action = action.sample()
        return torch.clip_(action, min=0, max=1)

    def constrain_action(self, task_nums, neighbor_nums, local_egde, actions):
        if isinstance(task_nums, np.ndarray):
            task_nums = torch.tensor(task_nums).to(local_egde.device)
        I_loc = (task_nums * actions[:, 0, :]).floor().int()
        p = torch.softmax(actions[:, 1, :], dim=1)
        if local_egde.shape[0] != actions.shape[0]:
            local_egde = local_egde.expand(actions.shape[0], neighbor_nums)
        I_mig = local_egde.gather(1, ((neighbor_nums-1) * actions[:, 2, :]).floor().long())
        return p, I_loc, I_mig


class MDDPG(torch.nn.Module):

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

    def generate_act(self, observation, task_nums):

        if (not isinstance(task_nums, np.ndarray)) and observation.device != self.local_edge.device:
            observation = observation.to(self.local_edge.device)
            task_nums = task_nums.to(self.local_edge.device)

        actions, a_prob = self.actor(observation)
        actions = self.actor.constrain_action(actions=actions, task_nums=task_nums, neighbor_nums=self.local_edge.shape[1],
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
