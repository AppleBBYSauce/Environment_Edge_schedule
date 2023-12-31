import torch
import numpy as np
from MultiMLP import MMLP
from MultiTransformer import MMTs
from torch_scatter import scatter_mean, scatter_add
from torch.distributions import MultivariateNormal
from GCN import GCN
from ChI import Choquet_Integral

CONST = (2 * torch.pi) ** 0.5


class Critic(torch.nn.Module):

    def __init__(self, hidden_nums: int, output_nums: int, neighbor_nums: list[int],
                 node_nums: int, drop_out: float = 0.1, heads: int = 3):
        super().__init__()

        self.heads = heads

        self.V_nn = torch.nn.Sequential(
            MMLP(input_num=hidden_nums, out_num=hidden_nums, multi_num=node_nums, dropout=drop_out),
            MMLP(input_num=hidden_nums, out_num=output_nums, multi_num=node_nums, act=False, ban=False,
                 dropout=drop_out),
        )

        self.ChIs = torch.nn.ModuleList([
            Choquet_Integral(neighbor_node_nums=nns - 1, dropout=drop_out, hidden=1, heads=heads)
            for nns in neighbor_nums
        ])

    def forward(self, observation, local_edges):
        Q_neighbors = self.V_nn(observation)
        ChI_Q = []
        for ChI, local_edge in zip(self.ChIs, local_edges):
            Q_neighbor = Q_neighbors[:, local_edge[0, 1:], :].data
            ChI_Q.append((ChI.forward_mobius(Q_neighbor) + Q_neighbors[:, local_edge[0, 0], :]).mean(dim=1))
        return torch.stack(ChI_Q, dim=1)

    def return_shapley_values(self):
        return [chi.generate_shapley_value() for chi in self.ChIs]


class Actor(torch.nn.Module):

    def __init__(self, hidden_nums: int, freedom_degree: list, state_nums: int, node_nums: int, drop_out=0.1):
        super().__init__()

        self.mlp = torch.nn.ModuleList([
            MMLP(input_num=hidden_nums, out_num=hidden_nums, multi_num=node_nums, dropout=drop_out) for i in
            freedom_degree
        ])

        # self.bi_rnn = torch.nn.ModuleList([
        #     torch.nn.GRU(input_size=hidden_nums, hidden_size=hidden_nums,
        #                  dropout=drop_out, bidirectional=True, num_layers=2)
        #     for _ in range(node_nums)
        # ])

        self.transformer = MMTs(multi_nums=node_nums, hidden_nums=hidden_nums, heads=3, dropout=drop_out)

        self.decide_mu = MMLP(input_num=len(freedom_degree) * hidden_nums, out_num=sum(freedom_degree), multi_num=node_nums, dropout=drop_out)

        # self.decide_In_sigma = torch.nn.ModuleList([
        #     MMLP(input_num=hidden_nums, out_num=i, multi_num=node_nums, dropout=drop_out) for i in freedom_degree
        # ])

        self.action_nums = len(freedom_degree)
        self.freedom_nums = freedom_degree[0]
        self.node_nums = node_nums
        self.magic_const = 1
        self.magic_const_temperature = 1.1
        self.VAR = torch.nn.Parameter(torch.ones(size=(node_nums, sum(freedom_degree))) * 0.5,
                                      requires_grad=False)
        self.temperature = 0.99999

    def forward(self, x, edge_index, shapley_index=None):

        x_ = []
        for mlp in self.mlp:
            x_.append(mlp(x))
        x = torch.stack(x_, dim=2)

        if shapley_index is None:
            x = scatter_add(x[:, edge_index[1], :, :], index=edge_index[0], dim=1, dim_size=self.node_nums)
        else:
            x = scatter_add(x[:, edge_index[1], :, :] * shapley_index, index=edge_index[0], dim=1,
                            dim_size=self.node_nums)


        h_I = self.transformer(x, x, x)
        action_parameters_mu = []
        action_parameter_In_sigma = []
        B, M, S, D = h_I.shape
        action_mu = self.decide_mu(h_I.view(B, M, S * D))
        action_mu = torch.where(torch.isnan(action_mu), 0, action_mu)
        act_var = self.VAR.expand_as(action_mu)
        cov_mat = torch.diag_embed(act_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        sample_action = dist.sample()
        action_logprobs = dist.log_prob(sample_action)
        action_entropy = dist.entropy()

        sample_action = sample_action.view(B, M, S, -1)

        sample_action[:, :, 0, :] = torch.softmax(torch.tanh(sample_action[:, :, 0, :]), dim=2)

        sample_action[:, :, 1, :] = torch.sigmoid(sample_action[:, :, 1, :])

        sample_action[:, :, 2, :] = torch.tanh(sample_action[:, :, 2, :])

        return sample_action, action_logprobs, action_entropy

    def constrain_action(self, task_nums, neighbor_nums, local_edges, actions):
        device = next(self.parameters()).device
        # actions = torch.clip(actions, min=0, max=1)
        if isinstance(task_nums, np.ndarray):
            task_nums = torch.tensor(task_nums).to(device)
        I_loc = actions[:, :, 2, :] > 0
        p = actions[:, :, 0, :]
        I_mig = []
        for idx, (neighbor_num, local_edge) in enumerate(zip(neighbor_nums, local_edges)):
            if local_edge.shape[0] != actions.shape[0]:
                local_edge = local_edge.expand(actions.shape[0], neighbor_num).to(device)
            I_mig.append(local_edge.gather(1, ((neighbor_num - 1) * actions[:, idx, 1, :]).floor().long()))
        return p[0], torch.stack(I_mig, dim=1)[0], I_loc[0]


class ChIMRL(torch.nn.Module):

    def __init__(self, state_nums: int,
                 freedom_nums: list,
                 hidden_nums: int,
                 agents_nums: int,
                 local_edges,
                 edge_index,
                 gamma: float,
                 step: int,
                 drop_out: float=0.3):
        super().__init__()

        gamma = torch.tensor([[gamma ** i] for i in range(step)])
        self.neighbor_nums = [i.shape[1] for i in local_edges]
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("gamma", gamma)
        self.step = step
        self.local_edges = local_edges

        self.global_gnn_update = GCN(input_num=state_nums,
                                     hidden_num=hidden_nums,
                                     out_num=hidden_nums, layer=2,
                                     node_num=len(local_edges),
                                     edge_index=edge_index,
                                     dropout=drop_out)

        self.global_gnn = GCN(input_num=state_nums,
                              hidden_num=hidden_nums,
                              out_num=hidden_nums, layer=2,
                              node_num=len(local_edges),
                              edge_index=edge_index,
                              dropout=0.0)

        self.critic = Critic(hidden_nums=hidden_nums, output_nums=1, drop_out=0,
                             node_nums=agents_nums, neighbor_nums=self.neighbor_nums)

        self.critic_update = Critic(hidden_nums=hidden_nums, neighbor_nums=self.neighbor_nums, output_nums=1,
                                    node_nums=agents_nums, drop_out=drop_out)

        self.actor = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                           node_nums=agents_nums, drop_out=0)

        self.actor_update = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                                  node_nums=agents_nums, drop_out=drop_out)

    def get_gnn(self, x, update=False):
        if update:
            return self.global_gnn_update(x)
        else:
            return self.global_gnn(x)

    def generate_act(self, observation, task_nums=None, update: bool = False):
        observation = self.get_gnn(x=observation, update=update)

        if not update:
            if (not isinstance(task_nums, np.ndarray)) and observation.device != self.gamma.device:
                observation = observation.to(self.gamma.device)
                task_nums = task_nums.to(self.gamma.device)
            action, action_prob, action_entropy = self.actor(x=observation, edge_index=self.edge_index)
        else:
            action, action_prob, action_entropy = self.actor_update(x=observation, edge_index=self.edge_index)

        if task_nums is not None:

            actions_ = self.actor.constrain_action(actions=action.data, task_nums=task_nums,
                                                   neighbor_nums=self.neighbor_nums,
                                                   local_edges=self.local_edges)
        else:
            actions_ = None
        B, M, _, _ = action.shape
        return action.view(B, M, -1), action_prob, action_entropy, actions_

    def get_pre(self, observation):
        observation = self.get_gnn(x=observation, update=True)
        return self.critic_update(observation, local_edges=self.local_edges)

    def get_target(self, observation):
        observation = self.get_gnn(x=observation, update=False)
        return torch.clip(self.critic(observation, local_edges=self.local_edges), max=0)

    def return_ChI_Q(self, states_t, states_t_plus, reward):
        reward = (reward * self.gamma[:-1, :]).sum(1)
        Q_t = self.get_pre(observation=states_t)
        Q_t_plus = reward + self.gamma[[-1], 0] * self.get_target(observation=states_t_plus)
        return Q_t, Q_t_plus

    def return_critic_update_parameters(self):
        return [{"params": self.global_gnn_update.parameters()}, {"params": self.critic_update.parameters()}]

    def return_actor_update_parameters(self):
        return [{"params": self.actor_update.parameters()}]

    def soft_update(self, flag: bool, soft_update_weight):
        if flag:
            for target_parameter, main_parameter in zip(self.actor.parameters(),
                                                        self.actor_update.parameters()):
                target_parameter.data.copy_(
                    (1 - soft_update_weight) * main_parameter + soft_update_weight * target_parameter)
        else:
            for target_parameter, main_parameter in zip(self.critic.parameters(),
                                                        self.critic_update.parameters()):
                target_parameter.data.copy_(
                    (1 - soft_update_weight) * main_parameter + soft_update_weight * target_parameter)

            for target_parameter, main_parameter in zip(self.global_gnn.parameters(),
                                                        self.global_gnn_update.parameters()):
                target_parameter.data.copy_(
                    (1 - soft_update_weight) * main_parameter + soft_update_weight * target_parameter)


if __name__ == '__main__':
    observe = torch.randn(size=(32, 4, 27))  # Batch_size ,node_nums, hidden_nums
    observe_plus = torch.randn(size=(32, 4, 27))  # Batch_size ,node_nums, hidden_nums

    action_self = torch.randn(size=(32, 4, 12))
    edge_index = torch.LongTensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 2, 3, 0, 1]])
    action_neighbor = scatter_mean(src=action_self[:, edge_index[1], :], index=edge_index[0], dim=1)
    action = torch.concat([action_self, action_neighbor], dim=-1)
    local_edges = [torch.tensor([[1, 2]]).long(),
                   torch.tensor([[2, 3]]).long(),
                   torch.tensor([[3, 0]]).long(),
                   torch.tensor([[0, 1]]).long()]
    # m = Critic(edge_index=edge_index, node_nums=4, hidden_nums=64, action_nums=12 * 2, state_nums=27, output_nums=1)
    # m(observations=observe, actions=action)
    # actor = Actor(hidden_nums=64, freedom_degree=[4, 4, 4], state_nums=27, node_nums=4)
    # actions, action_In_prob = actor(observe)
    # actor.constrain_action(task_nums=torch.randint(low=0, high=10, size=(32, 4, 4)),
    #                        neighbor_nums=[2, 2, 2, 2],
    #                        local_egdes=[torch.tensor([[1, 2]]),
    #     #                                     torch.tensor([[2, 3]]),
    #     #                                     torch.tensor([[3, 0]]),
    #     #                                     torch.tensor([[0, 1]])],
    #                        actions=actions)
    m = ChIMRL(
        state_nums=27, freedom_nums=[3, 3], agents_nums=4, local_edges=local_edges, gamma=0.9, step=2,
        hidden_nums=64, edge_index=edge_index
    )
    # m.generate_act(observation=observe, task_nums=torch.randint(low=0, high=10, size=(32, 4, 4)))
    # m.get_pre(observation=observe)
    # m.return_ChI_Q(states_t=observe_plus, states_t_plus=observe_plus, reward=torch.randn(size=(32, 1, 4)))
    x = m.critic_update.return_shapley_values()
    p = 1
