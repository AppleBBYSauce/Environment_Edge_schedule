import torch
import numpy as np
from MultiMLP import MMLP
from torch_scatter import scatter_mean, scatter_add


class Critic(torch.nn.Module):

    def __init__(self, hidden_nums: int, output_nums: int, action_nums: int,
                 state_nums: int, node_nums: int, drop_out: float = 0.1):
        super().__init__()

        # self.global_gnn = GCN(input_num=state_nums,
        #                       hidden_num=hidden_nums,
        #                       out_num=hidden_nums, layer=2,
        #                       node_num=node_nums, edge_index=edge_index, dropout=drop_out)

        self.mlp_state = MMLP(input_num=state_nums, out_num=hidden_nums, multi_num=node_nums)

        self.mlp_action = MMLP(input_num=2 * action_nums, out_num=hidden_nums, multi_num=node_nums)

        self.A_nn = torch.nn.Sequential(
            MMLP(input_num=2 * hidden_nums, out_num=hidden_nums, multi_num=node_nums),
            MMLP(input_num=hidden_nums, out_num=output_nums, multi_num=node_nums, act=False, ban=False)
        )
        self.V_nn = torch.nn.Sequential(
            MMLP(input_num=hidden_nums, out_num=hidden_nums, multi_num=node_nums),
            MMLP(input_num=hidden_nums, out_num=output_nums, multi_num=node_nums, act=False, ban=False)
        )



    def forward(self, observations, actions):
        B, N, H = observations.shape
        # observations = self.global_gnn(observations)
        h1 = self.mlp_action(actions)
        h2 = self.mlp_state(observations)
        h = torch.concat([h1, h2], dim=2).squeeze()
        advantage = self.A_nn(h).squeeze(-1)
        value = self.V_nn(h2).squeeze(-1)
        return advantage + value, value


class Actor(torch.nn.Module):

    def __init__(self, hidden_nums: int, freedom_degree: list, state_nums: int, node_nums: int, drop_out=0.1):
        super().__init__()

        self.mlp_state = torch.nn.Sequential(
            MMLP(input_num=state_nums, out_num=hidden_nums, multi_num=node_nums),
            MMLP(input_num=hidden_nums, out_num=hidden_nums, multi_num=node_nums))

        self.decide_mu = torch.nn.ModuleList([
            MMLP(input_num=hidden_nums, out_num=i, multi_num=node_nums) for i in freedom_degree
        ])

        self.decide_In_sigma = torch.nn.ModuleList([
            MMLP(input_num=hidden_nums, out_num=i, multi_num=node_nums) for i in freedom_degree
        ])

        self.register_buffer("device_flag", torch.empty(size=(0,)))

        self.action_nums = len(freedom_degree)
        self.freedom_nums = freedom_degree[0]
        self.magic_const = 1
        self.magic_const_temperature = 1.1
        self.VAR = 0.8
        self.temperature = 0.99999

    def forward(self, x):
        h = self.mlp_state(x)

        action_parameters_mu = []
        action_parameter_In_sigma = []

        for idx, mu in enumerate(self.decide_mu):
            action_parameters_mu.append(torch.sigmoid(mu(h)))

        for idx, sigma in enumerate(self.decide_In_sigma):
            action_parameter_In_sigma.append(torch.tanh(sigma(h)))

        action_mu = torch.stack(action_parameters_mu, dim=2)
        action_In_sigma = torch.stack(action_parameter_In_sigma, dim=2)

        sample_action = (action_mu + torch.randn(size=action_mu.shape, device=action_mu.device) * torch.exp(
            action_In_sigma))

        # sample_action = (action_mu + torch.randn(size=action_mu.shape, device=action_mu.device) * self.VAR)

        sample_In_prob = (-action_In_sigma / 2) - ((sample_action - action_mu) ** 2 / (2 * torch.exp(action_In_sigma)))

        # sample_In_prob = ((sample_action - action_mu) ** 2 / (2 * self.VAR**2))

        sample_action[:, :, 0, :] = torch.softmax(torch.tanh(sample_action[:, :, 0, :]), dim=2)

        sample_action[:, :, 1, :] = torch.sigmoid(sample_action[:, :, 1, :])

        return sample_action, sample_In_prob.sum(-1).sum(-1)

    def constrain_action(self, task_nums, neighbor_nums, local_edges, actions):
        device = next(self.parameters()).device
        # actions = torch.clip(actions, min=0, max=1)
        if isinstance(task_nums, np.ndarray):
            task_nums = torch.tensor(task_nums).to(device)
        p = actions[:, :, 0, :]
        I_mig = []
        for idx, (neighbor_num, local_edge) in enumerate(zip(neighbor_nums, local_edges)):
            if local_edge.shape[0] != actions.shape[0]:
                local_edge = local_edge.expand(actions.shape[0], neighbor_num).to(device)
            I_mig.append(local_edge.gather(1, ((neighbor_num - 1) * actions[:, idx, 1, :]).floor().long()))
        return p[0], torch.stack(I_mig, dim=1)[0]


class MFAC(torch.nn.Module):

    def __init__(self, state_nums: int,
                 freedom_nums: list,
                 hidden_nums: int,
                 agents_nums: int,
                 local_edges,
                 edge_index,
                 alpha: float,
                 gamma: float, step: int):
        super().__init__()

        self.critic = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                             node_nums=agents_nums, state_nums=state_nums, drop_out=0.0)

        self.critic_update = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                                    node_nums=agents_nums, state_nums=state_nums, drop_out=0.35)

        self.actor = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                           node_nums=agents_nums, drop_out=0.0)

        self.actor_update = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                                  node_nums=agents_nums, drop_out=0.35)

        gamma = torch.tensor([[gamma ** i] for i in range(step)])
        self.neighbor_nums = [i.shape[1] for i in local_edges]
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("gamma", gamma)
        self.local_edges = local_edges
        self.alpha = alpha

    def generate_act(self, observation, task_nums=None, update: bool = False):

        if not update:
            if (not isinstance(task_nums, np.ndarray)) and observation.device != self.gamma.device:
                observation = observation.to(self.gamma.device)
                task_nums = task_nums.to(self.gamma.device)
            action, action_prob = self.actor(observation)
        else:
            action, action_prob = self.actor_update(observation)

        if task_nums is not None:

            actions_ = self.actor.constrain_action(actions=action.data, task_nums=task_nums,
                                                   neighbor_nums=self.neighbor_nums,
                                                   local_edges=self.local_edges)
        else:
            actions_ = None

        B, M, _, _ = action.shape
        return action.view(B, M, -1), action_prob, actions_

    def get_pre(self, observation, actions):
        return self.critic_update(observation, actions)

    def get_target(self, observation, actions):
        Q, V = self.critic(observation, actions)
        return Q, V

    def Q_mul_prob(self, Qs, probs):
        return scatter_add((Qs * probs)[:, self.edge_index[1]], dim_size=len(self.local_edges), dim=1,
                           index=self.edge_index[0])

    def return_avg_joint_actions(self, actions):
        average_actions_t = scatter_mean(src=actions[:, self.edge_index[1], :],
                                         index=self.edge_index[0],
                                         dim=1,
                                         dim_size=len(self.local_edges))
        return torch.concat([actions, average_actions_t], dim=-1)

    def return_mean_field_Q(self, states_t, states_t_plus, action_t, action_probs_t, reward):
        reward = (reward * self.gamma[:-1, :]).sum(1)
        action_t = self.return_avg_joint_actions(actions=action_t)
        Q_t, _ = self.get_pre(observation=states_t, actions=action_t)
        Q_t_plus, _ = self.get_target(observation=states_t_plus, actions=action_t)
        V_s_prime = reward * self.gamma[[-1], :] * self.Q_mul_prob(Qs=Q_t_plus.data, probs=torch.exp(action_probs_t))
        # Q_t_plus = (1 - self.alpha) * Q_t + self.alpha * (reward + self.gamma[1, 0] * V_s_prime)
        return Q_t, V_s_prime


if __name__ == '__main__':
    observe = torch.randn(size=(32, 4, 27))  # Batch_size ,node_nums, hidden_nums
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
    m = MFAC(
        state_nums=27, freedom_nums=[4, 4, 4], agents_nums=4, local_edges=local_edges, gamma=0.9, step=2,
        hidden_nums=64,
    )
    m.generate_act(observation=observe, task_nums=torch.randint(low=0, high=10, size=(32, 4, 4)))
    m.get_pre(observation=observe, actions=action)
