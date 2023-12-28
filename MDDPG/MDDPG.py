import torch
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
            torch.nn.Linear(in_features=action_nums, out_features=hidden_nums),
            torch.nn.BatchNorm1d(num_features=node_nums),
            torch.nn.GELU(),
        )

        self.A_nn = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=2 * hidden_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=output_nums))

        self.V_nn = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=output_nums)
        )

    def forward(self, observations, actions, edge_index):
        B, N, H = observations.shape
        h1 = self.mlp_action(actions.view(B, N, -1)).sum(dim=1)
        h2 = self.mlp_state(observations).sum(dim=1)
        h = torch.concat([h1, h2], dim=1).squeeze()
        advantage = self.A_nn(h)
        value = self.V_nn(h2)
        return advantage + value


class Actor(torch.nn.Module):

    def __init__(self, hidden_nums: int, freedom_degree: list, state_nums: int, node_nums: int, drop_out=0.1):
        super().__init__()

        self.mlp_state = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(in_features=state_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
        )

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=hidden_nums, out_features=hidden_nums),
            # torch.nn.BatchNorm1d(num_features=hidden_nums),
            torch.nn.GELU(),
        )

        self.decide_mu = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_nums, out_features=num) for num in freedom_degree]
        )

        self.register_buffer("device_flag", torch.empty(size=(0,)))

        self.action_nums = len(freedom_degree)
        self.freedom_nums = freedom_degree[0]
        # self.action_nums = [[None, None] for _ in range(freedom_degree)]
        self.magic_const = 1
        self.magic_const_temperature = 1.1

    def forward(self, x, VAR=None):
        h = self.mlp_state(x)
        h = self.nn(h)

        action_parameters_mu = []
        for idx, mu in enumerate(self.decide_mu):
            action_parameters_mu.append(torch.sigmoid_(mu(h)))
        action = torch.stack(action_parameters_mu, dim=1)
        if VAR is not None:
            normal_random = torch.randn(size=action.shape, device=self.device_flag.device)
            action = (action + normal_random * VAR)
        return torch.clip_(action, min=0, max=1)

    def constrain_action(self, task_nums, neighbor_nums, local_egde, actions):
        if isinstance(task_nums, np.ndarray):
            task_nums = torch.tensor(task_nums).to(local_egde.device)
        I_loc = (task_nums * actions[:, 0, :]).floor().int()
        p = torch.softmax(actions[:, 1, :], dim=1)
        if local_egde.shape[0] != actions.shape[0]:
            local_egde = local_egde.expand(actions.shape[0], neighbor_nums)
        I_mig = local_egde.gather(1, ((neighbor_nums - 1) * actions[:, 2, :]).floor().long())
        return p, I_loc, I_mig


class MDDPG(torch.nn.Module):

    def __init__(self, state_nums: int, freedom_nums: list, hidden_nums: int, agents_nums: int, edge_index, local_edge,
                 gamma: float, step: int):
        super().__init__()

        self.critic = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                             node_nums=local_edge.shape[1], state_nums=state_nums, edge_index=edge_index)
        self.critic_update = Critic(hidden_nums=hidden_nums, action_nums=sum(freedom_nums), output_nums=1,
                                    node_nums=local_edge.shape[1], state_nums=state_nums, edge_index=edge_index)

        self.actor = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                           node_nums=local_edge.shape[1])
        self.actor_update = Actor(hidden_nums=hidden_nums, freedom_degree=freedom_nums, state_nums=state_nums,
                                  node_nums=local_edge.shape[1])

        gamma = torch.tensor([gamma ** i for i in range(step - 1)])
        self.register_buffer("gamma", gamma)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("local_edge", local_edge)

    def generate_act(self, observation, task_nums, VAR, update: bool = False):

        if (not isinstance(task_nums, np.ndarray)) and observation.device != self.local_edge.device:
            observation = observation.to(self.local_edge.device)
            task_nums = task_nums.to(self.local_edge.device)

        if not update:
            actions = self.actor(observation, VAR)
            actions_ = self.actor.constrain_action(actions=actions.data, task_nums=task_nums,
                                                  neighbor_nums=self.local_edge.shape[1],
                                                  local_egde=self.local_edge)
        else:
            actions = self.actor_update(observation, VAR)
            actions_ = self.actor_update.constrain_action(actions=actions.data, task_nums=task_nums,
                                                         neighbor_nums=self.local_edge.shape[1],
                                                         local_egde=self.local_edge)
        return [actions[:, i, :] for i in range(3)], actions_

    def forward(self, observation, actions):
        q_values = self.critic(observation, torch.stack(actions, dim=1), self.edge_index)
        return q_values

    def get_pre(self, observation, actions):
        return self.critic_update(observation, actions, self.edge_index)

    def get_target(self, observation, actions):
        return torch.clip_(self.critic(observation, actions, self.edge_index), max=0)


if __name__ == '__main__':
    observe = torch.randn(size=(32, 4, 27))  # Batch_size ,node_nums, hidden_nums
    action = [torch.randn(size=(32, 12)) for _ in range(4)]
    edge_index = torch.LongTensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 2, 3, 0, 1]])
    m = MDDPG(state_nums=27, freedom_nums=[4, 4, 4], hidden_nums=64, agents_nums=4, edge_index=edge_index,
              local_edge=torch.tensor([[0, 1, 2, 3]]), gamma=0.9, step=1)
    # m(observe, action)
    m.generate_act(observation=observe, task_nums=np.random.randint(low=1, high=10, size=(32, 4)))

    # actor = Actor(hidden_nums=64, freedom_degree=[4, 4, 4], state_nums=27)
    # critic = Critic(state_nums=27, action_nums=12, hidden_nums=64, output_nums=1)
    # a, p = actor(observe[:, 0, :])
    # a_constrain = actor.constrain_action([1, 2, 3, 4], 5, a)
    # q = critic(observe, a, edge_index)

    # x = torch.randn(size=(4, 128)) # N, D
    # edge_index = torch.LongTensor([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 2, 3, 0, 1]]) # 2, M
    # edge_features = torch.concat([x[edge_index[0]], x[edge_index[1]]], dim=1)
    # p = 1
