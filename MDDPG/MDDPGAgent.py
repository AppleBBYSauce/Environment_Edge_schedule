# Item's name: RandomAgent
# Autor: bby
# DateL 2023/12/3 18:41
from MDDPG import MDDPG
from ACAgent import ACAgent
from TaskQueue import TaskQueue
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

np.random.seed(0)


def softmax(x):
    x -= np.max(x)
    exp_sum = np.sum(np.exp(x))
    return np.exp(x) / exp_sum


class MDDPGAgent(ACAgent):

    def __init__(self, service_nums: int, f: int, f_mean: int, rho: float, tr: int, dm: int,
                 idx: int, cpu_cycles, avg_arrive_nums, max_run_memorys, local_edge, step, lr: float,
                 state_nums: int, freedom_nums: list, hidden_nums: int, agents_nums: int, soft_upate_weight: float,
                 w_cpu: float = 0.5, w_memory: float = 0.5, refresh_frequency: int = 30, sw: float = 30,
                 task_queue_size: int = None, gamma=0.9, soft_update_step: int = 10):

        super().__init__(service_nums=service_nums, f=f, f_mean=f_mean, rho=rho, tr=tr, dm=dm, local_edge=local_edge,
                         idx=idx, w_cpu=w_cpu, w_memory=w_memory, refresh_frequency=refresh_frequency,
                         task_queue_size=task_queue_size,
                         cpu_cycles=cpu_cycles, avg_arrive_nums=avg_arrive_nums, max_run_memorys=max_run_memorys, sw=sw)

        self.local_edge_index = torch.zeros(size=(2, self.local_edge.shape[-1]), dtype=torch.int64)
        self.local_edge_index[1, :] = torch.arange(start=0, end=self.local_edge.shape[-1])
        self.soft_update_weight = soft_upate_weight
        self.soft_update_step = soft_update_step
        self.lr = lr
        self.step = step
        self.VAR = 1
        self.temperature = 1.01



        self.MDDPGS = MDDPG(state_nums=state_nums, freedom_nums=freedom_nums, hidden_nums=hidden_nums, gamma=gamma,
                            step=step,
                            agents_nums=agents_nums, edge_index=self.local_edge_index, local_edge=local_edge)

    def generate_action(self, observation, task_nums, update=False):

        actions, actions_constrain = self.MDDPGS.generate_act(observation=observation, task_nums=task_nums, update=update, VAR=self.VAR)

        return actions, actions_constrain

    def act(self, new_task_queue: TaskQueue, observation):

        self.temporary_state_inf: TaskQueue = new_task_queue

        self.temporary_state_inf.set_recycle(self.re_direct_queue)  # set re_direct queue as the recycle station

        action, action_constrain = self.generate_action(observation=observation, task_nums=new_task_queue.queue_size)

        p, I_loc, I_mig = action
        self.p, self.I_loc, self.I_mig = action_constrain

        self.p = self.p[0]
        self.I_loc = self.I_loc[0]
        self.I_mig = self.I_mig[0]

        self.pre_process()

        done_task = self.process_cur()

        load_ratio, task_latency = self.process_post()

        result = [

            # action
            torch.concat([I_loc[0], I_mig[0], p[0]], 0),

            # action probability
            torch.empty(size=(1,)),

            # states
            observation[0],

            # reward
            torch.tensor([load_ratio, task_latency]),

            # complete num
            done_task.get_size()
        ]

        return result, self.idx

    def return_states(self):
        return torch.concat([
            torch.tensor(self.task_queue.get_size()),
            self.p.data,
            torch.tensor(self.task_queue.return_mean()).view(-1),
            torch.tensor(self.task_queue.return_std()).view(-1),
        ], dim=0)

    def return_action(self):
        return torch.tensor(np.array([self.I_loc, self.I_mig, self.p]))

    def update_policy(self, *args, **kwargs):
        pass

    def update_critic(self, data):

        # update critic
        optimizer = torch.optim.AdamW(params=self.MDDPGS.critic_update.parameters(), lr=self.lr)
        scaler = GradScaler()
        TD_recorder = []
        for e, data_each in enumerate(data):

            action, action_probs, state, reward, task_nums, self_state = data_each
            state_t = state[:, 0, :, :]
            state_t_plus = state[:, 1, :, :]
            action_t = action[:, 0, :, :]
            action_t_plus = action[:, 1, :, :]
            reward = reward[:, :-1]

            with autocast():
                reward = torch.sum(self.MDDPGS.gamma * reward, dim=-1)
                Q_pre = self.MDDPGS.get_pre(observation=state_t, actions=action_t.data)
                Q_target = self.MDDPGS.get_target(observation=state_t_plus, actions=action_t_plus.data) + reward
                TD_Error = Q_target - Q_pre
                TD_recorder.append(TD_Error)

            optimizer.zero_grad()
            scaler.scale(torch.pow(TD_Error, 2).mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            if e % self.soft_update_step == 0:
                for target_parameter, main_parameter in zip(self.MDDPGS.critic.parameters(),
                                                            self.MDDPGS.critic_update.parameters()):
                    target_parameter.data.copy_(
                        self.soft_update_weight * main_parameter + (1 - self.soft_update_weight )* target_parameter)

        for target_parameter, main_parameter in zip(self.MDDPGS.critic.parameters(),
                                                    self.MDDPGS.critic_update.parameters()):
            target_parameter.data.copy_(
                (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)

        torch.cuda.empty_cache()
        return TD_recorder

    def actor_loss(self, state_t, action_t):
        with autocast():
            return self.MDDPGS.get_target(observation=state_t, actions=action_t).mean()


if __name__ == '__main__':
    pass
