# Item's name: RandomAgent
# Autor: bby
# DateL 2023/12/3 18:41
from typing import List
from GA_AC import GA_AC
from ACAgent import ACAgent
from Services import Services
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


class GAACAgent(ACAgent):

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
        self.GA_AC = GA_AC(state_nums=state_nums, freedom_nums=freedom_nums, hidden_nums=hidden_nums, gamma=gamma,
                           step=step,
                           agents_nums=agents_nums, edge_index=self.local_edge_index, local_edge=local_edge)

    def generate_action(self, observation, task_nums, update=False):

        actions, probaility = self.GA_AC.generate_act(observation=observation, task_nums=task_nums, update=update)

        return *actions, probaility

    def act(self, new_task_queue: TaskQueue, observation):

        self.tempoary_state_inf: TaskQueue = new_task_queue

        self.tempoary_state_inf.set_recycle(self.re_direct_queue)  # set re_direct queue as the recycle station

        self.p, self.I_loc, self.I_mig, a_prob = self.generate_action(observation=observation,
                                                                      task_nums=new_task_queue.queue_size)
        self.p = self.p[0]
        self.I_loc = self.I_loc[0]
        self.I_mig = self.I_mig[0]

        self.pre_process()

        done_task = self.process_cur()

        load_ratio, task_latency = self.process_post()

        result = [

            # action
            torch.concat([self.I_loc, self.I_mig, self.p], 0),

            # action probability
            a_prob,

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
            torch.tensor(self.p),
            torch.tensor(self.task_queue.return_mean()).view(-1),
            torch.tensor(self.task_queue.return_std()).view(-1),
        ], dim=0)

    def return_action(self):
        return torch.tensor(np.array([self.I_loc, self.I_mig, self.p]))

    def update_policy(self, *args, **kwargs):
        pass

    def update_critic(self, data):

        # update critic
        optimizer = torch.optim.AdamW(params=self.GA_AC.critic_update.parameters(), lr=self.lr)
        scaler = GradScaler()
        TD_recorder = []
        for e, data_each in enumerate(data):

            action, action_probs, state, reward, task_nums, self_state = data_each
            state_t = state[:, 0, :, :].detach()
            state_t_plus = state[:, 1, :, :].detach()
            action_t = action[:, 0, :, :].detach()
            action_t_plus = action[:, 1, :, :].detach()
            reward = reward[:, :-1].detach()

            with autocast():
                reward = torch.sum(self.GA_AC.gamma * reward, dim=-1)
                Q_pre = self.GA_AC.get_pre(observation=state_t, actions=action_t)
                Q_target = self.GA_AC.get_target(observation=state_t_plus, actions=action_t_plus) + reward
                TD_Error = (Q_target - Q_pre)
                TD_recorder.append(TD_Error)

            optimizer.zero_grad()
            scaler.scale(TD_Error.mean()).backward()
            scaler.step(optimizer)
            scaler.update()

            if e % self.soft_update_step == 0:
                for target_parameter, main_parameter in zip(self.GA_AC.critic.parameters(),
                                                            self.GA_AC.critic_update.parameters()):
                    target_parameter.data.copy_(
                        (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)

        for target_parameter, main_parameter in zip(self.GA_AC.critic.parameters(),
                                                    self.GA_AC.critic_update.parameters()):
            target_parameter.data.copy_(
                (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)
        torch.cuda.empty_cache()
        return TD_recorder

    def actor_loss(self, reward, state_t, action_t,
                   action_t_plus, action_prob_t, state_t_plus, TD_error,
                   action_prob_sample, import_sample: bool = True, epsilon: float = 0.1):

        reward = torch.sum(self.GA_AC.gamma * reward, dim=-1)
        with autocast():
            # Q_t = self.GA_AC.get_target(observation=state_t, actions=action_t)
            # Q_t_plus = self.GA_AC.get_target(observation=state_t_plus, actions=action_t_plus) + reward
            # Advantage = (Q_t_plus - Q_t).detach()
            Advantage = TD_error.detach()
            if import_sample:
                weight = torch.exp(action_prob_t) / torch.exp(action_prob_sample.detach())
            else:
                weight = 1
            clip_weight = torch.clip_(weight, 1 - epsilon, 1 + epsilon)
            loss = -torch.min(weight * Advantage, clip_weight * Advantage)
        return loss.mean()


if __name__ == '__main__':
    pass
