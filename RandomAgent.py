# Item's name: RandomAgent
# Autor: bby
# DateL 2023/12/3 18:41
from typing import List

from ACAgent import ACAgent
from Services import Services
from TaskQueue import TaskQueue
import numpy as np
import _heapq
import torch
import copy

np.random.seed(0)


def softmax(x):
    x -= np.max(x)
    exp_sum = np.sum(np.exp(x))
    return np.exp(x) / exp_sum


class RandomAgent(ACAgent):

    def __init__(self, service_nums: int, f: int, f_mean: int, rho: float, tr: int, dm: int, local_edge: list,
                 idx: int, cpu_cycles, avg_arrive_nums, max_run_memorys, delta_t,w_cpu: float = 0.5,
                 w_memory: float = 0.5, refresh_frequency: int = 30, sw: float = 30, task_queue_size: int = None
                 ):

        super().__init__(service_nums=service_nums, f=f, f_mean=f_mean, rho=rho, tr=tr, dm=dm, lcoal_edge=local_edge,
                         idx=idx, w_cpu=w_cpu, w_memory=w_memory, refresh_frequency=refresh_frequency, task_queue_size=task_queue_size,
                         cpu_cycles=cpu_cycles, avg_arrive_nums=avg_arrive_nums, max_run_memorys=max_run_memorys,sw=sw,
                         delta_t = delta_t)

    def generate_action(self, *args, **kwargs):
        # re-allocate calculation resource
        p = torch.nn.functional.softmax(torch.rand(size=[self.service_nums]), dim=0).numpy()
        # select tasks into process queue
        I_loc = (torch.rand(size=[self.service_nums]) * torch.tensor(
            self.tempoary_state_inf.get_size())).floor().int().numpy()
        # select re-direct tasks
        I_mig = np.random.choice(self.local_edge, size=self.service_nums)
        return p, I_loc, I_mig, 0

    def act(self, new_task_queue: TaskQueue):

        self.tempoary_state_inf: TaskQueue = new_task_queue

        self.tempoary_state_inf.set_recycle(self.re_direct_queue)  # set re_direct queue as the recycle station

        self.p, self.I_loc, self.I_mig, p_value = self.generate_action()

        self.I_loc = self.pre_process(I_loc=self.I_loc)

        done_task = self.process_cur()

        load_ratio, task_latency = self.process_post()

        result = [

            # action
            torch.tensor(np.array([self.I_loc, self.I_mig, self.p])),

            # states
            self.return_state(),

            # reward
            torch.tensor([load_ratio, task_latency]),

            # complete num
            done_task.get_size()

        ]

        return result, self.idx

    def return_state(self):
        return torch.concat([
                          torch.tensor(self.task_queue.get_size()),
                          torch.tensor(self.task_queue.return_mean()).view(-1),
                          torch.tensor(self.task_queue.return_std()).view(-1)],
            dim=0),

    # def act_share(self, new_task_queue: TaskQueue):
    #     self.tempoary_state_inf: TaskQueue = new_task_queue
    #
    #     self.tempoary_state_inf.set_recycle(self.re_direct_queue)  # set re_direct queue as the recycle station
    #
    #     self.p, self.I_loc, self.I_mig = self.generate_action()
    #
    #     self.I_loc = self.pre_process(I_loc=self.I_loc)
    #
    #     done_task = self.process_cur()
    #
    #     load_ratio, task_latency = self.process_post()
    #
    #     action = [self.I_loc, self.I_mig, self.p]
    #     states = [self.task_queue.get_size(),
    #         ] + self.task_queue.return_mean().tolist() + self.task_queue.return_std().tolist()
    #     reward = [load_ratio, task_latency]
    #
    #     action_share.put((action, self.idx))
    #     states_share.put((states, self.idx))
    #     reward_share.put((reward, self.idx))
    #
    #     self.memorandum.append((action, states, reward))
    #
    #     return action, states, reward

    def update_policy(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    cpu_cycles = [250, 200, 300]
    max_run_memorys = [100, 200, 500]
    arrive_nums = [10, 5, 3]
    data_nums = [50, 100, 300]
    device_nums = 1

    service_num = 3
    f = 10
    rho = 0.1
    tr = 15000
    dm = 1500
    adjacency = [[1, 2], [0, 2], [0, 1]]
    idx = 1

    agent = RandomAgent(service_nums=service_num, f=f, rho=rho, tr=tr, dm=dm, adjacency=adjacency, idx=idx)
    S = Services(cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys, arrive_nums=arrive_nums, data_nums=data_nums,
                 device_nums=device_nums, service_num=3)
    for i in range(3):
        agent.act(**S.generate_new_services(device_nums=1))
