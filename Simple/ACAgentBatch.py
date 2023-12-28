# Item's name: ACAgent
# Autor: bby
# DateL 2023/12/3 18:27
# Item's name: BaseAgent
# Autor: bby
# DateL 2023/12/3 18:26

from BaseAgent import BaseAgent
from Services import Services
from TaskQueue import TaskQueue
from abc import abstractmethod
from collections import deque
import numpy as np
import _heapq
import heapq
import torch
import copy

np.random.seed(0)


def softmax(x):
    exp_x = np.exp(zero_one(x))
    return exp_x / np.sum(exp_x)


def zero_one(x):
    max_x = np.max(x)
    min_x = np.min(x)
    if max_x == min_x:
        return np.ones_like(x)
    else:
        return (x - min_x) / (max_x - min_x)


class ACAgentBatch:

    def __init__(self, service_nums: int, f: list[int], f_mean: int, rho: list[float], tr: list[int], dm: list[int],
                 cpu_cycles: list, avg_arrive_nums, agents_num: int, w_cpu: float = 0.5, w_memory: float = 0.5,
                 sw: float = 30, refresh_frequency: int = 1, task_queue_size: list[int] = None):

        self.service_nums = service_nums
        self.rho = rho
        self.f = np.expand_dims(np.array(f), axis=-1)
        self.tr = tr
        self.dm = dm
        self.refresh_frequency = refresh_frequency
        self.agent_nums = agents_num
        self.w_cpu = w_cpu
        self.w_memory = w_memory
        self.sw = sw
        self.cpu_cycles = np.array(cpu_cycles)

        self.p = torch.ones(size=(agents_num, service_nums)) / service_nums
        self.I_mig = None

        # calculate the expiry time
        self.tau = np.ceil((self.cpu_cycles) / (self.f / self.service_nums**2)) + 1

        # calculate the maximum queue size of each service
        if task_queue_size is None:
            self.task_queue_size = np.floor((self.f * self.tau) / self.cpu_cycles)
        else:
            self.task_queue_size = task_queue_size

        # destructure queue
        self.destructure_queue = [TaskQueue(service_num=service_nums) for _ in range(self.agent_nums)]

        # re-direct queue
        self.re_direct_queue = [TaskQueue(service_num=service_nums, recycle=dq) for dq in self.destructure_queue]

        # task queue
        self.task_queue = [TaskQueue(service_num=service_nums,
                                     queue_size_max=tqs,
                                     recycle=rdq) for tqs, rdq in zip(self.task_queue_size, self.re_direct_queue)]

        self.temporary_state_inf = None
        self.arrive_ratio = avg_arrive_nums

    def get_states(self):
        states = []
        for idx, (tq, rdq) in enumerate(zip(self.task_queue, self.re_direct_queue)):
            states.append(
                torch.concat([
                    torch.tensor(tq.get_size()),
                    torch.tensor(rdq.get_size()),
                    self.p[idx],
                    torch.tensor(tq.return_mean()).view(-1),
                    torch.tensor(tq.return_std()).view(-1),
                ])
            )
        return torch.stack(states, dim=0)

    def process_post(self, SW, TR, CT, cpu_usage_ratios, memory_usage_ratios, p_multiple):

        destructure_task_sizes = []

        # add the life time for each task
        for agents in range(self.agent_nums):
            self.task_queue[agents].op(func=lambda x: x + 1, key="rd")
            self.re_direct_queue[agents].op(func=lambda x: x + 1, key="rd")
            destructure_task_sizes.append(self.destructure_queue[agents].get_size())

        destructure_task_sizes = np.stack(destructure_task_sizes, axis=0)
        # calculate the load ratio
        phi = self.w_cpu * cpu_usage_ratios + self.w_memory * memory_usage_ratios

        # calculate the penalty of overdue task
        P = self.tau * destructure_task_sizes
        P = np.nansum(P, axis=1)

        # calculate the task latency
        psi = SW + TR + CT + P

        return torch.tensor(phi), torch.tensor(psi)

    def process_cur(self, p_multiple):

        done_task_queues = []
        cpu_usage_ratios = []
        memory_usage_ratios = []
        for agent in range(self.agent_nums):
            cpu_usage_ratio = np.array([0 for _ in range(self.service_nums)])
            memory_usage_ratio = np.array([0 for _ in range(self.service_nums)])

            ### process current task ###
            task_size = self.task_queue[agent].get_size()
            processed_num = np.array([0 for _ in range(self.service_nums)])
            remain_time = self.refresh_frequency
            select_mask = np.array([1 if i > 0 else np.inf for i in task_size])
            rms = self.task_queue[agent].get_key_data(key="rm")
            cps = self.task_queue[agent].get_key_data(key="cp")

            done_tk = [False for _ in range(self.service_nums)]
            cur_rm = np.array([rm[i - 1] if i > 0 else 0 for rm, i in zip(rms, task_size)])
            cur_cp = np.array([cp[i - 1] if i > 0 else np.inf for cp, i in zip(cps, task_size)])
            cp_l = p_multiple[agent] * self.f[agent, 0]
            max_rm = self.dm[agent]
            while remain_time > 0 and sum(task_size) > 0:

                while sum(cur_rm) > max_rm:
                    pos = np.argmax(cur_rm)
                    cur_rm[pos] = 0

                if sum(cur_rm) == 0:
                    break

                rm_mask = cur_rm > 0
                min_time_consume = min(np.min((cur_cp / cp_l) * select_mask), remain_time)
                remain_time -= min_time_consume
                done_cp = cur_cp - cp_l * min_time_consume
                done_tk = done_cp <= 1
                cur_cp[rm_mask] = done_cp[rm_mask]
                task_size[done_tk] -= 1
                processed_num[done_tk] += 1

                cur_cp[done_tk] = np.array([cp[i - 1] if i > 0 else np.inf for cp, i in zip(cps, task_size)])[done_tk]
                cur_rm[done_tk] = np.array([rm[i - 1] if i > 0 else 0 for rm, i in zip(rms, task_size)])[done_tk]

                cpu_usage_ratio = cpu_usage_ratio + min_time_consume * rm_mask  # calculate the cpu usage ratio
                memory_usage_ratio = np.where(memory_usage_ratio < cur_rm, cur_rm,
                                              memory_usage_ratio)  # calculate the memory usage ratio

            # record the ongoing task
            [self.task_queue[agent].set_element_inf(l=i, idx=j - 1, key="cp", value=cur_cp[i])
             for i, j in enumerate(task_size) if j > 0]

            done = self.task_queue[agent].drop_task_(number=processed_num,
                                                     reverse=True,
                                                     sort=False,
                                                     enable_recycle=False)

            cpu_usage_ratio = np.sum((p_multiple[agent] * (cpu_usage_ratio / self.refresh_frequency)))
            cpu_usage_ratios.append(cpu_usage_ratio)
            memory_usage_ratios.append(np.sum(memory_usage_ratio) / self.dm[agent])
            done_task_queues.append(done)
        return done_task_queues, np.array(cpu_usage_ratios), np.array(memory_usage_ratios)

    def pre_process(self,
                    I_mig_multiple,  # (M L)
                    p_multiple  # (M L)
                    ):

        task_nums = []
        # delete overdue tasks
        for idx, (task_queue, temporary_state_inf, re_direct_queue) in enumerate(
                zip(self.task_queue, self.temporary_state_inf, self.re_direct_queue)):
            task_queue.drop_with_key(func=lambda x, y: x > y, key="rd", y=self.tau[idx] / self.refresh_frequency)
            temporary_state_inf.drop_with_key(func=lambda x, y: x > y, key="rd",
                                              y=self.tau[idx] / self.refresh_frequency)
            re_direct_queue.drop_with_key(func=lambda x, y: x > y, key="rd", y=self.tau[idx] / self.refresh_frequency)

        # add unselected tasks into redirect queue
        for idx, (I_mig, temporary_state_inf, task_queue) in enumerate(zip(I_mig_multiple, self.temporary_state_inf, self.task_queue)):
            drop_nums = temporary_state_inf.get_size() * (I_mig != idx)
            temporary_state_inf.drop_task_(number=drop_nums, reverse=True, key="rd", sort=True)

        # merger new task in process queue
        for task_queue, temporary_state_inf in zip(self.task_queue, self.temporary_state_inf):
            task_queue.merger_task_queue(temporary_state_inf, drop=True)
            temporary_state_inf.clean()
            task_nums.append(task_queue.get_size())

        # calculate the task latency #
        task_nums = np.stack(task_nums, axis=0)
        # calculate the switch latency
        SW = np.array(
            [np.sum(re_direct_queue.return_sum(key="rd")) * self.sw for re_direct_queue in self.re_direct_queue])

        # calculate the transmission latency
        TR = np.array(
            [np.sum(task_queue.return_sum(key="dl") / self.tr[idx]) for idx, task_queue in enumerate(self.task_queue)])

        # calculate the calculation latency
        # arrive_ratio = new_task_nums / self.refresh_frequency
        # service_ratio = (self.f * p_multiple) / (self.cpu_cycles)
        # sojourn_time = service_ratio - arrive_ratio



        # if service ratio below the arriving ratio, we use the overdue time to substitute sojourn time
        # sojourn_time = np.where(sojourn_time > 0, service_ratio, 1 / (3 * self.tau))
        # CT = np.sum(1 / sojourn_time, axis=1)

        mask_span_time = np.where(task_nums > 0, (self.cpu_cycles * task_nums) / (self.f * p_multiple),  self.tau * task_nums).sum(1)

        return SW, TR, mask_span_time

    def return_states(self):
        return


if __name__ == '__main__':
    pass
    # service_nums = 2
    # f = [100, 200, 300]
    # f_mean = 200
    # rho = [0.1, 0.1, 0.1]
    # tr = [10, 10, 10]
    # dm = [100, 100, 100]
    #
    # cpu_cycles = [100, 500]
    # avg_arrive_nums = [10, 5]
    # agent_nums = 3
    #
    # A = ACAgentBatch(
    #     service_nums=service_nums,
    #     f=f,
    #     f_mean=f_mean,
    #     rho=rho,
    #     tr=tr,
    #     dm=dm,
    #     cpu_cycles=cpu_cycles,
    #     avg_arrive_nums=avg_arrive_nums,
    #     agents_num=agent_nums
    # )
