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


class ACAgent(BaseAgent):

    def __init__(self, service_nums: int, f: int, f_mean:int, rho: float, tr: int, dm: int, local_edge:list,
                 idx: int, cpu_cycles,avg_arrive_nums, max_run_memorys, w_cpu: float = 0.5,
                 w_memory: float = 0.5, sw: float = 30, refresh_frequency: int = 30, task_queue_size:list[int]=None,
                 memory_size:int=3):
        self.service_nums = service_nums
        self.rho = rho
        self.f = f
        self.tr = tr
        self.dm = dm
        self.remain_dm = dm
        self.idx = idx
        self.p = torch.tensor([0] * service_nums).int()
        self.I_loc = None
        self.I_mig = None
        self.refresh_frequency = refresh_frequency
        self.local_edge = local_edge
        self.w_cpu = w_cpu
        self.w_memory = w_memory
        self.sw = sw
        self.load_index = [0, 0]
        self.task_latency = [0, 0, 0, 0]

        # calculate the expiry time
        self.tau = (np.ceil(cpu_cycles / (((f / self.service_nums) / self.service_nums) * refresh_frequency)) * refresh_frequency)

        # calculate the maxmium queue size of each service
        if task_queue_size is None:
            self.task_queue_size = np.floor((f * self.tau) / cpu_cycles)
        else:
            self.task_queue_size = task_queue_size

        # destructure queue
        self.destructure_queue = TaskQueue(service_num=service_nums)

        # re-direct queue
        self.re_direct_queue = TaskQueue(service_num=service_nums, recycle=self.destructure_queue)

        # task queue
        self.task_queue = TaskQueue(service_num=service_nums, queue_size_max=self.task_queue_size, recycle=self.re_direct_queue)

        self.tempoary_state_inf = None

        self.cpu_cycles = cpu_cycles
        self.arrive_num = avg_arrive_nums
        self.max_run_memorys = max_run_memorys

    def process_post(self):

        # add the life time for each task
        self.task_queue.op(func=lambda x:x+1, key="rd")
        self.re_direct_queue.op(func=lambda x:x+1, key="rd")

        # calculate the load ratio
        phi = self.w_cpu * self.load_index[0] + self.w_memory * self.load_index[1]

        # calculate the penalty of overdue task
        P = ((self.f * self.p) / (self.cpu_cycles)) * np.array(self.destructure_queue.get_size())
        P = np.nansum(P)
        self.task_latency[3] = P

        # calculate the task latency
        psi = np.sum(self.task_latency)

        return phi, psi

    def process_cur(self):

        p = self.p.numpy()

        cpu_useage_ratio = np.array([0 for _ in range(self.service_nums)])
        memory_usage_ratio = np.array([0 for _ in range(self.service_nums)])

        ### process current task ###
        task_size = self.task_queue.get_size()
        processed_num = np.array([0 for _ in range(self.service_nums)])
        remain_time = self.refresh_frequency
        select_mask = np.array([1 if i > 0 else np.inf for i in task_size])
        rms = self.task_queue.get_key_data(key="rm")
        cps = self.task_queue.get_key_data(key="cp")

        done_tk = [False for _ in range(self.service_nums)]
        cur_rm = np.array([rm[i - 1] if i > 0 else 0 for rm, i in zip(rms, task_size)])
        cur_cp = np.array([cp[i - 1] if i > 0 else np.inf for cp, i in zip(cps, task_size)])
        cp_l = p * self.f
        max_rm = self.dm
        while remain_time > 0 and sum(task_size) > 0:

            while sum(cur_rm) > max_rm:
                pos = np.argmax(cur_rm)
                cur_rm[pos] = 0

            if sum(cur_rm) == 0:
                break

            rm_mask = cur_rm > 0

            min_time_consum = min(np.min((cur_cp / cp_l) * select_mask), remain_time)
            remain_time -= min_time_consum
            done_cp = cur_cp - cp_l * min_time_consum
            done_tk = done_cp <= 1
            cur_cp[rm_mask] = done_cp[rm_mask]
            task_size[done_tk] -= 1
            processed_num[done_tk] += 1

            cur_cp[done_tk] = np.array([cp[i - 1] if i > 0 else np.inf for cp, i in zip(cps, task_size)])[done_tk]
            cur_rm[done_tk] = np.array([rm[i - 1] if i > 0 else 0 for rm, i in zip(rms, task_size)])[done_tk]

            cpu_useage_ratio = cpu_useage_ratio + min_time_consum * rm_mask # calculate the cpu useage ratio
            memory_usage_ratio = np.where(memory_usage_ratio < cur_rm, cur_rm, memory_usage_ratio) # calculate the memory useage ratio

        # record the on-going task
        [self.task_queue.set_element_inf(l=i, idx=j-1, key="cp", value=cur_cp[i]) for i, j in enumerate(task_size) if j > 0]
        done = self.task_queue.drop_task_(number=processed_num, reverse=True, sort=False, enable_recycle=False)
        cpu_useage_ratio = np.mean((p * (cpu_useage_ratio / self.refresh_frequency)))
        self.load_index[0] = cpu_useage_ratio
        self.load_index[1] = np.sum(memory_usage_ratio) / self.dm
        return done

    def pre_process(self, *args, **kwargs):


        I_loc = self.I_loc.numpy()
        p = self.p.numpy()
        # delete overdue tasks
        self.task_queue.drop_with_key(func=lambda x, y: x > y, key="rd", y=self.tau / self.refresh_frequency)
        self.tempoary_state_inf.drop_with_key(func=lambda x, y: x > y, key="rd", y=self.tau / self.refresh_frequency)
        self.re_direct_queue.drop_with_key(func=lambda x, y: x > y, key="rd", y=self.tau / self.refresh_frequency)

        # add unselected tasks into redirect queue
        drop_nums = self.tempoary_state_inf.get_size() - I_loc
        self.tempoary_state_inf.drop_task_(number=drop_nums, reverse=True, key="rd", sort=True)

        # merger new task in process queue

        self.task_queue.merger_task_queue(self.tempoary_state_inf, drop=True)
        self.tempoary_state_inf.clean()

        ## calculate the task latency ##

        # calculate the switch latency
        SW = np.sum(self.task_queue.return_mean(key="rd")) * self.sw

        # calculate the transmission latency
        TR = np.sum(self.task_queue.return_sum(key="dl") / self.tr)

        # calculate the calculation latency
        arrive_ratio = np.array(I_loc) / self.refresh_frequency
        service_ratio = ((self.f * p) / self.cpu_cycles)
        sojourn_time = service_ratio - arrive_ratio

        # if service ratio below the arrive ratio, we use the overdue time to substitude sojourn time
        sojourn_time = np.where(sojourn_time > 0, service_ratio,  1 / self.tau)[arrive_ratio > 0]
        CT = np.sum(1 / sojourn_time)

        self.task_latency[0] = SW
        self.task_latency[1] = TR
        self.task_latency[2] = CT



if __name__ == '__main__':
    pass
