# Item's name: CouldEdgeSchedule
# Autor: bby
# DateL 2023/12/2 17:21
import torch
from Services import Services
from TaskQueue import TaskQueue
from MDDPGAgent import MDDPGAgent
from Buffer import Buffer
import numpy as np
from torch.cuda.amp import GradScaler

np.random.seed(0)

MAX_BUFFER = 200
N_WORKER = 1


# mp = mp.get_context("spawn")


class Env:

    def __init__(self, service_nums: int, f: list[int], rho: list[float], tr: list[int], dm: list[int], gamma: float,
                 cpu_cycles: list, max_run_memorys: list, arrive_nums: list, data_nums: list, device_nums: int,
                 lr: float,
                 state_nums: int, freedom_nums: list, hidden_nums: int, edge_index, local_edge, batch_size,
                 soft_update_weight, soft_update_step, buffer_size: int,
                 sigma_gumbel=2, w_cpu: float = 0.9, w_memory: float = 0.1, refresh_frequency: int = 30, sw: float = 30,
                 delta_t: int = 10, buffer_step: int = 5, start_record: int = None):

        self.round_count = 0
        self.service_nums = service_nums
        self.device_nums = device_nums
        self.delta_t = delta_t
        self.batch_size = batch_size
        self.soft_update_step = soft_update_step
        self.soft_update_weight = soft_update_weight
        self.service_generator = Services(cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys,
                                          arrive_nums=arrive_nums, data_nums=data_nums,
                                          device_nums=device_nums, service_nums=service_nums)

        self.Devices = [MDDPGAgent(service_nums=service_nums, f=f[idx], f_mean=np.median(f), rho=rho[idx], tr=tr[idx],
                                   dm=dm[idx], local_edge=local_edge[idx], idx=idx, cpu_cycles=np.array(cpu_cycles),
                                   avg_arrive_nums=np.array(arrive_nums), step=buffer_step, lr=lr,
                                   max_run_memorys=max_run_memorys, sw=sw, refresh_frequency=refresh_frequency,
                                   state_nums=state_nums, freedom_nums=freedom_nums, hidden_nums=hidden_nums,
                                   agents_nums=device_nums, gamma=gamma, soft_upate_weight=soft_update_weight,
                                   soft_update_step=soft_update_step)
                        for idx in range(device_nums)]

        if start_record is None:
            self.start_record = int(np.min([max(i.tau) for i in self.Devices]))
        else:
            self.start_record = start_record

        self.re_direct_station = None
        self.total_task_nums = np.array([0 for _ in range(self.service_nums)])
        self.done_tasks_num = np.array([0 for _ in range(self.service_nums)])

        # action, state, reward
        self.buffer = Buffer(step=buffer_step, buffer_size=buffer_size)
        self.round_reward = np.empty(shape=(1, 2))
        self.ultimate_record = []

    def step(self):

        self.round_count += 1

        # generate new task for each device
        new_tasks = self.service_generator.generate_new_services()

        self.total_task_nums += np.sum([i.queue_size for i in new_tasks], axis=0)

        # merger the redirect tasks
        if self.re_direct_station is not None:
            for t_new, t_rd in zip(new_tasks, self.re_direct_station):
                t_new.merger_task_queue(t_rd)

        # process task in corresponding device
        actions, actions_prob, states, rewards, done_tasks_num = self.process(new_tasks)

        if self.round_count >= self.start_record:
            self.buffer.add_data(action=actions, state=states, reward=rewards, action_prob=actions_prob)
            self.round_reward = np.concatenate([self.round_reward, np.array(rewards)], axis=0)

        self.done_tasks_num += done_tasks_num

        if self.round_count - self.start_record > self.delta_t:
            self.round_count = 0
            print(
                f"load ratio std: {self.round_reward[:, 0].std()}, task latency avg: {self.round_reward[:, 1].sum()}, done ratio: {self.done_tasks_num / self.total_task_nums},")
            self.round_reward = np.empty(shape=(1, 2))
            self.total_task_nums = np.array([0 for _ in range(self.service_nums)])
            self.done_tasks_num = np.array([0 for _ in range(self.service_nums)])
            self.update()
            [i.task_queue.clean() for i in self.Devices]

        # clean the destructure station
        [d.destructure_queue.clean() for d in self.Devices]

        # re-direct tasks
        self.re_direct_station = self.re_direct()

    def re_direct(self):
        re_direct_queue = [TaskQueue(service_num=self.service_nums) for _ in range(self.device_nums)]

        for d in self.Devices:
            for l, target in enumerate(d.I_mig):
                re_direct_queue[target].merger_single_service(l=l, task_queue=d.re_direct_queue.task_queue[l])

        [d.re_direct_queue.clean() for d in self.Devices]

        return re_direct_queue

    def process(self, new_tasks):

        observation = self.get_states()
        new_tasks_size = torch.stack([torch.tensor(i.queue_size) for i in new_tasks], dim=0)
        observation = torch.concat([observation, new_tasks_size], dim=1).unsqueeze(0).float()
        action, action_prob, state, reward, done_task_nums = zip(
            *[d.act(s, observation[:, idx, :])[0] for idx, (d, s) in enumerate(zip(self.Devices, new_tasks))])
        return torch.stack(action, dim=0),torch.stack(action_prob, dim=0) ,torch.stack(state, dim=0), torch.stack(
            reward, dim=0), np.sum(done_task_nums, axis=0)

    def get_states(self):
        return torch.stack([agent.return_states() for agent in self.Devices], dim=0)

    @staticmethod
    def graph_generate(num_nodes: list, num_links: list):
        total = np.sum(num_nodes)
        idx = []
        for type in range(len(num_nodes)):
            if type == 0:
                start = 0
            else:
                start = idx[type - 1][1]
            idx.append([start, start + num_nodes[type]])
        adj = np.eye(total, total)
        edge_index = [[i for i in range(total)], [i for i in range(total)]]
        for source_type in range(len(num_nodes)):
            for target_type in range(source_type, len(num_nodes)):
                target_idx = np.arange(start=idx[target_type][0], stop=idx[target_type][1], step=1)
                for s in range(idx[source_type][0], idx[source_type][1]):
                    if source_type == target_type:
                        target_idx_ = target_idx[target_idx != s]
                    else:
                        target_idx_ = target_idx
                    choices = np.random.choice(target_idx_, size=(num_links[source_type][target_type]), replace=False)
                    edge_index[0] += [s for _ in range(len(choices))]
                    edge_index[1] += list(choices)

        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        local_edges = []
        for i in range(total):
            local_edge_ = torch.tensor(np.nonzero(adj[i])[0])
            local_edge_ = local_edge_[local_edge_ != i]
            local_edge_ = torch.concat([torch.tensor([i]), local_edge_], dim=0).unsqueeze(0)
            local_edges.append(local_edge_)
        return adj, torch.LongTensor(np.array(edge_index)), local_edges

    def update(self):
        cmp_device = "cuda"
        data_s = [[] for _ in range(len(self.Devices))]
        # self.Devices = [i.MDDPGS.to(cmp_device) for i in self.Devices]
        for counter, data in enumerate(self.buffer.BufferLoader(self.batch_size)):
            actions, action_probs, states, rewards = data
            for d in self.Devices:
                data_s[d.idx].append(
                    [actions[:, :, d.local_edge[0], :].to(cmp_device),  # neighbor action
                     action_probs[:, :, d.idx].to(cmp_device),  # action probability
                     states[:, :, d.local_edge[0], :].to(cmp_device),  # neighbor state
                     rewards[:, :, d.idx].to(cmp_device),  # self reward
                     states[:, 0, d.idx, -self.service_nums:].to(cmp_device),  # task nums
                     states[:, 0, d.idx, :].to(cmp_device)]  # self state
                )

        # update the critic
        TD_ERROR = [[] for _ in range(len(self.Devices))]
        for device in self.Devices:
            device.MDDPGS = device.MDDPGS.to(cmp_device)
            TD_ERROR[device.idx].extend(device.update_critic(data_s[device.idx]))

        # update the actor
        device_optimizers = [[i, torch.optim.AdamW(params=i.MDDPGS.actor.parameters(), lr=i.lr)] for i in self.Devices]
        scaler = GradScaler()
        for data_each in range(len(data_s[0])):
            for device, opt in device_optimizers:

                action_t = []
                for i in device.local_edge[0].tolist():

                    action_i, _ = self.Devices[i].generate_action(observation=data_s[i][data_each][5],
                                                             task_nums=data_s[i][data_each][4], update=False)
                    action_t.append(torch.concat(action_i, dim=-1))

                action_t = torch.stack(action_t, dim=1)
                loss = device.actor_loss(action_t=action_t,
                                         state_t=data_s[device.idx][data_each][2][:, 0, :, :])

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                if data_each % self.soft_update_step == 0:
                    for target_parameter, main_parameter in zip(device.MDDPGS.actor.parameters(),
                                                                device.MDDPGS.actor_update.parameters()):
                        target_parameter.data.copy_(
                            (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)
                        if torch.isnan(target_parameter).any():
                            print("error")

        torch.cuda.empty_cache()
        for i in self.Devices:
            i.MDDPGS.to("cpu")
            i.VAR *= i.temperature


if __name__ == '__main__':

    all_time = 20000
    delta_t = 15
    refresh_frequency = 1
    sw = 20
    batch_size = 64
    gamma = 0.9
    soft_update_weight = 0.1
    soft_update_step = 30
    buffer_step = 3
    buffer_size = 200
    hidden_nums = 64
    lr = 0.005

    cpu_cycles = [300, 1000]
    max_run_memorys = [2 ** 10, 2 ** 10 * 2]
    arrive_nums = [6, 2]
    arrive_nums = [i * (refresh_frequency / delta_t) for i in arrive_nums]
    data_nums = [50, 300]
    service_nums = 2

    device_nums_level_1 = 5
    device_nums_level_2 = 5
    device_nums_level_3 = 5
    link_nums = [[0, 1, 3],
                 [3, 1, 3],
                 [1, 2, 2]]

    num_nodes = [device_nums_level_1, device_nums_level_2, device_nums_level_3]
    adj, edge_index, local_edge = Env.graph_generate(num_nodes=num_nodes, num_links=link_nums)

    device_nums = device_nums_level_1 + device_nums_level_2 + device_nums_level_3
    f = [1000 for _ in range(device_nums_level_1)] + [5000 for _ in range(device_nums_level_2)] + [10000 for _ in range(
        device_nums_level_3)]
    total_service = (np.sum(f) / 3) / cpu_cycles

    rho = [0.1 for _ in range(device_nums)]
    tr = [np.random.normal(loc=15000, scale=5000) for _ in range(device_nums)]
    dm = [2 ** 10 * 2 for _ in range(device_nums_level_1)] + [2 ** 10 * 4 for _ in range(device_nums_level_2)] + [
        2 ** 10 * 8 for _ in range(device_nums_level_3)]

    e = Env(service_nums=service_nums, f=f, rho=rho, tr=tr, dm=dm, gamma=gamma,
            cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys, arrive_nums=arrive_nums, data_nums=data_nums,
            device_nums=device_nums, sw=sw, refresh_frequency=refresh_frequency, delta_t=delta_t, lr=lr,
            state_nums=11 * service_nums, freedom_nums=[service_nums for _ in range(3)], hidden_nums=hidden_nums,
            soft_update_step=soft_update_step, soft_update_weight=soft_update_weight,
            edge_index=edge_index, local_edge=local_edge, batch_size=batch_size, buffer_step=buffer_step,
            buffer_size=buffer_size)

    import time

    start = time.time()
    for _ in range(all_time):
        e.step()
    print(time.time() - start)
