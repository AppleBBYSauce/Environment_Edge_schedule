# Item's name: CouldEdgeSchedule
# Autor: bby
# DateL 2023/12/2 17:21
import torch
from Services import Services
from TaskQueue import TaskQueue
from ACAgentBatch import ACAgentBatch
from CIMRL import ChIMRL
from Buffer_all_step import Buffer
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

np.random.seed(0)

MAX_BUFFER = 200
N_WORKER = 1


# mp = mp.get_context("spawn")


class Env:

    def __init__(self, service_nums: int, f: list[int], rho: list[float], tr: list[int], dm: list[int], gamma: float,
                 cpu_cycles: list, max_run_memorys: list, arrive_nums: list, data_nums: list, device_nums: int,
                 lr_critic: float, lr_actor: float, state_nums: int, freedom_nums: list, hidden_nums: int, edge_index,
                 local_edge, batch_size, soft_update_weight, soft_update_step, buffer_size: int, sigma_gumbel=2,
                 w_cpu: float = 0.9, w_memory: float = 0.1, refresh_frequency: int = 30, sw: float = 30,
                 delta_t: int = 10, buffer_step: int = 5, start_record: int = None, DAT_period: int = 50):

        self.round_count = 0
        self.DAT_period = DAT_period
        self.DAT_flag = True
        self.service_nums = service_nums
        self.device_nums = device_nums
        self.delta_t = delta_t
        self.batch_size = batch_size
        self.soft_update_step = soft_update_step
        self.soft_update_weight = soft_update_weight
        self.service_generator = Services(cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys,
                                          arrive_nums=arrive_nums, data_nums=data_nums,
                                          device_nums=device_nums, service_nums=service_nums)

        self.nn = ChIMRL(agents_nums=device_nums, freedom_nums=freedom_nums, hidden_nums=hidden_nums, step=buffer_step,
                         local_edges=local_edge, gamma=gamma, state_nums=state_nums, edge_index=edge_index,
                         action_nums=sum(freedom_nums))

        self.Devices = ACAgentBatch(service_nums=service_nums, f=f, f_mean=np.median(f), rho=rho, tr=tr,
                                    dm=dm, cpu_cycles=cpu_cycles, avg_arrive_nums=arrive_nums,
                                    sw=sw, refresh_frequency=refresh_frequency, agents_num=device_nums)

        if start_record is None:
            self.start_record = int(np.median(np.max(self.Devices.tau, axis=1)))
        else:
            self.start_record = start_record

        self.re_direct_station = None
        self.total_task_nums = np.array([0 for _ in range(self.service_nums)])
        self.done_tasks_num = np.array([0 for _ in range(self.service_nums)])
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.opt_critic = torch.optim.Adam(params=self.nn.return_critic_update_parameters(), lr=self.lr_critic,
                                           weight_decay=1e-3)
        self.opt_actor_I = torch.optim.AdamW(params=self.nn.return_actor_update_parameters_I(), lr=self.lr_actor,
                                             weight_decay=1e-5)
        self.opt_actor_P = torch.optim.AdamW(params=self.nn.return_actor_update_parameters_P(), lr=self.lr_actor,
                                             weight_decay=1e-5)
        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_critic, T_max=100,
                                                                           eta_min=5e-5)
        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_actor_I, T_max=50,
                                                                          eta_min=5e-5)
        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_actor_P, T_max=50,
                                                                          eta_min=5e-5)

        # action, state, reward
        self.buffer = Buffer(step=buffer_step, buffer_size=buffer_size)
        self.round_reward = np.empty(shape=(1, 2))
        self.update_round = 0
        self.ultimate_record = []

        self.recorder = open("result.txt", "w+")

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
            loss_critic, loss_actor = self.update(self.update_round)
            self.round_count = 0
            self.update_round += 1
            f = (
                f"load ratio std: {self.round_reward[:, 0].std()}, task latency mean: {np.mean(self.round_reward[:, 1])},"
                f"complete ratio: {self.done_tasks_num / self.total_task_nums}, "
                f"critic loss: {loss_critic}, actor loss: {loss_actor}")
            self.recorder.write(f + "\n")
            print(f)
            self.round_reward = np.empty(shape=(1, 2))
            self.total_task_nums *= 0
            self.done_tasks_num *= 0
            # [i.clean() for i in self.Devices.task_queue]

        # clean the destructure station
        [d.clean() for d in self.Devices.destructure_queue]

        # re-direct tasks
        self.re_direct_station = self.re_direct()

    def re_direct(self):
        re_direct_queue = [TaskQueue(service_num=self.service_nums) for _ in range(self.device_nums)]

        for idx in range(self.device_nums):
            for l, (target, loc) in enumerate(zip(self.Devices.I_mig[idx], self.Devices.I_loc[idx])):
                if not loc:
                    re_direct_queue[target].merger_single_service(l=l,
                                                                  task_queue=
                                                                  self.Devices.re_direct_queue[idx].task_queue[
                                                                      l])

        [rdq.clean() for rdq in self.Devices.re_direct_queue]

        return re_direct_queue

    def process(self, new_tasks: list[TaskQueue]):

        observation = self.Devices.get_states()
        new_tasks_size = torch.stack([torch.tensor(i.queue_size) for i in new_tasks], dim=0)
        observation = torch.concat([observation, new_tasks_size], dim=1).unsqueeze(0).float()

        for rdq, nt in zip(self.Devices.re_direct_queue, new_tasks):
            nt.set_recycle(rdq)

        self.Devices.temporary_state_inf = new_tasks

        actions, action_probs, _, actions_constrain = self.nn.generate_act(observation=observation,
                                                                           task_nums=new_tasks_size, update=False)

        p_multiple, I_mig_multiple, I_loc_multiple = actions_constrain
        self.Devices.p = p_multiple
        p_multiple, I_mig_multiple, I_loc_multiple = p_multiple.numpy(), I_mig_multiple.numpy(), I_loc_multiple.numpy()
        self.Devices.I_mig = I_mig_multiple
        self.Devices.I_loc = I_loc_multiple
        SW, TR, CT = self.Devices.pre_process(I_mig_multiple=I_mig_multiple,
                                              p_multiple=p_multiple,
                                              I_loc_multiple=I_loc_multiple)
        done_task_queue, cpu_usage_ratio, memory_usage_ratio = self.Devices.process_cur(p_multiple=p_multiple)
        load_ratio, tasks_latency = self.Devices.process_post(SW=SW, TR=TR, CT=CT, cpu_usage_ratios=cpu_usage_ratio,
                                                              memory_usage_ratios=memory_usage_ratio,
                                                              p_multiple=p_multiple)

        return (actions, action_probs, observation,
                torch.stack([load_ratio, tasks_latency], dim=1),
                np.sum([dtq.get_size() for dtq in done_task_queue], axis=0))

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
            for target_type in range(len(num_nodes)):
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
        # adj[edge_index[1], edge_index[0]] = 1

        local_edges = []
        for i in range(total):
            local_edge_ = torch.tensor(np.nonzero(adj[i])[0])
            local_edge_ = local_edge_[local_edge_ != i]
            local_edge_ = torch.concat([torch.tensor([i]), local_edge_], dim=0).unsqueeze(0)
            local_edges.append(local_edge_.long())
        return adj, torch.LongTensor(np.array(edge_index)), local_edges

    def update(self, step):
        cmp_device = "cuda"
        loss_critic = []
        loss_actor = []
        self.nn.to(cmp_device)
        self.nn.train()
        scaler = GradScaler()
        # torch.autograd.set_detect_anomaly(True)
        for counter, data in enumerate(self.buffer.BufferLoader(self.batch_size)):
            actions, _, states, rewards = data
            action = actions.to(cmp_device).data
            reward = rewards.to(cmp_device)
            states = states.to(cmp_device)
            with autocast():
                A = self.nn.return_ChI_Q(states=states,
                                         reward=reward,
                                         action_old=action)
                loss = torch.nanmean(torch.square(A), dim=0).nansum()

            loss_critic.append(loss.item())
            self.opt_critic.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.opt_critic)
            scaler.update()

            if counter > 0 and counter % self.soft_update_step == 0:
                self.nn.soft_update(flag=False, soft_update_weight=self.soft_update_weight)
        self.nn.soft_update(flag=False, soft_update_weight=self.soft_update_weight)
        self.scheduler_critic.step()

        if step > 50:
            if step % self.DAT_period == 0:
                self.DAT_flag = not self.DAT_flag
            # update the actor
            scaler = GradScaler()
            for counter, data in enumerate(self.buffer.BufferLoader(self.batch_size)):
                actions, action_prob, states, rewards = data
                states = states.to(cmp_device)
                action = actions.to(cmp_device).data
                reward = rewards.to(cmp_device)
                state = states[:, 0, :, :]
                with autocast():
                    # action, _, _ = self.nn.generate_act(observation=state, update=True)
                    # loss,_ = self.nn.get_pre(observation=state, action=action, update=False)
                    # loss = torch.nanmean(loss, dim=0).nansum()
                    A = self.nn.return_ChI_Q(states=states,
                                             reward=reward,
                                             action_old=action,
                                             update_actor=True)
                    loss = torch.nanmean(-A, dim=0).nansum()

                loss_actor.append(loss.item())

                if self.DAT_flag:

                    self.opt_actor_P.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.opt_actor_P)
                    scaler.update()

                else:

                    self.opt_actor_I.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.opt_actor_I)
                    scaler.update()

                if counter > 0 and counter % self.soft_update_step == 0:
                    self.nn.soft_update(flag=True, soft_update_weight=self.soft_update_weight)
            self.nn.soft_update(flag=True, soft_update_weight=self.soft_update_weight)
            # self.scheduler_actor.step()
            # self.nn.actor_update.VAR *= self.nn.actor_update.temperature
            # self.nn.actor.VAR *= self.nn.actor.temperature
        else:
            loss_actor = 0.
        torch.cuda.empty_cache()
        self.nn.to("cpu")
        self.nn.eval()
        return np.mean(loss_critic), np.mean(loss_actor)


if __name__ == '__main__':
    all_time = 500000
    delta_t = 45
    refresh_frequency = 1
    sw = 1
    batch_size = 16
    gamma = 0.9
    soft_update_weight = 0.15
    soft_update_step = 20
    buffer_step = 4
    buffer_size = 150 + buffer_step
    hidden_nums = 128
    lr_critic = 0.01
    lr_actor = 0.005

    DAT_period = 3

    cpu_cycles = [300, 500, 1000]
    max_run_memorys = [2 ** 10, 2 ** 10 * 2, 2 ** 10 * 3]
    arrive_nums = [45, 30, 20]
    arrive_nums = [i * (refresh_frequency / delta_t) for i in arrive_nums]
    data_nums = [50, 150, 300]
    service_nums = 3

    device_nums_level_1 = 3
    device_nums_level_2 = 3
    device_nums_level_3 = 3
    # link_nums = [[0, 1, 1],
    #              [0, 1, 1],
    #              [0, 0, 1]]

    link_nums = [[2, 3, 1],
                 [3, 0, 1],
                 [3, 0, 0]]

    num_nodes = [device_nums_level_1, device_nums_level_2, device_nums_level_3]
    adj, edge_index, local_edge = Env.graph_generate(num_nodes=num_nodes, num_links=link_nums)

    device_nums = device_nums_level_1 + device_nums_level_2 + device_nums_level_3
    f = [1000 for _ in range(device_nums_level_1)] + [5000 for _ in range(device_nums_level_2)] + [10000 for _ in range(
        device_nums_level_3)]


    def zero_one(x):
        max_x = np.max(x)
        min_x = np.min(x)
        if max_x == min_x:
            return np.ones_like(x)
        else:
            return (x - min_x) / (max_x - min_x)


    def softmax(x):
        exp_x = np.exp(zero_one(x))
        return exp_x / np.sum(exp_x)


    f_total = np.sum(f) * softmax(np.array(cpu_cycles))
    sojourn_time_ideal = 1 / ((f_total / np.array(cpu_cycles)) - np.array(arrive_nums))
    print(sojourn_time_ideal)

    rho = [0.1 for _ in range(device_nums)]
    tr = [np.random.normal(loc=15000, scale=5000) for _ in range(device_nums)]
    dm = [2 ** 10 * 4 for _ in range(device_nums_level_1)] + [2 ** 10 * 8 for _ in range(device_nums_level_2)] + [
        2 ** 10 * 16 for _ in range(device_nums_level_3)]

    e = Env(service_nums=service_nums, f=f, rho=rho, tr=tr, dm=dm, gamma=gamma,
            cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys, arrive_nums=arrive_nums, data_nums=data_nums,
            device_nums=device_nums, sw=sw, refresh_frequency=refresh_frequency, delta_t=delta_t,
            lr_critic=lr_critic, lr_actor=lr_actor,
            state_nums=20 * service_nums, freedom_nums=[service_nums for _ in range(3)], hidden_nums=hidden_nums,
            soft_update_step=soft_update_step, soft_update_weight=soft_update_weight,
            edge_index=edge_index, local_edge=local_edge, batch_size=batch_size, buffer_step=buffer_step,
            buffer_size=buffer_size, DAT_period=DAT_period)

    import time

    start = time.time()
    for i in range(all_time):
        e.step()
    print(time.time() - start)
    e.recorder.close()
