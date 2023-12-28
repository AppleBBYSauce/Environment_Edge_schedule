# Item's name: CouldEdgeSchedule
# Autor: bby
# DateL 2023/12/2 17:21
import torch
from Services import Services
from TaskQueue import TaskQueue
from ACAgentBatch import ACAgentBatch
from MFAC import MFAC
from Buffer import Buffer
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch_scatter import scatter_add

np.random.seed(0)

MAX_BUFFER = 200
N_WORKER = 1


# mp = mp.get_context("spawn")


class Env:

    def __init__(self, service_nums: int, f: list[int], rho: list[float], tr: list[int], dm: list[int], gamma: float,
                 cpu_cycles: list, max_run_memorys: list, arrive_nums: list, data_nums: list, device_nums: int,
                 lr: float, state_nums: int, freedom_nums: list, hidden_nums: int, edge_index, local_edge, batch_size,
                 soft_update_weight, soft_update_step, buffer_size: int, alpha: float, sigma_gumbel=2,
                 w_cpu: float = 0.9,
                 w_memory: float = 0.1, refresh_frequency: int = 30, sw: float = 30, delta_t: int = 10,
                 buffer_step: int = 5, start_record: int = None):

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

        self.nn = MFAC(agents_nums=device_nums, freedom_nums=freedom_nums, hidden_nums=hidden_nums, step=buffer_step,
                       local_edges=local_edge, gamma=gamma, state_nums=state_nums, edge_index=edge_index, alpha=alpha)

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
        self.lr = lr

        # action, state, reward
        self.buffer = Buffer(step=buffer_step, buffer_size=buffer_size)
        self.round_reward = np.empty(shape=(1, 2))
        self.update_round = 0
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
            loss_critic, loss_actor = self.update(self.update_round)
            self.round_count = 0
            self.update_round += 1
            print(
                f"load ratio std: {self.round_reward[:, 0].std()}, task latency mean: {self.round_reward[:, 1].mean()}, "
                f"complete ratio: {self.done_tasks_num / self.total_task_nums}, critic loss: {loss_critic}, actor loss: {loss_actor}")
            self.round_reward = np.empty(shape=(1, 2))
            self.total_task_nums = np.array([0 for _ in range(self.service_nums)])
            self.done_tasks_num = np.array([0 for _ in range(self.service_nums)])
            [i.clean() for i in self.Devices.task_queue]

        # clean the destructure station
        [d.clean() for d in self.Devices.destructure_queue]

        # re-direct tasks
        self.re_direct_station = self.re_direct()

    def re_direct(self):
        re_direct_queue = [TaskQueue(service_num=self.service_nums) for _ in range(self.device_nums)]

        for idx in range(self.device_nums):
            for l, target in enumerate(self.Devices.I_mig[idx]):
                re_direct_queue[target].merger_single_service(l=l,
                                                              task_queue=self.Devices.re_direct_queue[idx].task_queue[
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

        actions, action_probs, actions_constrain = self.nn.generate_act(observation=observation,
                                                                        task_nums=new_tasks_size, update=False)

        p_multiple, I_loc_multiple, I_mig_multiple = actions_constrain
        self.Devices.p = p_multiple
        p_multiple, I_loc_multiple, I_mig_multiple = (p_multiple.numpy(),
                                                      I_loc_multiple.numpy(),
                                                      I_mig_multiple.numpy())
        self.Devices.I_mig = I_mig_multiple
        SW, TR, CT = self.Devices.pre_process(I_loc_multiple=I_loc_multiple, p_multiple=p_multiple)
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
            local_edges.append(local_edge_.long())
        return adj, torch.LongTensor(np.array(edge_index)), local_edges

    def update(self, step):
        cmp_device = "cuda"
        data_s = []
        loss_critic = []
        loss_actor = []
        self.nn.to(cmp_device)
        self.nn.train()
        opt = torch.optim.AdamW(params=self.nn.critic_update.parameters(), lr=self.lr)
        scaler = GradScaler()
        # torch.autograd.set_detect_anomaly(True)
        for counter, data in enumerate(self.buffer.BufferLoader(self.batch_size)):
            actions, action_probs, states, rewards = data

            action_t = actions[:, 0, :, :].to(cmp_device)
            action_probs_t = action_probs[:, 0, :].to(cmp_device).data
            reward = rewards[:, :-1, :].to(cmp_device)
            states_t = states[:, 0, :, :].to(cmp_device)
            states_t_plus = states[:, 1, :, :].to(cmp_device)
            data_s.append(states_t_plus)
            with autocast():
                Q_t, Q_t_plus = self.nn.return_mean_field_Q(states_t=states_t,
                                                            states_t_plus=states_t_plus,
                                                            reward=reward,
                                                            action_t=action_t,
                                                            action_probs_t=action_probs_t)
                loss = torch.abs(Q_t_plus - Q_t).mean(dim=0).sum()

            loss_critic.append(loss.item())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if counter % self.soft_update_step == 0:
                for target_parameter, main_parameter in zip(self.nn.critic.parameters(),
                                                            self.nn.critic_update.parameters()):
                    target_parameter.data.copy_(
                        (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)

        if step > 500:
            # update the actor
            opt = torch.optim.AdamW(params=self.nn.actor_update.parameters(), lr=self.lr)
            scaler = GradScaler()
            for counter, data in enumerate(self.buffer.BufferLoader(self.batch_size)):
                _, action_probs, states, rewards = data
                # action_t = actions[:, 0, :, :].to(cmp_device)
                action_probs_t_sample = action_probs[:, 0, :].to(cmp_device).data
                reward = rewards[:, :-1, :].to(cmp_device)
                states_t = states[:, 0, :, :].to(cmp_device)
                states_t_plus = states[:, 1, :, :].to(cmp_device)
                with autocast():
                    # action_t_plus, action_probs_t_plus, _ = self.nn.generate_act(observation=states_t_plus, update=False)
                    action_t, action_probs_t, _ = self.nn.generate_act(observation=states_t, update=True)
                    Q_t, Q_t_plus = self.nn.return_mean_field_Q(states_t=states_t,
                                                                states_t_plus=states_t_plus,
                                                                reward=reward,
                                                                action_t=action_t,
                                                                action_probs_t=action_probs_t)
                    loss = (Q_t_plus - Q_t).mean(dim=0).sum()

                loss_actor.append(loss.item())
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                if counter % self.soft_update_step == 0:
                    for target_parameter, main_parameter in zip(self.nn.actor.parameters(),
                                                                self.nn.actor_update.parameters()):
                        target_parameter.data.copy_(
                            (1 - self.soft_update_weight) * main_parameter + self.soft_update_weight * target_parameter)

            self.nn.actor_update.VAR *= self.nn.actor_update.temperature
            self.nn.actor.VAR *= self.nn.actor.temperature
        else:
            loss_actor = 0.
        torch.cuda.empty_cache()
        self.nn.to("cpu")
        self.nn.eval()
        return np.mean(loss_critic), np.mean(loss_actor)


if __name__ == '__main__':
    all_time = 200000
    delta_t = 30
    refresh_frequency = 1
    sw = 5
    batch_size = 8
    gamma = 0.9
    soft_update_weight = 0.1
    soft_update_step = 5
    buffer_step = 4
    buffer_size = 500
    hidden_nums = 64
    lr = 0.0005
    alpha = 0.1

    cpu_cycles = [300, 500, 1000]
    max_run_memorys = [2 ** 10, 2 ** 10 * 2, 2 ** 10 * 3]
    arrive_nums = [30, 15, 20]
    arrive_nums = [i * (refresh_frequency / delta_t) for i in arrive_nums]
    data_nums = [50, 150, 300]
    service_nums = 3

    device_nums_level_1 = 3
    device_nums_level_2 = 3
    device_nums_level_3 = 3
    link_nums = [[0, 0, 2],
                 [3, 1, 1],
                 [1, 2, 0]]


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
            device_nums=device_nums, sw=sw, refresh_frequency=refresh_frequency, delta_t=delta_t, lr=lr,
            state_nums=11 * service_nums, freedom_nums=[service_nums for _ in range(3)], hidden_nums=hidden_nums,
            soft_update_step=soft_update_step, soft_update_weight=soft_update_weight,
            edge_index=edge_index, local_edge=local_edge, batch_size=batch_size, buffer_step=buffer_step,
            buffer_size=buffer_size, alpha=alpha)

    import time

    start = time.time()
    for i in range(all_time):
        e.step()
    print(time.time() - start)
