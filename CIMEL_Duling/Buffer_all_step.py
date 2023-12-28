# Item's name: Buffer
# Autor: bby
# DateL 2023/12/6 20:48

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from collections import deque
import torch


class Buffer(Dataset):

    def __init__(self, step: int = 1, buffer_size: int = 200):
        self.action = deque(maxlen=buffer_size)
        self.state = deque(maxlen=buffer_size)
        self.reward = deque(maxlen=buffer_size)
        self.action_prob = deque(maxlen=buffer_size)
        self.step = step
        self.load_w = 1
        self.task_latency_w = 1

    def add_data(self, action, action_prob, state, reward):
        reward[:, 0] = (reward[:, 0] - reward[:, 0].mean()) ** 2 * self.load_w
        reward[:, 1] = reward[:, 1] * self.task_latency_w
        reward = -reward.sum(dim=1, keepdims=True)

        self.action.appendleft(action)
        self.state.appendleft(state)
        self.reward.appendleft(reward)
        self.action_prob.appendleft(action_prob)

    def __len__(self):
        return len(self.action) - self.step

    def get_length(self):
        return len(self.action)

    def __getitem__(self, item):
        return (
            torch.stack([self.action[i] for i in range(item, item + self.step)], dim=0),
            torch.stack([self.action_prob[i] for i in range(item, item + self.step)], dim=0).squeeze(-1),
            torch.stack([self.state[i] for i in range(item, item + self.step)], dim=0),
            torch.stack([self.reward[i] for i in range(item, item + self.step)], dim=0).squeeze(-1),
        )

    def BufferLoader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, pin_memory=True)


if __name__ == '__main__':
    pass
