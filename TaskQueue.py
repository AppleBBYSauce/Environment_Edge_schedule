# Item's name: TaskQueue
# Autor: bby
# Date: 2023/12/4 21:40
import numpy as np
import torch


class TaskQueue:

    def __init__(self, service_num: int, queue_size_max: list[int] = None, task_queue: list = None, recycle=None):
        """

        :param service_num: The number of services.
        This parameter represents the number of service queues in the system.
        There is no specific mean associated with this parameter since it depends on the specific system configuration
         where the TaskQueue class is being used.

        :param queue_size_max:The number of services. This parameter represents the number of service queues in the system.
        There is no specific mean associated with this parameter since it
        depends on the specific system configuration where the TaskQueue class is being used.

        :param task_queue:A  NumPy array representing the task queue for each service.
        This parameter stores the tasks in the system, organized by service,
        key, and task index. The mean of task_queue is not applicable since it represents the initial state of the task queues and not numerical values.

        :param recycle:An instance of the TaskQueue class representing a recycle station.
        This parameter provides an optional recycle station where dropped tasks can be sent.
         The mean of recycle is not applicable since it represents the initial state of the recycle station.
        """

        # Initialize TaskQueue instance
        self.service_num = service_num
        self.recycle = recycle

        # Set the maximum queue sizes
        if queue_size_max is None:
            self.queue_size_max = np.array([np.inf for _ in range(service_num)])
        else:
            self.queue_size_max = queue_size_max

        # Set the current queue sizes
        self.queue_size = np.array([0 for _ in range(service_num)])

        # Define the mapping from keys to indices
        # rd | cp | rm | dl #
        self.idx2key = {"rd": 0, "cp": 1, "rm": 2, "dl": 3}

        # Initialize the task queues
        self.task_queue = [np.array([[] for _ in range(4)]) for _ in range(service_num)]

        # Populate the task queues with initial tasks
        if task_queue is not None:
            for l in range(self.service_num):
                self.task_queue[l] = np.concatenate([self.task_queue[l], np.array(task_queue[l])], axis=1)
                self.queue_size[l] = self.task_queue[l].shape[1]

    def drop_task(self, l: int, number: int, key: str = "rd", reverse: bool = True, sort: bool = True):
        # Drop tasks from the task queue of a specific service

        if sort:
            # Sort the task queue based on the specified key
            idx = np.argsort(self.task_queue[l][self.idx2key[key]], kind="mergesort")
            self.task_queue[l] = self.task_queue[l][:, idx]

        if reverse:
            # Remove tasks from the end of the task queue
            drop_tasks = self.task_queue[l][:, self.task_queue[l].shape[1] - number:]
            self.task_queue[l] = self.task_queue[l][:, :self.task_queue[l].shape[1] - number]
        else:
            # Remove tasks from the beginning of the task queue
            drop_tasks = self.task_queue[l][:, :number]
            self.task_queue[l] = self.task_queue[l][:, number:]

        # Update the queue size
        self.queue_size[l] = self.task_queue[l].shape[1]

        # Return the updated task queue and the dropped tasks
        return self.task_queue[l], drop_tasks

    def drop_task_(self, number: list[int], key: str = "rd", reverse: bool = False, sort: bool = True,
                   enable_recycle: bool = True):
        # Drop tasks from the task queue for each service

        drop_tasks = []  # List to store dropped tasks for each service

        for l in range(self.service_num):
            if number[l]:
                # Drop tasks from the current service's task queue
                _, drop_task = self.drop_task(l=l, number=number[l], key=key, reverse=reverse, sort=sort)
                drop_tasks.append(drop_task)
            else:
                # If the number is 0, no tasks are dropped for the current service
                drop_tasks.append(np.array([[] for _ in range(4)]))

        if self.recycle is not None and enable_recycle:
            # If a recycle station is available and recycling is enabled

            # Merge the dropped tasks with the recycle station's task queue
            self.recycle.merger_task_queue(TaskQueue(service_num=self.service_num, task_queue=drop_tasks))

            # Return the recycle station
            return self.recycle
        else:
            # If no recycle station is available or recycling is disabled

            # Return a new TaskQueue instance with the dropped tasks
            return TaskQueue(service_num=self.service_num, task_queue=drop_tasks)

    def return_mean(self, key: str = None):
        # Return the mean of the specified key's tasks in the task queue for each service

        if key is None:
            # If no key is specified, calculate the mean for all keys

            # Iterate over each service and calculate the mean for each key
            return np.array([
                np.nanmean(self.task_queue[l], axis=1) if self.queue_size[l] > 0 else np.array(
                    [1e-8 for _ in range(len(self.idx2key))])
                for l in range(self.service_num)
            ])
        else:
            # If a key is specified, calculate the mean only for that key

            # Iterate over each service and calculate the mean for the specified key
            return np.array([
                np.nanmean(self.task_queue[l][self.idx2key[key], :]) if self.queue_size[l] > 0 else 1e-8
                for l in range(self.service_num)
            ])

    def return_std(self, key: str = None):
        if key is None:
            return np.array([np.nanstd(self.task_queue[l], axis=1) if self.queue_size[l] > 0 else np.array(
                [1e-8 for _ in range(len(self.idx2key))]) for l in range(self.service_num)])
        else:
            return np.array(
                [np.nanstd(self.task_queue[l][self.idx2key[key], :]) if self.queue_size[l] > 0 else 1e-8 for l in
                 range(self.service_num)])

    def return_sum(self, key: str = None):
        if key is None:
            return np.array([np.sum(self.task_queue[l], axis=1) for l in range(self.service_num)])
        else:
            return np.array([np.sum(self.task_queue[l][self.idx2key[key], :]) for l in range(self.service_num)])

    def merger_task_queue(self, new_task_queue, drop: bool = False, key: str = "rd", reverse: bool = True):
        # Merge the task queues of the current instance and a new TaskQueue instance

        drop_task_queue = []  # List to store dropped tasks for each service

        for l in range(self.service_num):
            # Concatenate the task queues of the current instance and the new TaskQueue instance
            self.task_queue[l] = np.concatenate([self.task_queue[l], new_task_queue.task_queue[l]], axis=1)

            if drop:
                # If drop is enabled, drop excess tasks based on the queue size limit

                # Drop tasks from the current service's task queue
                _, drop_tasks = self.drop_task(l, max(int(self.task_queue[l].shape[1] - self.queue_size_max[l]), 0),
                                               key=key, reverse=reverse)
                drop_task_queue.append(drop_tasks)
            else:
                # If drop is disabled, add an empty array to indicate no tasks are dropped
                drop_task_queue.append(np.array([[] for _ in range(4)]))

        for l in range(self.service_num):
            # Update the queue size for each service
            self.queue_size[l] = self.task_queue[l].shape[1]

        if self.recycle is not None:
            # If a recycle station is available

            # Merge the dropped tasks with the recycle station's task queue
            self.recycle.merger_task_queue(TaskQueue(service_num=self.service_num, task_queue=drop_task_queue))

            # Return the recycle station
            return self.recycle
        else:
            # If no recycle station is available

            # Return a new TaskQueue instance with the dropped tasks
            return TaskQueue(service_num=self.service_num, task_queue=drop_task_queue)

    def op(self, func, key: str = "rd"):
        for l in range(self.service_num):
            self.task_queue[l][self.idx2key[key]] = func(self.task_queue[l][self.idx2key[key]])

    def merger_single_service(self, l: int, task_queue):
        self.task_queue[l] = np.append(self.task_queue[l], task_queue, axis=1)
        self.queue_size[l] += len(task_queue[0])

    def pop(self, l):
        pop_item = self.task_queue[l][:, 0]
        self.task_queue[l] = np.delete(self.task_queue[l], 0, axis=1)
        self.queue_size[l] -= 1
        return pop_item

    def insert(self, l, item):
        self.task_queue[l] = np.append(arr=self.task_queue[l], values=item, axis=1)
        self.queue_size[l] += 1

    def get_size(self):
        self.queue_size = [i.shape[1] for i in self.task_queue]
        return np.array(self.queue_size.copy())

    def get_key_data(self, key: str, l: int = None):
        if l is None:
            return [self.task_queue[l][self.idx2key[key]] for l in range(self.service_num)]
        return self.task_queue[l][self.idx2key[key]]

    def get_element_inf(self, l: int, idx: int, key: str = None):
        if key is None:
            return self.task_queue[l][:, idx]
        else:
            return self.task_queue[l][self.idx2key[key], idx]

    def set_element_inf(self, l: int, idx: int, key: str, value):
        self.task_queue[l][self.idx2key[key], idx] = value

    def drop_with_key(self, func, key: str, *args, **kwargs):
        # Drop tasks based on a given key and a specified function

        drop_tasks = []  # List to store dropped tasks for each service

        for l in range(self.service_num):
            tmp = self.task_queue[l][self.idx2key[key]]

            if tmp.size > 0:
                # If there are tasks associated with the key in the current service

                # Apply the specified function to determine indices of tasks to be dropped
                idx = func(tmp, kwargs["y"][l])

                # Retrieve the dropped tasks and update the task queue
                drop_tasks.append(self.task_queue[l][:, idx])
                self.task_queue[l] = self.task_queue[l][:, ~idx]
                self.queue_size[l] = self.task_queue[l].shape[1]
            else:
                # If no tasks are associated with the key in the current service
                drop_tasks.append(np.array([[] for _ in range(4)]))

        droper = TaskQueue(service_num=self.service_num, task_queue=drop_tasks)

        if self.recycle:
            # If a recycle station is available

            # Merge the dropped tasks with the recycle station's task queue
            self.recycle.merger_task_queue(TaskQueue(service_num=self.service_num, task_queue=drop_tasks))

        return droper

    def set_recycle(self, recycle_station):
        self.recycle = recycle_station

    def clean(self):
        self.queue_size = np.array([0 for _ in range(self.service_num)])
        self.task_queue = [np.array([[] for _ in range(4)]) for _ in range(self.service_num)]
        # self.recycle = None
