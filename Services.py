# Item's name: Services
# Autor: bby
# DateL 2023/12/2 17:24
from scipy.stats import expon, poisson, gumbel_l
import numpy as np
import matplotlib.pyplot as plt
from TaskQueue import TaskQueue

euler_constant = 0.57721
np.random.seed(0)


class Services:
    def __init__(self, cpu_cycles: list, max_run_memorys: list, arrive_nums: list, data_nums: list, device_nums: int,
                 service_nums: int, sigma_gumbel=2):
        # Constructor method for the Services class

        """
        Services class encapsulates the functionality to generate new services with random task parameters,
        allowing for the simulation or modeling of a system with multiple services and their associated tasks.

        :param cpu_cycles: This parameter represents the CPU cycles for each task in the services. Since it is a list,
        the mean would be the average of all the values in the list.

        :param max_run_memorys: This parameter represents the maximum run memory for each task in the services. Again,
        it is a list, so the mean would be the average of the values in the list.

        :param arrive_nums: This parameter represents the arrival numbers for the inactive queues of tasks for each device.
        It is also a list, so the mean would be the average of the values in the list.

        :param data_nums: This parameter represents the arrival numbers for the inactive queues of tasks for each device.
        It is also a list, so the mean would be the average of the values in the list.

        :param device_nums:  This parameter represents the number of devices in the system.
        It is an integer value, so its mean is simply the value itself.
        :param service_nums: This parameter represents the number of services in the system. Similar to device_nums,
        it is an integer, so its mean is the value itself.
        """

        # Initialize instance variables
        self.cpu_cycles = cpu_cycles
        self.max_run_memorys = max_run_memorys
        self.arrive_nums = arrive_nums
        self.data_nums = data_nums
        self.device_nums = device_nums
        self.sigma_gumbel = sigma_gumbel
        self.service_nums = service_nums

        # Set up random number generators
        self.task_num_generator = np.vectorize(np.random.poisson)  # Generates Poisson-distributed random numbers

        self.max_run_memory_generator = np.random.gumbel  # Generates Gumbel-distributed random numbers
        self.sigma_gembel = [self.sigma_gumbel for _ in
                             range(service_nums)]  # List of sigma values for Gumbel distribution
        self.loc_gumbel = [m - euler_constant * sig for m, sig in
                           zip(self.max_run_memorys,
                               self.sigma_gembel)]  # Calculate loc parameter for Gumbel distribution

        self.data_num_generator = lambda x, size: np.array([x] * size)  # Generates an array of data_nums

        self.cpu_cycle_generator = np.random.normal  # Generates exponentially-distributed random numbers

    def generate_new_services(self, device_nums: int = None):
        # Generate new services with random task parameters

        if device_nums is None:
            device_nums = self.device_nums

        wt = [self.task_num_generator(self.arrive_nums) for _ in range(device_nums)]
        # wt: the number of inactive queues of tasks for each device, generated using Poisson distribution

        rm = [[self.max_run_memory_generator(loc, scale, size).astype(int).tolist() for loc, scale, size in
               zip(self.loc_gumbel, self.sigma_gembel, wt[i])] for i in range(device_nums)]
        # rm: the run memory for each task, generated using Gumbel distribution

        dl = [[self.data_num_generator(num, size).astype(int).tolist() for num, size in zip(self.data_nums, wt[i])] for
              i in range(device_nums)]
        # dl: the data num for each task, generated using the provided data_nums

        cp = [[(self.cpu_cycle_generator(loc=num, size=size) + 1).astype(int).tolist() for num, size in
               zip(self.cpu_cycles, wt[i])]
              for i in range(device_nums)]
        # cp: the CPU cycle for each task, generated using exponential distribution

        rd = [[[0] * num for num in wt_] for wt_ in wt]
        # rd: the re-direct num for each task, initialized with zeros

        new_task_queues = []
        for collect in zip(rd, cp, rm, dl):
            # Collect the generated task parameters for each device
            new_task_queues.append(
                TaskQueue(service_num=self.service_nums,
                          task_queue=[[rd_, cp_, rm_, dl_] for rd_, cp_, rm_, dl_ in zip(*collect)]))
            # Create a TaskQueue instance with the collected task parameters

        return new_task_queues


if __name__ == '__main__':
    cpu_cycles = [15, 100, 300]
    max_run_memorys = [100, 200, 500]
    arrive_nums = [10, 5, 3]
    data_nums = [50, 100, 300]
    device_nums = 1

    S = Services(cpu_cycles=cpu_cycles, max_run_memorys=max_run_memorys, arrive_nums=arrive_nums, data_nums=data_nums,
                 device_nums=device_nums, service_nums=3)
    S.generate_new_services()

# # Seed random number generator
# np.random.seed(42)
#
# # Compute mean no-hitter time: tau
# tau = 10
#
# # Draw out of an exponential distribution with parameter tau: inter_nohitter_time
# inter_nohitter_time = np.random.exponential(tau, 100000)
#
# # Plot the PDF and label axes
# _ = plt.hist(inter_nohitter_time, bins=50)
# _ = plt.xlabel('Games between no-hitters')
# _ = plt.ylabel('PDF')
#
# # Show the plot
# plt.show()
