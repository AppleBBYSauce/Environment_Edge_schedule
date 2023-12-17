# Item's name: BaseAgent
# Autor: bby
# DateL 2023/12/3 18:26

from abc import ABC, abstractmethod
from Services import Services
import numpy as np
import _heapq
import torch


class BaseAgent(ABC):


    @abstractmethod
    def act(self,*args, **kwargs):

        """
        ### process task ###

        ### pre-process new task ###
        """

    @abstractmethod
    def update_policy(self, *args, **kwargs):

        """
        ### update policy ###
        """

    def generate_action(self, *args, **kwargs):

        """
        ###  generate action ###
        """
