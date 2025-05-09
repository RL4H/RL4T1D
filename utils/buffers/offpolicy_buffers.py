from collections import namedtuple, deque
import random
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, args):
        self.args = args
        self.memory = deque([], maxlen=args.replay_buffer_size)
        self.batch_size = args.batch_size

    def store(self, *args):
        """Save a transition, convert to tensors"""
        tensor_args = (torch.as_tensor([arg], dtype=torch.float32, device=self.args.device) for arg in args)
        self.memory.append(Transition(*tensor_args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get(self):
        transitions = self.sample(self.batch_size)
        return Transition(*zip(*transitions))
