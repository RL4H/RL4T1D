from collections import namedtuple, deque
import random
import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, args):
        self.args = args
        self.memory = deque([], maxlen=args.replay_buffer_size)
        self.batch_size = args.batch_size

        random.seed(args.seed)
        self.current_seed = random.random()

    def store(self, *args):
        """Save a transition, convert to tensors"""
        tensor_args = (torch.as_tensor([arg], dtype=torch.float32, device=self.args.device) for arg in args)
        self.memory.append(Transition(*tensor_args))

    def store_batch(self, datapoints):
        """ Save a transition, convert to tensors, as a batch"""

        #send list of each field to the gpu
        with torch.no_grad():
            fields = list(zip(*datapoints))
            num_data = len(fields[0])
            tensor_fields = [torch.as_tensor(field, dtype=torch.float32, device=self.args.device) for field in fields]
            
            #store each transition as a single item, linking back to the overall list
            for i in range(num_data):
                self.memory.append(Transition(*[field[i].unsqueeze(0).clone() for field in tensor_fields])) #use .clone() to avoid memory defragmentation issues

    def sample(self, batch_size):
        random.seed(self.current_seed)
        self.current_seed = random.random()
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get(self):
        transitions = self.sample(self.batch_size)
        return Transition(*zip(*transitions))
