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
        """Save a transition batch directly as tensors on GPU."""

        # Transpose list of datapoints into fields
        fields = list(zip(*datapoints))

        # Check shape consistency (same as before, but vectorized)
        for idx, field in enumerate(fields):
            shapes = {tuple(x.shape) for x in field}
            if len(shapes) > 1:
                raise ValueError(f"Inconsistent shapes in field {idx}: {[x.shape for x in field]}")

        # Convert each field into a single batched tensor on GPU
        tensor_fields = [
            torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in field]).to(self.args.device)
            for field in fields
        ]

        # Store the batch as one transition object instead of per-element loop
        self.memory.append(Transition(*tensor_fields))

    def sample(self, batch_size):
        random.seed(self.current_seed)
        self.current_seed = random.random()
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get(self):
        transitions = self.sample(self.batch_size)
        return Transition(*zip(*transitions))
