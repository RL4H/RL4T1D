# Based on the paper: Non-linear Continuous Action Spaces for Reinforcement Learning in Type 1 Diabetes
# Hettiarachchi, Chirath, et al. "Non-linear Continuous Action Spaces for Reinforcement Learning in Type 1 Diabetes."
# Australasian Joint Conference on Artificial Intelligence. Cham: Springer International Publishing, 2022.
import math


class ControlSpace:
    def __init__(self, args):
        self.pump_min = args.insulin_min
        self.pump_max = args.insulin_max
        self.control_space_type = args.control_space_type

    def map(self, agent_action=None):

        if self.control_space_type == 'normal':
            agent_action = (agent_action + 1) / 2  # convert to [0, 1]
            agent_action = agent_action * self.pump_max

        elif self.control_space_type == 'sparse':
            if agent_action <= 0:
                agent_action = 0
            else:
                agent_action = agent_action * self.pump_max

        elif self.control_space_type == 'exponential':
            agent_action = self.pump_max * (math.exp((agent_action - 1) * 4))

        elif self.control_space_type == 'quadratic':
            if agent_action < 0:
                agent_action = (agent_action**2) * 0.05
                agent_action = min(0.05, agent_action)
            elif agent_action == 0:
                agent_action = 0
            else:
                agent_action = (agent_action**2) * self.pump_max

        elif self.control_space_type == 'proportional_quadratic':
            if agent_action <= 0.5:
                agent_action = ((agent_action-0.5)**2) * (0.5/(1.5**2))
                agent_action = min(0.5, agent_action)
            else:
                agent_action = ((agent_action-0.5)**2) * (self.pump_max/(0.5**2))

        pump_action = max(self.pump_min, agent_action)  # check if greater than 0
        pump_action = min(pump_action, self.pump_max)

        return pump_action
