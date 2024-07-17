import numpy as np
from collections import deque
from utils import core


class ObservationSpace:
    def __init__(self, args):
        self.args = args
        self.obs_window = args.obs_window
        self.obs_features = args.obs_features
        self.n_obs_features = len(args.obs_features)
        self.feature_dict = {key: deque(self.obs_window*[0], self.obs_window) for key in self.obs_features}

    def update(self, dict):
        dict = self.scale_state_space(dict)
        for key, _ in self.feature_dict.items():
            self.feature_dict[key].append(dict[key])
        return np.stack(list(self.feature_dict.values()), axis=-1).astype(np.float32)

    def scale_state_space(self, dict):
        for key, value in dict.items():
            if key == 'cgm':
                dict['cgm'] = core.linear_scaling(x=dict['cgm'], x_min=self.args.glucose_min, x_max=self.args.glucose_max)
            elif key == 'insulin':
                dict['insulin'] = core.linear_scaling(x=dict['insulin'], x_min=self.args.insulin_min, x_max=self.args.insulin_max)

            elif key == 'time_to_meal':
                dict['time_to_meal'] = core.linear_scaling(x=dict['time_to_meal'], x_min=0, x_max=self.args.t_meal)
            elif key == 'future_carb':
                dict['future_carb'] = core.linear_scaling(x=dict['future_carb'], x_min=0, x_max=120)  # assume max carbs 120
            # meal_type: is already scaled to (0,1) =>  0 - no meal,  0.3 - snack, 1 - meain meal

            elif key == 'day_hour':
                dict['day_hour'] = core.linear_scaling(x=dict['day_hour'], x_min=0, x_max=23)   # hour is given 0-23
            elif key == 'day_min':
                dict['day_min'] = core.linear_scaling(x=dict['day_min'], x_min=0, x_max=59)  # assume max carbs 120
            return dict


# Note: In our previous research we used additional handcrafted features like insulin on board (IoB)
# Removed from this implementation since the raw data provided has these information for the neural network to learn.

