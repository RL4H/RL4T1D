from simglucose.simulation.scenario import Action, Scenario
import numpy as np
from scipy.stats import truncnorm
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomScenario(Scenario):
    def __init__(self, start_time=None, seed=None, opt=None, env_type=None):  #, env_type='training'
        Scenario.__init__(self, start_time=start_time)
        self.opt = opt
        self.env_type= env_type
        self.seed = seed

    def get_action(self, t):  # t must be datetime.datetime object
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()
        if t_sec < 1:
            logger.info('Creating new one day scenario ...')
            self.scenario = self.create_scenario()
        t_min = np.floor(t_sec / 60.0)
        if t_min in self.scenario['meal']['time']:
            logger.info('Time for meal!')
            idx = self.scenario['meal']['time'].index(t_min)
            return Action(meal=self.scenario['meal']['amount'][idx])
        else:
            return Action(meal=0)

    def create_scenario(self):
        scenario = {'meal': {'time': [],'amount': []}}
        opt = self.opt
        time_lb = np.array(opt.time_lower_bound) * 60  # if using fractions make sure time_sigma is 30
        time_ub = np.array(opt.time_upper_bound) * 60
        time_mu = np.array(opt.meal_times_mean) * 60

        if self.env_type == 'training':
            prob = opt.meal_prob
            amount_mu = opt.meal_amount
            amount_sigma = opt.meal_variance
            time_sigma = opt.time_variance
        elif self.env_type == 'testing':
            prob = opt.val_meal_prob
            amount_mu = opt.val_meal_amount
            amount_sigma = opt.val_meal_variance
            time_sigma = opt.val_time_variance
        else:
            print("Incorrect simulation environment type")
            exit()

        # for i in range(0, len(opt.meal_amount)):
        #     amount_sigma.append(opt.meal_amount[i] * opt.meal_variance)
        # bw_arr = [68.706, 51.046, 44.791, 49.564, 47.074, 45.408, 37.898, 41.218, 43.885, 47.378]
        # bw = bw_arr[opt.patient_id]
        # opt.meal_amount[i] = opt.meal_amount[i] * bw

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma):
            if self.env_type == 'training':
                if self.random_gen.rand() < p:
                    tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd, b=(tub - tbar) / tsd, loc=tbar, scale=tsd, random_state=self.random_gen))
                    scenario['meal']['time'].append(tmeal)
                    # scenario['meal']['amount'].append(max(round(self.random_gen.normal(mbar, msd)), 0))  # normal dist
                    scenario['meal']['amount'].append(max(round(self.random_gen.uniform(mbar - 3*msd, mbar + 3*msd)), 0))  # uniform
            elif self.env_type == 'testing':
                scenario['meal']['time'].append(tbar)  # fixed meal time
                scenario['meal']['amount'].append(mbar)  # fixed meal amount
        # print(scenario['meal']['time'])
        # print(scenario['meal']['amount'])
        return scenario

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()
