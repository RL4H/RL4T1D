import pandas as pd
import pkg_resources
from collections import deque


CONTROL_QUEST = pkg_resources.resource_filename('simglucose', 'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class BasalBolusController:
    def __init__(self, args, patient_name=None, use_bolus=True, use_cf=True):
        quest = pd.read_csv(CONTROL_QUEST)
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params[patient_params.Name.str.match(patient_name)]
        u2ss = params.u2ss.values.item()
        self.BW = params.BW.values.item()

        self.q = quest[quest.Name.str.match(patient_name)]
        self.use_bolus = args.use_bolus
        self.use_cf = args.use_cf
        self.target = args.target_glucose
        self.sample_time = args.sampling_rate
        self.cf_target = args.glucose_cf_target

        self.past_meal_memory = deque(36 * [0], 36)  # 3 hours # todo: if not using 5 min sample rate

        self.basal = u2ss * self.BW / 6000
        self.TDI = self.q.TDI.values
        self.CR = self.q.CR.values  # (500/self.TDI)  #
        self.CF = self.q.CF.values  # (1800/self.TDI)  #

        self.adjust_parameters()

    def get_action(self, meal=0, glucose=0):
        # the meal value used here is the info['meal'] * sampling time. so the actual carb amount!.
        cooldown = True if sum(self.past_meal_memory) == 0 else False
        bolus = 0
        if self.use_bolus:
            if meal > 0:
                bolus = (meal / self.CR +
                         (glucose > self.cf_target) * cooldown * (glucose - self.target) / self.CF).item()
                bolus = bolus / self.sample_time
        self.past_meal_memory.append(meal)
        return self.basal + bolus

    def get_bolus(self, meal=0, glucose=0):
        # the meal value used here is the info['meal'] * sampling time. so the actual carb amount!.
        cooldown = True if sum(self.past_meal_memory) == 0 else False
        bolus = 0
        if self.use_bolus:
            if meal > 0:
                bolus = (meal / self.CR +
                         (glucose > self.cf_target) * cooldown * (glucose - self.target) / self.CF ).item()
                bolus = bolus / self.sample_time
        self.past_meal_memory.append(meal)
        return bolus

    def adjust_parameters(self):
        #self.TDI = self.BW * 0.55
        self.CR = 500 / self.TDI
        self.CF = 1800 / self.TDI
        self.basal = (self.TDI * 0.48) / (24 * 60)
        #print('Parameters adjusted!')

# patient:adolescent#001, Basal: 0.01393558889998341
# patient:adolescent#002, Basal: 0.01529933523331466
# patient:adolescent#003, Basal: 0.0107966168000268
# patient:adolescent#004, Basal: 0.01456052239999348
# patient:adolescent#005, Basal: 0.012040315333360101
# patient:adolescent#006, Basal: 0.014590183333350241
# patient:adolescent#007, Basal: 0.012943099999997907
# patient:adolescent#008, Basal: 0.009296317679986218
# patient:adolescent#009, Basal: 0.010107192533314517
# patient:adolescent#010, Basal: 0.01311652320003506
# patient:child#001, Basal: 0.006578422760004344
# patient:child#002, Basal: 0.006584850490398568
# patient:child#003, Basal: 0.004813171311526304
# patient:child#004, Basal: 0.008204957581639397
# patient:child#005, Basal: 0.00858548873873053
# patient:child#006, Basal: 0.006734515005432704
# patient:child#007, Basal: 0.007786704078078988
# patient:child#008, Basal: 0.005667427170273473
# patient:child#009, Basal: 0.006523757656342553
# patient:child#010, Basal: 0.006625406512238658
# patient:adult#001, Basal: 0.02112267499992533
# patient:adult#002, Basal: 0.022825539499994
# patient:adult#003, Basal: 0.023755205833326954
# patient:adult#004, Basal: 0.014797182203265
# patient:adult#005, Basal: 0.01966383496660751
# patient:adult#006, Basal: 0.028742228666635828
# patient:adult#007, Basal: 0.022858123833300104
# patient:adult#008, Basal: 0.01902372999996952
# patient:adult#009, Basal: 0.018896863133377337
# patient:adult#010, Basal: 0.01697815740005382
