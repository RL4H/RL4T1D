import os 
import sys 
from decouple import config 
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)


from metrics.statistics_alt import get_summary_stats, read_file

ALGO = "custom"




data = read_file(experiment_name="CustomTest1", algorithm=ALGO, n_trials=2, base_num=5000)

for nums in data:
    print("worker_episode_",nums,sep='')
    for c,episode_data in enumerate(data[nums]):
        print("\tEpisode",c+1)
        for k in episode_data:
            print("\t\t",k,':',','.join([str(round(float(i),2)) for i in episode_data[k]]))
        print()