import pandas as pd
import numpy as np
import datetime
import glob
from joblib import Parallel, delayed
from data_clean.traj import *
from data_clean.stpoint import *

max_interval_in_minute =
max_speed =
max_stay_dist_in_meter =
max_stay_time_in_second =



def gen_traj(path):
    traj_id = path.split("\\")[-1].rstrip(".csv")
    df = pd.read_csv(path, encoding='utf-8', sep=',', engine='python')
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    pt_lst = []
    for idx, row in df.iterrows():
        pt_lst.append(STPoint(row['x'], row['y'], row['time']))
        pt_lst(STPoint(row['x'], row['y'], row['time']))
    return Trajectory(traj_id,pt_lst)

def cleaning(path):
    traj = gen_traj(path)



if __name__ == '__main__':
    train_feat = Parallel(n_jobs=1)(delayed(feature_engine)(path__, True)
                                     for path__ in glob.glob(r'E:/天池/hy_round1_train_20200102/*')[:])
    train_feat = pd.DataFrame(train_feat)