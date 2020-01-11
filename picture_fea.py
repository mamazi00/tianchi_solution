import matplotlib.pyplot as plt
import matplotlib
import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

def get_location(path,test_mode=False):
    print(path.split("\\")[-1])
    df = pd.read_csv(path, encoding='utf-8', sep=',', engine='python')
    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    x = df['x'].tolist()
    y = df['y'].tolist()
    if test_mode:
        return x,y
    type = df['type'][0]
    return x,y,type

def gen_picture(path,test_mode=False):
    fig_path = 'E:/天池/fig/mixture/test'
    if test_mode:
        x,y = get_location(path,test_mode)
        fig_name = path.split("\\")[-1].rstrip(".csv") + ".jpg"
    else:
        x,y,type = get_location(path)
        print(type)
        fig_name = path.split("\\")[-1].rstrip(".csv") + "_" + type + ".jpg"
    plt.scatter(x, y, 10, 'k')
    x_delta = (np.max(x)-np.min(x))/20
    y_delta = (np.max(y)-np.min(y))/20
    plt.ylim(np.min(y)-y_delta, np.max(y)+y_delta)
    plt.xlim(np.min(x)-x_delta, np.max(x)+x_delta)
    plt.plot(x, y, 'k')
    plt.axis('off') # 关闭坐标
    # chinfo = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    # plt.title(type, fontsize=24, fontproperties=chinfo)

    plt.savefig(fig_path+'/'+fig_name)
    # plt.show()

if __name__ == '__main__':
    gen_ = Parallel(n_jobs=10)(delayed(gen_picture)(path,True)
                                     for path in glob.glob(r'E:/天池/hy_round1_testA_20200102/*')[:])