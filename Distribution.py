import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Distribution:
    def __init__(self, path):
        self.df = pd.read_csv(path, encoding='utf-8', sep=',', engine='python')

    def draw_histogram(self, attr, is_all=True):
        if is_all:
            distribution = self.df.loc[self.df[attr] != 0, attr].tolist()
            print("test")
            plt.hist(distribution, bins=np.arange(np.min(distribution), np.max(distribution), 0.1), facecolor="blue", edgecolor="black",alpha=0.5)
            print(np.max(distribution), np.min(distribution), np.mean(distribution))
        else:
            id_lst = self.df['渔船ID'].unique()
            for id in id_lst:
                distribution = self.df.loc[self.df["渔船ID"] == id, attr].tolist()
                if len(distribution) != 0 and len(set(distribution)) > 1:
                    plt.hist(distribution, bins=np.arange(np.min(distribution), np.max(distribution), 0.1), facecolor="blue", edgecolor="black", alpha=0.5)
                    # print(id, np.max(distribution), np.min(distribution), np.mean(distribution))
                    continue
                # print(id, "is none...")
        title = "distribution of " + attr
        plt.title(title, fontsize=24)


        plt.show()


    def __str__(self):
        return self.df

if __name__ == '__main__':
    path = 'E:/天池/all_table.csv'
    # path = 'E:/天池/hy_round1_train_20200102/0.csv'
    Distribution(path).draw_histogram("速度")
