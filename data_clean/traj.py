from datetime import datetime
import numpy as np

class Trajectory:
    def __init__(self, traj_id, pt_list):
        self.traj_id = traj_id
        self.pt_list = pt_list

    def length(self):
        if len(self.pt_list) <= 1:
            return 0.0
        else:
            dist = 0.0
            pre_pt = None
            for pt in self.pt_list:
                if pre_pt is None:
                    pre_pt = pt
                else:
                    dist += np.sqrt((pre_pt.lat - pre_pt.lat) **2 + (pt.lng - pt.lng)**2)
                    pre_pt = pt
            return dist

    def __str__(self):
        return '{},{},{}'.format(self.traj_id, self.pt_list[0].time.strftime('%Y-%m-%d %H:%M:%S'),
                                 self.pt_list[-1].time.strftime('%Y-%m-%d %H:%M:%S'))
