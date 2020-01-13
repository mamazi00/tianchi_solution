from data_clean.traj import Trajectory
import numpy as np

class StayPointDensitySegmenter:
    def __init__(self, max_stay_dist_in_meter, max_stay_time_in_second):
        self.max_distance = max_stay_dist_in_meter
        self.max_stay_time = max_stay_time_in_second

    # 返回下一个超过距离阈值的点
    def find_first_exceed_max_distance(self, pt_list, cur_idx):
        cur_pt = pt_list[cur_idx]
        next_idx = cur_idx + 1
        # find all successors whose distance is within MaxStayDist w.r.t. anchor
        while next_idx < len(pt_list):
            next_pt = pt_list[next_idx]
            distance = np.sqrt((cur_pt.lat - next_pt.lat) **2 + (cur_pt.lng - next_pt.lng)**2)
            if distance > self.max_distance:
                break
            next_idx += 1
        return next_idx

    # 返回是否超过时间阈值
    def exceed_max_time(self, pt_list, cur_idx, next_idx):
        time_span = (pt_list[cur_idx].time - pt_list[next_idx - 1].time).seconds
        # the time span is larger than maxStayTimeInSecond, a stay point is detected
        return time_span > self.max_stay_time

    def segment(self, traj):
        taxi_id = traj.traj_id.split('_')[0]
        segment_traj_list = []
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return segment_traj_list
        cur_idx = 0
        cur_furthest_next_idx = float('-inf')
        # this flag is used to check whether to start a new stay point
        new_sp_flag = True
        sp_start_idx = float('-inf')
        traj_idx = 0

        while cur_idx < len(pt_list) - 1:
            next_idx = self.find_first_exceed_max_distance(pt_list, cur_idx)
            if cur_furthest_next_idx < next_idx:
                if self.exceed_max_time(pt_list, cur_idx, next_idx):
                    if new_sp_flag:
                        sp_start_idx = cur_idx
                        new_sp_flag = False
                    # the next index is expended
                    cur_furthest_next_idx = next_idx
            # nextIdx is just the next point of curIdx, we cannot expand the stay point group any further,
            # which means previous stay point cluster is determined (if has)
            if not new_sp_flag and cur_idx == cur_furthest_next_idx - 1:
                sp_end_idx = cur_idx
                if len(pt_list[traj_idx:sp_start_idx]) > 1:
                    traj_id = taxi_id + '_' + pt_list[traj_idx].time.strftime('%y%m%d%H%M') + '_' + \
                              pt_list[sp_start_idx - 1].time.strftime('%y%m%d%H%M')
                    segment_traj = Trajectory(traj_id, pt_list[traj_idx:sp_start_idx])
                    segment_traj_list.append(segment_traj)
                traj_idx = sp_end_idx + 1
                new_sp_flag = True
            cur_idx += 1
        if new_sp_flag:
            if len(pt_list[traj_idx:len(pt_list)]) > 1:
                traj_id = taxi_id + '_' + pt_list[traj_idx].time.strftime('%y%m%d%H%M') + '_' + \
                          pt_list[len(pt_list) - 1].time.strftime('%y%m%d%H%M')
                segment_traj = Trajectory(traj_id, pt_list[traj_idx:len(pt_list)])
                segment_traj_list.append(segment_traj)
        return segment_traj_list
