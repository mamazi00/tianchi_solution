from data_clean.traj import Trajectory
from data_clean.stay_point import StayPointDensitySegmenter
import numpy as np

class TrajectoryFilter:
    def __init__(self, max_interval_in_minute, max_speed,
                 max_stay_dist_in_meter, max_stay_time_in_second):
        self.max_interval = max_interval_in_minute
        self.max_speed = max_speed
        self.segmenter = StayPointDensitySegmenter(max_stay_dist_in_meter, max_stay_time_in_second)

    def normal(self, pre_pt, cur_pt):
        time_span = (cur_pt.time - pre_pt.time).seconds
        distance = np.sqrt((pre_pt.lat - cur_pt.lat) **2 + (pre_pt.lng - cur_pt.lng)**2)
        return time_span > 0 and distance / time_span <= self.max_speed

    def temporal_near(self, pre_pt, cur_pt):
        time_span = (cur_pt.time - pre_pt.time).seconds
        return time_span <= self.max_interval * 60

    def filter(self, traj):
        clean_traj_list = []
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return clean_traj_list
        clean_pt_list = []
        pre_pt = None
        for cur_pt in pt_list:

            if pre_pt is None:
                clean_pt_list.append(cur_pt)
                pre_pt = cur_pt
            else:
                if cur_pt.time > pre_pt.time:
                    if self.normal(pre_pt, cur_pt):
                        if self.temporal_near(pre_pt, cur_pt):
                            clean_pt_list.append(cur_pt)
                        else:
                            # segment the trajectory if exceed time interval limitation
                            if len(clean_pt_list) > 1:
                                raw_traj = Trajectory(traj.traj_id, clean_pt_list)
                                segment_traj_list = self.segmenter.segment(raw_traj)
                                clean_traj_list.extend([t for t in segment_traj_list if self.dist_requirement(t)])
                            clean_pt_list = [cur_pt]
                        pre_pt = cur_pt
                        # skip noise point
                    # skip time no increase point
        # construct trajectory for the last clean_pt_list
        if len(clean_pt_list) > 1:
            raw_traj = Trajectory(traj.traj_id, clean_pt_list)
            segment_traj_list = self.segmenter.segment(raw_traj)
            clean_traj_list.extend([t for t in segment_traj_list if self.dist_requirement(t)])
        return clean_traj_list
