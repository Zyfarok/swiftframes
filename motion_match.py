import cv2 as cv
import numpy as np
from motion_feature import MotionFeature
from typing import Tuple, Optional
from functools import reduce
from math import floor

class KP:
    def __init__(self, kp: cv.KeyPoint):
        self.response = kp.response
        self.size = kp.size
        self.octave = kp.octave
        self.pt = np.array(kp.pt)

class MotionMatch:
    def __init__(self, start_id: int,
            kps: Tuple[KP, ...], kp_ids: Tuple[int, ...],
            match_distances: Tuple[float, ...],
            pt_derivss: Optional[Tuple[Tuple[np.ndarray, ...], ...]] = None,
            sq_distss: Optional[Tuple[Tuple[float, ...], ...]] = None,
            accel_inv_sq_scale: float = 2 * (3 ** 2), speed_sq_scale: float = 0.5 / (30 ** 2)):
        
        self.start_id = start_id
        self.kps = kps
        self.kp_ids = kp_ids
        self.match_distances = match_distances

        if pt_derivss is None:
            query_pt, train_pt = kps[0].pt, kps[1].pt
            speed = train_pt - query_pt
            pt_derivss = ((query_pt, train_pt), (speed,),)
        self.pt_derivss: Tuple[Tuple[np.ndarray, ...], ...] = pt_derivss

        if sq_distss is None:
            speed = pt_derivss[1][0]
            sq_dist = np.dot(speed, speed)
            sq_distss = ((sq_dist,),)
        self.sq_distss: Tuple[Tuple[float, ...], ...] = sq_distss

        self.accel_inv_sq_scale = accel_inv_sq_scale
        self.speed_sq_scale = speed_sq_scale

        self.size = len(kps)
        self.degree = len(match_distances)
        self.end_id = start_id + self.degree

        self._average_sq_dists = None
        self._final_distance = -1.0

    def startKeypointIds(self):
        return self.kp_ids[:-1]
    
    def endKeypointIds(self):
        return self.kp_ids[1:]

    def merge(self, secondMatch: 'MotionMatch'):
        new_kps = self.kps + secondMatch.kps[-1:]
        new_kp_ids = self.kp_ids + secondMatch.kp_ids[-1:]
        new_match_distances = self.match_distances + secondMatch.match_distances[-1:]


        # Concatenate previous 
        new_pt_derivss = list(map(lambda x, y: x + y[-1:], self.pt_derivss, secondMatch.pt_derivss))
        new_sq_distss = list(map(lambda x, y: x + y[-1:], self.sq_distss, secondMatch.sq_distss))
        
        # Compute and add new deriv if current degree < 3
        if len(self.sq_distss) < 3:
            deriv = new_pt_derivss[-1][1] - new_pt_derivss[-1][0]
            new_pt_derivss.append((deriv,))
            sq_dist = np.dot(deriv, deriv)
            new_sq_distss.append((sq_dist,))
        return MotionMatch(self.start_id, new_kps, new_kp_ids, new_match_distances,
                pt_derivss=tuple(new_pt_derivss), sq_distss=tuple(new_sq_distss),
                speed_sq_scale=self.speed_sq_scale, accel_inv_sq_scale=self.accel_inv_sq_scale)

    def avg_sq_dists(self):
        if self._average_sq_dists is None:
            self._average_sq_dists = tuple(map(lambda dists: sum(dists) / len(dists), self.sq_distss))
        return self._average_sq_dists

    def final_distance(self):
        if self._final_distance == -1.0 :
            # Compute avg match distance
            match_dist = (reduce(lambda x, y: x + y, self.match_distances, 0) / (2 * self.degree)) + 0.5
            match_dist *= match_dist
            #print("match_dist:", match_dist, self.match_distances, self.degree)

            # Compute motion coherence distance
            
            asds = self.avg_sq_dists()
            pt_dist = asds[0] * self.speed_sq_scale + 0.5
            
            if(len(asds) > 1):
                accel_sq_scale = 1 / (self.accel_inv_sq_scale + (asds[0] / 32))
                for asd in asds[1:]:
                    pt_dist *= asd * accel_sq_scale + 0.5
                
            
            #print("pt_dist:", pt_dist, asds, self.sq_distss, 1/accel_sq_scale, self.kps)

            # Compute final distance
            self._final_distance = match_dist * pt_dist
            #print("final:", self._final_distance)
        return self._final_distance

    def genMotionFeature(self, at_pos: int, at_time: float, verbose=False):
        assert at_time >= 0 and at_time <= 1, "at_time should be between 0 and 1"
        match_time = (at_pos - self.start_id) + at_time
        match_pos = max(min(int(floor(match_time)), self.degree - 1), 0)
        relative_time = match_time - match_pos
        match_distance = self.final_distance()
        quality = 1 / (1 + match_distance)
        
        query_kp, train_kp = self.kps[match_pos], self.kps[match_pos + 1]
        query_pt, train_pt = query_kp.pt, train_kp.pt
        query_resp, train_resp = query_kp.response, train_kp.response
        query_size, train_size = query_kp.size, train_kp.size

        pt = query_pt + relative_time * (train_pt - query_pt)
        size = max(query_size + relative_time * (train_size - query_size), 0)

        if(verbose):
            print("pos:", pt, "from:", query_pt, "to:", train_pt, "fnl_dist:", self.final_distance(), "quality:", quality, "asds:", self.avg_sq_dists(), "resp:", query_resp, train_resp)
        if at_pos < self.start_id:
            # past motion : reduce quality
            quality = quality * (max(0, (at_time - self.start_id) + 1) ** 2)
            return MotionFeature(pt, None, query_pt, size, quality * query_resp)
        elif at_pos > self.end_id:
            # future motion : reduce quality
            quality = quality * (max(0, (end_id - at_time) + 1) ** 2)
            return MotionFeature(pt, train_pt, None, size, quality * train_resp)
        else:
            resp = query_resp + relative_time * (train_resp - query_resp)
            return MotionFeature(pt, query_pt, train_pt, size, quality * resp)
