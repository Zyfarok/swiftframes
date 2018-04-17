from typing import List, Tuple
import numpy as np
import cv2 as cv
import numpy as np
import time

## SURF Defaults : hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False
surf: cv.Feature2D = cv.xfeatures2d.SURF_create(hessianThreshold=25, nOctaves=4, nOctaveLayers=3, extended=False, upright=True)

#### Matchers
bfmatcher: cv.DescriptorMatcher = cv.BFMatcher()
#bfcrossmatcher: cv.DescriptorMatcher = cv.BFMatcher(crossCheck=True)
#flannmatcher = cv.FlannBasedMatcher(
#    dict(algorithm = 3), # , trees = 5
#    dict() # check = 50
#)

#degree_one_max = 400

class motionMatch:
    def __init__(self, start_id: int, match_distance: float, *kps: Tuple[cv.KeyPoint, ...]):
        self.start_id = start_id
        self.match_distance = match_distance
        self.kps = kps
        self.degree = len(kps) - 1
        self.end_id = start_id + self.degree
        self.center = (start_id + self.end_id) / 2.0

    def genMotionFeature(self, at_time: float):
        time_pos = max(min(int(floor(at_time)), self.degree - 1),0)
        relative_time = at_time - time_pos
        quality = pow(2, self.degree) / (0.01 + self.matchDistance)
        query_kp, train_kp = self.kps[time_pos], self.kps[time_pos + 1]
        query_pt, train_pt = np.array(query_kp.pt), np.array(train_kp.pt)
        pt = query_pt + relative_time * (train_pt - query_pt)
        size = query_kp.size + relative_time * (train_kp.size - query_kp.size)
        if at_time < start_id:
            # past motion : reduce quality
            quality = quality * pow(2, 2 * (at_time - start_id))
            return motionFeature(pt, None, query_pt, size, quality)
        elif at_time > end_id:
            # future motion : reduce quality
            quality = quality * pow(2, 2 * (end_id - at_time))
            return motionFeature(pt, train_pt, None, size, quality)
        else:
            return motionFeature(pt, query_pt, train_pt, size, quality)

class motionFeature:
    def __init__(self, pt: np.ndarray, query_pt, train_pt, size, quality: float):
        self.pt = np.array(pt)
        
        self.query_pt = query_pt
        self.train_pt = train_pt
        self.size = size
        self.quality = quality

def computeMotion(imgs: List[np.ndarray], degree: int = 1):
    assert isinstance(degree,int), "degree should be an int"
    assert degree > 0 and degree < 5, "degree should be between 1 and 4"

    #### Detect and compute features :
    start_time = time.time()
    kpss = surf.detect(imgs, None)
    print("detect time :", time.time() - start_time)

    start_time = time.time()
    descss = surf.compute(imgs,kpss)[1]
    print("compute time :", time.time() - start_time)

    for kps in zip(kpss, descss):
        print('kps len : ', len(kps))

    #### Match features
    matchesss = matchFeaturess(descss, bfmatcher, k=5)
        
    # Translate matches to motionMatches and improve distance

    simple_motion_matchesss = computeMotionMatchesss(matchesss, kpss, imgs)
    if(degree == 1):
        return bestMotionMatchess(simple_motion_matchesss), list(), list()
    


    """
    if(degree == 1):
        matchess = list(sortMatchess(matchess, kpss, imgs))
        for i in range(len(matchess)):
            matches = matchess[i][:degree_one_max]
            query_kps = kpss[i]
            train_kps = kpss[i+1]
            motion_features = list()
            for match in matches:
                query_kp = query_kps[match.queryIdx]
                train_kp = train_kps[match.trainIdx]
                motion = (query_kp.pt, train_kp.pt)
                motion_features.append(motionVector(motion))
            motion_featuresss.append((motion_features,)) # Inside tuples
            appearing_featuress.append(list())
            dispappearing_featuress.append(list())
        # TODO : Compute Motion (simple feature matching)
    elif (degree == 2):
        for i in range(len(matchess)):
            matches = matchess[i][:degree_one_max]
            query_kps = kpss[i]
            train_kps = kpss[i+1]
            motion_features = list()
            for match in matches:
                query_kp = query_kps[match.queryIdx]
                train_kp = train_kps[match.trainIdx]
                motion = (query_kp.pt, train_kp.pt)
                motion_features.append(motionVector(motion))
            motion_featuresss.append((motion_features,)) # Inside tuples
            appearing_matchesss.append(list())
            disappearing_matchesss.append(list())
        # TODO : Compute Motion (simple motion coherence)
    elif (degree == 3):
        pass
        # TODO : Compute Motion (better motion coherence
        # + hidden features)
    elif (degree == 4):
        pass
        # TODO : Compute Motion (better motion coherence
        # + better hidden features motion coherence)
    else:
        print("Wrong motion interpolation degree")
        exit(1)
    """
    return final_motion_matchesss, appearing_matchesss, disappearing_matchesss


def matchFeaturess(descss: list, matcher: cv.DescriptorMatcher, k=1):
    def matchSelect(matches):
        return list(filter(lambda match: match.distance < (matches[0].distance * 2), matches))
    matchesss = list()
    for query_descs, train_descs in zip(descss[:-1], descss[1:]):
        matchess = matcher.knnMatch(query_descs, train_descs, k)
        matchesss.append(list(map(matchSelect, matchess)))
    return matchesss


def computeMotionMatchesss(matchesss, kpss, imgs):
    motion_matchesss = list()
    for i in range(len(matchesss)):
        matchess = matchesss[i]
        query_kps, train_kps = kpss[i], kpss[i+1]
        query_img, train_img = imgs[i], imgs[i+1]
        motion_matchesss.append(list(map(
            lambda matches: matchesToMotion(
                matches,
                query_kps,
                train_kps,
                query_img, train_img, i
            ), matchess)))
    return motion_matchesss


def backwardInsertSorted(l: list, e, k: callable):
    i = len(l)
    while  i > 0 and k(l[i - 1]) > k(e):
        i -= 1
    l.insert(i,e)


def matchesToMotion(matches, query_kps, train_kps, query_img, train_img, start_id):
    motion_matches = list()
    for i in range(len(matches)):
        match = matches[i]
        query_kp, train_kp = query_kps[match.queryIdx], train_kps[match.trainIdx]
        
        # Compute new distance
        distance = matchDistance(match, query_kp, train_kp, query_img, train_img)
        
        # Generate motionMatch from distance
        motion_match = motionMatch(start_id, distance, query_kp, train_kp)

        # Insert motionMatch in list in distance order (check from the end)
        backwardInsertSorted(motion_matches, motion_match, lambda m: m.match_distance)
    return motion_matches

def matchDistance(match, query_kp, train_kp, query_img, train_img):
    # TODO : Compute better match distance (Considering position, orientation, scale/size, color, ...)
    return match.distance


# TODO : degree 1 motion coherence check function

def bestMotionMatchess(motion_matchesss: List[List[List[motionMatch]]]):
    def bestMotionMatches(motion_matchess: List[List[motionMatch]]):
        return sorted(
            map(
                lambda motion_matches: motion_matches[0],
                filter(
                    lambda motion_matches: len(motion_matches) > 0,
                    motion_matchess
                )
            ), key=lambda m: m.match_distance)
    return list(map(bestMotionMatches,motion_matchesss))

#def sortMatchess(matchess, kpss, imgs):
#    return map(sortMatches, zip(matchess, kpss[:-1], kpss[1:], imgs[:-1], imgs[1:]))

#def sortMatches(x):
#    matches, query_kps, train_kps, query_img, train_img = x
#    return sorted(matches,
#    key=lambda match: matchDistance(match, query_kps, train_kps, query_img, train_img))