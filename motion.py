from typing import List, Tuple, Iterable, Optional
import numpy as np
import cv2 as cv
import numpy as np
import time
from util.collection_util import *
from functools import reduce
from motion_match import MotionMatch, KP

#### Detectors, fastest first. (fast is a bit faster but agast is a bit better. Orb offers sparse but good quality features and surf is very good but very slow) :
#fast = cv.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True, type=cv.FastFeatureDetector_TYPE_9_16)
agast = cv.AgastFeatureDetector_create(threshold=5, nonmaxSuppression=True, type=cv.AgastFeatureDetector_OAST_9_16)
orb_dtc = cv.ORB_create(edgeThreshold=24, patchSize=31, nlevels=3, fastThreshold=8, scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=6000)
## SURF Defaults : hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False
#surf_dtc = cv.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=4, nOctaveLayers=3, extended=False, upright=True)
default_dtc = agast

#### Descriptors, Matchers, and match-distance result scale (worst to best. orb and brief are good choices)
surf_dsc = (cv.xfeatures2d.SURF_create(hessianThreshold=10,nOctaves=4,nOctaveLayers=3,extended=False,upright=True),
            cv.BFMatcher_create(normType=cv.NORM_L1), 1 / (1))
orb_dsc = (cv.ORB_create(edgeThreshold=24, patchSize=31, nlevels=8, fastThreshold=14, scaleFactor=1.2, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=6000),
            cv.BFMatcher_create(normType=cv.NORM_HAMMING), 1 / (18))
brief = (cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False),
            cv.BFMatcher_create(normType=cv.NORM_HAMMING), 1 / (12))
sift = (cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=100.0, sigma=1.6),
            cv.BFMatcher_create(normType=cv.NORM_L1), 1 / (500))
default_dsc = brief

def computeMotion(imgs: Iterable[np.ndarray], degree: int = 1, detector: cv.Feature2D = default_dtc,
        descriptor: cv.Feature2D = default_dsc[0], matcher: cv.DescriptorMatcher = default_dsc[1],
        distscale: float = default_dsc[2], dist_thresh: float = 1.5, better_thresh: float = 1.5
        ) -> Tuple[List[List[MotionMatch]], ...]:
    assert isinstance(degree, int), "degree should be an int"
    assert degree > 0 and degree < 5, "degree should be between 1 and 4"

    ## Detect features and Compute descriptors :
    #start_time = time.time()
    kpss = list()
    descss = list()
    for img in imgs:
        kps, descs = descriptor.compute(img, detector.detect(img, None))
        kpss.append(kps)
        descss.append(descs)
    #kpss, descss = descriptor.compute(imgs, detector.detect(imgs, None))
    # TODO : Compute RGB colors at keypoints ?
    #print("detect time :", time.time() - start_time)

    for i, kps in enumerate(kpss):
        print(i, 'kps len : ', len(kps))

    #### Match features
    print("Matching")
    matchess = matchFeaturess(descss, matcher, k=2)
    #test = list(sorted(map(lambda x: x[0].distance, matchesss[0])))
    #print(np.percentile(test, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

        
    # Translate matches to motionMatches
    print("Translating to motion matches")
    motion_matchess = computeMotionMatchess(matchess, kpss, distscale=distscale)
    size = 2
    print("Merging motion matches")
    for i in range(degree - 1):
        print("degree :", i + 2)
        motion_matchess = mergeMotionMatchess(motion_matchess, i + 2, dist_thresh * 3)
        size += 1


    print("select best motionmatches and return")
    # TODO : Handle 
    return (bestMotionMatchess(motion_matchess, dist_thresh, better_thresh=better_thresh, size=size),)
    

def getMotionFeatures(at_pos: int, at_time: float, mms: Iterable[MotionMatch], verbose=False):
    return [mm.genMotionFeature(at_pos, at_time, verbose=verbose) for mm in mms]

def matchFeaturess(descss: list, matcher: cv.DescriptorMatcher, k: int = 3):
    # TODO : Improve matching speed (divide in areas)
    matchess = (
        [
            match
            for matches in matcher.knnMatch(query_descs, train_descs, k)
            for match in matches
        ]
        for query_descs, train_descs in zip(descss[:-1], descss[1:])
    )
    return matchess

def matchToMotion(match, start_id, query_kps, train_kps, distscale=1.0):
    query_idx = match.queryIdx
    train_idx = match.trainIdx
    kp_pts = (KP(query_kps[query_idx]), KP(train_kps[train_idx]))
    kp_ids = (query_idx, train_idx)
    
    distance = match.distance * distscale
    distance *= distance

    return MotionMatch(start_id, kp_pts, kp_ids, match_distances=(distance,))

def computeMotionMatchess(matchess, kpss, distscale=1.0):
    motion_matchess = (
        list(map(
            lambda match, i=i: matchToMotion(
                match,
                i,
                kpss[i],
                kpss[i+1],
                distscale=distscale
            ), matches))
        for i, matches in enumerate(matchess)
    )
    return motion_matchess

def mergeMotionMatchess(motion_matchess: Iterable[List[MotionMatch]], degree: int, goodEnoughThreshold: float = 0):
    def mergeMotionMatches(query_motion_matches: List[MotionMatch],
    train_motion_matches: List[MotionMatch]):
        groupedTrain = groupbydefault(train_motion_matches, lambda match: match.startKeypointIds())
        if goodEnoughThreshold != 0:
            return list(filter(lambda mm: mm.final_distance() < goodEnoughThreshold,
                (
                    query.merge(train)
                    for query in query_motion_matches
                    for train in groupedTrain[query.endKeypointIds()]
                )))
        else:
            return [
                query.merge(train)
                for query in query_motion_matches
                for train in groupedTrain[query.endKeypointIds()]
            ]
    query = next(motion_matchess)
    for i, train in enumerate(motion_matchess):
        print("degree", degree, "merge :", i)
        yield mergeMotionMatches(query, train)
        query = train
    print("finished merging.")

def bestMotionMatchess(motion_matchess: Iterable[List[MotionMatch]], dist_thresh: float, better_thresh: float = 1.0, size: int = 2):
    def secondBestDist(motion_matches: Iterable[MotionMatch]):
        bestDist = float('inf')
        secondBestDist = bestDist
        for match in motion_matches:
            dist = match.final_distance()
            if(dist < secondBestDist):
                if(dist < bestDist):
                    secondBestDist = bestDist
                    bestDist = dist
                else:
                    secondBestDist = dist
        return secondBestDist

    def bestMotionMatches(motion_matches: List[MotionMatch]):
        print("start : ", len(motion_matches))
        groupedSecondBestDist = [
            {k: secondBestDist(v) for k, v in groupby(motion_matches, key=lambda match, i=i: match.kp_ids[i])}
            for i in range(size)
        ]
        l = list(filter(
            lambda match: all(
                [match.final_distance() * better_thresh < groupedSecondBestDist[i][match.kp_ids[i]]
                    for i in range(size)]
                ),
            motion_matches
        ))
        print("best : ", len(l))
        return l

    def goodMotionMatches(motion_matches: Iterable[MotionMatch]):
        l = list(sorted(
            filter(
                lambda motion_match: motion_match.final_distance() < dist_thresh,
                motion_matches
            ), key=lambda m: m.final_distance()))
        print("good : ", len(l))
        return l
    best = map(bestMotionMatches, motion_matchess)
    good = list(map(goodMotionMatches, best))
    return good