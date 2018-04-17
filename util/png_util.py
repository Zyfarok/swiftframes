from typing import Tuple, List

import numpy as np
import cv2 as cv2

inPrefix = "input/"
inMidfix = " ("
inSuffix = ").png"

outPrefix = "output/"
outMidfix = " ("
outSuffix = ").png"


def readFrame(framesName: str, frameId: int,
    shapeCheck: Tuple[int,int] = (-1, -1)) -> np.ndarray:
    """Load a frame and check frame shape.

    Arguments:
        framesName {str} -- Name of the frame sequence
        frameId {int} -- Id of the frame

    Keyword Arguments:
        shapeCheck {Tuple[int,int]} -- expected shape of the frame (default: {(-1, -1)})

    Returns:
        np.ndarray -- the frame
    """
    (heightCheck, widthCheck) = shapeCheck

    img_next = cv2.imread(inPrefix + framesName + inMidfix + str(frameId) + inSuffix)

    ## TODO : Handle grayscale images ?
    (height, width, _) = img_next.shape
    if (heightCheck != -1 and height != heightCheck)\
        or (widthCheck != -1 and width != widthCheck):
        print("ERROR : The frames ", frameId, "is not of the same shape")
        exit(1)

    return img_next


def readFrames(framesName: str, start: int,
    count: int) -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """This function reads N frames.

    Arguments:
        framesName {str} -- Name of the frame sequence
        start {int} -- Id of the first frame
        count {int} -- Amount of frames to load

    Returns:
        Tuple[Tuple[int, int], List[np.ndarray]] -- The shape of the frames and the list of frames
    """
    ## Check parameters
    if count < 1:
        print("ERROR : readFrames can not read less than 1 frame")
        exit(1)

    ## Read first frame first.
    imgs = list()
    imgs.append(readFrame(framesName, start))

    ## TODO : Handle grayscale images ?
    (height, width, _) = imgs[0].shape
    shape = (height, width)

    ## Read/load any additionnal past frames.
    for i in range(start+1, start+count):
        imgs.append(readFrame(framesName, i, shape))

    return (height, width), imgs

def writeFrame(framesName: str, frameId: float,
    img: np.ndarray) -> None:
    """Write a frame to a png file.

    Arguments:
        framesName {str} -- Name of the frame sequence
        frameId {float} -- Id of the frame
        img {np.ndarray} -- frame to write
    """
    filename = outPrefix + framesName + outMidfix + str(frameId) + outSuffix
    cv2.imwrite(filename, img)
