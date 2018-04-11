from typing import Tuple, List

import numpy as np
import cv2 as cv2

inPrefix = "input/"
inMidfix = " ("
inSuffix = ").png"


def readFrame(framesName: str, frameId: int, shapeCheck: Tuple[int,int] = (-1, -1))\
     -> np.ndarray:
    """
    Load a frame and check frame shape.

    :param framesName: Name of the frame sequence.
    :param frameId: Id of the frame.
    :param shapeCheck: expected shape of the frame
    :return: the frame.
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


def readFrames(framesName: str, start: int, count: int)\
        -> Tuple[Tuple[int, int], List[np.ndarray]]:
    """
    This function reads N frames,
        previousId being the id of the last past frame.

    :param framesName: Name of the frame sequence.
    :param start: Id of the first frame.
    :param count: Amount of frames to load. (minimum 1)
    :return: The shape of the frames and the list of frames.
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

def writeFrame(framesName: str, frameId: float, img: np.ndarray) -> None:
    """
    Write a frame to png file.

    :param framesName: Name of the frame sequence.
    :param frameId: Id of the frame.
    :param img: frame to write.
    :return: file name
    """
    filename = framesName + " (" + str(frameId) + ").png"
    cv2.imwrite(filename, img)
