from typing import Tuple, List, Iterable

import numpy as np
import cv2 as cv2

def readFrames(fileName: str) -> Tuple[Tuple[int, int], Iterable[np.ndarray], float, int]:
    """This function reads frames from video file.
    """

    cap = cv2.VideoCapture(fileName)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def frames():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            yield frame
        cap.release()

    return frames(), (height, width), fps, length

def writeFrames(fileName: str, shape: Tuple[int, int], frames: Iterable[np.ndarray], fps: float) -> None:
    """Write frames to a video file.
    """
    (height, width) = shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fileName,fourcc, fps, (width,height))

    for frame in frames:
        out.write(frame)

    out.release()