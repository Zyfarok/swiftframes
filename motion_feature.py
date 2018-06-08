import numpy as np

class MotionFeature:
    def __init__(self, pt: np.ndarray, query_pt: np.ndarray, train_pt: np.ndarray, size: float, quality: float):
        self.pt = pt
        self.query_pt = query_pt
        self.train_pt = train_pt
        self.size = size
        self.quality = quality
    def sq_dist(self, pos: np.ndarray) -> float:
        diff = pos - self.pt
        return np.dot(diff, diff)
    def query_at(self, pos: np.ndarray) -> np.ndarray:
        if(self.query_pt is None):
            return None
        else:
            return pos + self.query_pt - self.pt
    def query_pixel_at(self, pos: np.ndarray):
        pix = self.query_at(pos)
        if pix is None:
            return None
        else:
            return (int(round(pix[0])), int(round(pix[1])))
    def train_at(self, pos: np.ndarray):
        if(self.train_pt is None):
            return None
        else:
            return pos + self.train_pt - self.pt
    def train_pixel_at(self, pos: np.ndarray):
        pix = self.train_at(pos)
        if pix is None:
            return None
        else:
            return (int(round(pix[0])), int(round(pix[1])))
    #def quality_at(pos: np.ndarray):
    #    dif = self.pt - pos
    #    sq_dist = np.dot(dif, dif)
    #    return self.quality * pow(2, -sq_dist / size)
    #def quality_x(x: int):
    #    dif = self.pt[0] - x
    #    sq_dist = dif ** 2
    #    return self.quality * pow(2, -sq_dist / size)
    #def quality_y(y: int):
    #    dif = self.pt[1] - y
    #    sq_dist = dif ** 2
    #    return self.quality * pow(2, -sq_dist / size)
    #def quality_matrix(imgShape: Tuple[int, int]):
    #    pass