from .detectors import NearestNeighboursDetector


class BaseKNNDetector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=10, n_dim=10,
                 *args, **kwargs):
        super(BaseKNNDetector, self).__init__(n_neighbours=n_neighbours,
                                              f_lambda=1e-4,
                                              n_dim=n_dim,
                                              n_offset=0,
                                              n_depth="auto",
                                              *args, **kwargs)


class KNN2719Detector(BaseKNNDetector):
    def __init__(self, *args, **kwargs):
        super(KNN2719Detector, self).__init__(n_neighbours=27, n_dim=19,
                                              *args, **kwargs)


class KNN0101Detector(BaseKNNDetector):
    def __init__(self, *args, **kwargs):
        super(KNN0101Detector, self).__init__(n_neighbours=1, n_dim=1,
                                              *args, **kwargs)
