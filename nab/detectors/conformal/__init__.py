from .detectors import ConformalkNNDetector, NearestNeighboursDetector


class KNNDetector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=10, n_dim=10, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(KNNDetector, self).__init__(n_neighbours=n_neighbours,
                                          f_lambda=1e-4,
                                          n_dim=n_dim,
                                          n_offset=n_offset,
                                          n_depth=n_depth,
                                          *args, **kwargs)


class CAD_KNNDetector(ConformalkNNDetector):
    def __init__(self, n_neighbours=10, n_dim=10, n_depth=500,
                 *args, **kwargs):
        super(CAD_KNNDetector, self).__init__(method="full",
                                              n_neighbours=n_neighbours,
                                              f_lambda=1e-4,
                                              n_dim=n_dim,
                                              n_offset=0,
                                              n_depth=n_depth,
                                              *args, **kwargs)


class LDCD_KNNDetector(ConformalkNNDetector):
    def __init__(self, n_neighbours=10, n_dim=10, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(LDCD_KNNDetector, self).__init__(method="lazy",
                                               n_neighbours=n_neighbours,
                                               f_lambda=1e-4,
                                               n_dim=n_dim,
                                               n_offset=n_offset,
                                               n_depth=n_depth,
                                               *args, **kwargs)


class KNN_1_1_0_500Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(KNN_1_1_0_500Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


class KNN_1_20_0_500Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(KNN_1_20_0_500Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


class LDCDKNN_1_1_0_500Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(LDCDKNN_1_1_0_500Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


class LDCDKNN_1_20_0_500Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0, n_depth=500,
                 *args, **kwargs):
        super(LDCDKNN_1_20_0_500Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


class LDCDKNN_1_1_500_500Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=500, n_depth=500,
                 *args, **kwargs):
        super(LDCDKNN_1_1_500_500Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


class LDCDKNN_1_20_500_500Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=500, n_depth=500,
                 *args, **kwargs):
        super(LDCDKNN_1_20_500_500Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=n_depth, *args, **kwargs)


# New classes with adaptive `n_depth` parameter
class KNN_1_1_000Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0.0,
                 *args, **kwargs):
        super(KNN_1_1_000Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class KNN_1_1_050Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0.5,
                 *args, **kwargs):
        super(KNN_1_1_050Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class KNN_1_1_100Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=1.0,
                 *args, **kwargs):
        super(KNN_1_1_100Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_1_000Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0.0,
                 *args, **kwargs):
        super(LDCDKNN_1_1_000Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_1_050Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=0.5,
                 *args, **kwargs):
        super(LDCDKNN_1_1_050Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_1_100Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=1, n_offset=1.0,
                 *args, **kwargs):
        super(LDCDKNN_1_1_100Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)


class KNN_1_20_000Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0.0,
                 *args, **kwargs):
        super(KNN_1_20_000Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class KNN_1_20_050Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0.5,
                 *args, **kwargs):
        super(KNN_1_20_050Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class KNN_1_20_100Detector(NearestNeighboursDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=1.0,
                 *args, **kwargs):
        super(KNN_1_20_100Detector, self).__init__(
            n_neighbours=n_neighbours, f_lambda=1e-4, n_dim=n_dim,
            n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_20_000Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0.0,
                 *args, **kwargs):
        super(LDCDKNN_1_20_000Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_20_050Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=0.5,
                 *args, **kwargs):
        super(LDCDKNN_1_20_050Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)


class LDCDKNN_1_20_100Detector(ConformalkNNDetector):
    def __init__(self, n_neighbours=1, n_dim=20, n_offset=1.0,
                 *args, **kwargs):
        super(LDCDKNN_1_20_100Detector, self).__init__(
            method="lazy", n_neighbours=n_neighbours, f_lambda=1e-4,
            n_dim=n_dim, n_offset=n_offset, n_depth=None, *args, **kwargs)
