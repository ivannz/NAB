import numpy as np

from .base import covariance_inverse
from .base import EmbedderDetector


class NearestNeighboursDetector(EmbedderDetector):
    """
    The simplest k-NN detector.

    Parameters
    ----------
    n_neighbours: int, default 5
        The number of neighbours to use in computing the abnormality score.

    f_lambda: float, default 1e-4
        The covariance regularizer.

    Attributes
    ----------
    _calibration_ : list, private
        The calibration sample used for anomaly scoring.
    """

    def __init__(self, n_neighbours=5, f_lambda=1e-4, distance="mean-l1",
                 *args, **kwargs):
        super(NearestNeighboursDetector, self).__init__(*args, **kwargs)
        self.n_neighbours = n_neighbours
        self.f_lambda = f_lambda
        self.distance = distance

    def initialize(self):
        """Check the arguments for validity."""
        super(NearestNeighboursDetector, self).initialize()
        if not isinstance(self.distance, str):
            raise TypeError("""`distance` must be a string indentifier.""")

        valid_distance_ = ["l1", "mean-l1", "l2", "mean-l2"]
        if self.distance not in valid_distance_:
            raise ValueError("""`distance` must be one of [`%s`]""" % (
                             "`, `".join(valid_distance_)))

    def get_rho(self, data, X):
        """Get the average k nearest neighbour distance of `X` to `data`."""
        cov_ = np.cov(data, rowvar=False, bias=False)\
                 .reshape((data.shape[1], data.shape[1]))
        try:
            inv_ = np.linalg.inv(cov_)
        except np.linalg.LinAlgError:
            cov_.flat[::cov_.shape[0] + 1] += self.f_lambda
            inv_ = np.linalg.inv(cov_)

        inv_ /= np.linalg.norm(inv_, axis=0)

        # Get all distances from the new test point to the data
        delta_ = data - X[np.newaxis]
        nn_dist_ = np.einsum("ij,ik,jk->i", delta_, delta_, inv_)

        # Force zero distances out of the array
        index_ = np.flatnonzero(nn_dist_ > 0)
        neighbours_ = index_[nn_dist_[index_].argsort()[:self.n_neighbours]]
        if len(neighbours_) < 1:
            return 0.0

        # Get the average distance to `k` neighbours
        if self.distance == "l1":
            return np.sqrt(nn_dist_[neighbours_]).sum()
        elif self.distance == "l2":
            return np.sqrt(nn_dist_[neighbours_].sum())
        elif self.distance == "mean-l1":
            return np.sqrt(nn_dist_[neighbours_]).mean()
        elif self.distance == "mean-l2":
            return np.sqrt(nn_dist_[neighbours_].mean())

    def get_score(self, X):
        """Computes the p-value of `X[-1]` with respect to the history
        in `X[:-1]`. `X` has `n_offset_ + n_depth_ + 1` observations.
        """
        X_train = X[:-(self.n_offset_ + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if not hasattr(self, "_calibration_"):
            self._calibration_ = [self.get_rho(X_train, X[self.n_offset_ + i])
                                  for i in range(len(X_train))]
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = self.default
        if len(self._calibration_) >= self.n_depth_ + 1:
            min_, max_ = min(self._calibration_), max(self._calibration_)
            if max_ > min_:
                p_value = (rho - min_) / (max_ - min_)

        # Finalize the update of the calibration sample, by keeping it
        #  fixed-size
        if len(self._calibration_) >= self.n_depth_ + 1:
            self._calibration_.pop(0)

        return p_value


class ConformalkNNDetector(NearestNeighboursDetector):
    """A conformal k-NN detector.

    Parameters
    ----------
    method: string, default 'lazy'
        Determines the method used when computing the conformal p-value.

    Attributes
    ----------
    _calibration_ : list, private
        The calibration sample used for anomaly scoring.
    """

    def __init__(self, method="lazy", *args, **kwargs):
        super(ConformalkNNDetector, self).__init__(*args, **kwargs)
        self.method = method

    def _score_lazy(self, X):
        """Compute the lazy (offline) p-value score of `X[-1]`."""
        X_train = X[:-(self.n_offset_ + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if not hasattr(self, "_calibration_"):
            self._calibration_ = [self.get_rho(X_train, Z)
                                  for Z in X[-(self.n_offset_ + 1):-1]]
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = 0.5
        if len(self._calibration_) >= self.n_depth_ + 1:
            p_value = np.mean([score_ <= rho for score_ in self._calibration_])
            self._calibration_.pop(0)

        return p_value

    def _score_full(self, X):
        """Compute the full p-value score of `X[-1]`."""
        X_train = X[:-(self.n_offset_ + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if self.n_offset_ > 0:
            self._calibration_ = [self.get_rho(X_train, X[self.n_offset_ + i])
                                  for i in range(self.n_depth_)]
        else:
            self._calibration_ = [self.get_rho(np.delete(X, i, axis=0), X[i])
                                  for i in range(self.n_depth_)]
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = 0.5
        if len(self._calibration_) >= self.n_depth_ + 1:
            p_value = np.mean([score_ <= rho for score_ in self._calibration_])

        return p_value

    def get_score(self, X):
        """Computes the p-value of `X[-1]` with respect to the history
        in `X[:-1]`. `X` has `n_offset_ + n_depth_ + 1` observations.
        """
        # Use the chosen metohd to get the p-value
        p_value = 0.5
        if self.method == "full":
            p_value = self._score_full(X)
        elif self.method == "lazy":
            p_value = self._score_lazy(X)
        else:
            raise ValueError("""Unrecognized method.""")

        return p_value
