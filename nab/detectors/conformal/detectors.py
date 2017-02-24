import numpy as np

from .base import CovarianceMixin, covariance_inverse
from .base import EmbedderDetector


class NearestNeighboursDetector(EmbedderDetector, CovarianceMixin):
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
    def __init__(self, n_neighbours=5, f_lambda=1e-4, *args, **kwargs):
        super(NearestNeighboursDetector, self).__init__(*args, **kwargs)
        self.n_neighbours = n_neighbours
        self.f_lambda = f_lambda

    def get_rho(self, data, X):
        """Get the average distance from X to its k nearest neighbours
        in the data.
        """
        cov_ = self.covariance_get(data, bias=False, update=False)
        inv_ = covariance_inverse(cov_, f_lambda=self.f_lambda)

        # Get all distances from the new test point to the data
        delta_ = data - X[np.newaxis]
        nn_dist_ = np.einsum("ij,ik,jk->i", delta_, delta_, inv_)

        # Force zero distances out of the array
        index_ = np.flatnonzero(nn_dist_ > 0)
        neighbours_ = index_[nn_dist_[index_].argsort()[:self.n_neighbours]]

        # Get the average distance to `k` neighbours
        return np.sqrt(nn_dist_[neighbours_]).mean()

    def get_score(self, X):
        """Computes the p-value of `X[-1]` with respect to the history
        in `X[:-1]`. `X` has `n_offset + n_depth + 1` observations.
        """
        X_train = X[:-(self.n_offset + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if not hasattr(self, "_calibration_"):
            self._calibration_ = []
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = 0.5
        if len(self._calibration_) >= self.n_depth + 1:
            min_, max_ = min(self._calibration_), max(self._calibration_)
            if max_ > min_:
                p_value = (rho - min_) / (max_ - min_)

        # Finalize the update of the calibration sample, by keeping it
        #  fixed-size
        if len(self._calibration_) >= self.n_depth + 1:
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
        """Computes the lazy (offline) p-value score of `X[-1]`."""
        X_train = X[:-(self.n_offset + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if not hasattr(self, "_calibration_"):
            self._calibration_ = []
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = 0.5
        if len(self._calibration_) >= self.n_depth + 1:
            p_value = np.mean([score_ <= rho for score_ in self._calibration_])
            self._calibration_.pop(0)

        return p_value

    def _score_full(self, X):
        """Computes the full p-value score of `X[-1]`."""
        X_train = X[:-(self.n_offset + 1)]
        rho = self.get_rho(X_train, X[-1])

        # Prepare the calibration sample
        if self.n_offset > 0:
            self._calibration_ = [self.get_rho(X_train, X[self.n_offset + i])
                                  for i in range(self.n_depth)]
        else:
            self._calibration_ = [self.get_rho(np.delete(X, i, axis=0), X[i])
                                  for i in range(self.n_depth)]
        self._calibration_.append(rho)

        # Use the calibration sample to get the p-value
        p_value = 0.5
        if len(self._calibration_) >= self.n_depth + 1:
            p_value = np.mean([score_ <= rho for score_ in self._calibration_])

        return p_value

    def get_score(self, X):
        """Computes the p-value of `X[-1]` with respect to the history
        in `X[:-1]`. `X` has `n_offset + n_depth + 1` observations.
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
