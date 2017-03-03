import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..base import AnomalyDetector


class HistoryMixin(object):
    """A mixin class to record and access the history of observations.

    Attributes
    ----------
    _history_data_ : np.ndarray, private
        The storage ussed to keep history.

    _history_size_ : int, private
        The current length of the recorded history.
    """

    def history_record(self, X):
        """Add a new observation to the history."""
        if not hasattr(self, "_history_size_"):
            self._history_data_ = np.empty((1024,), np.float)
            self._history_size_ = 0

        while self._history_size_ >= len(self._history_data_):
            shape_ = (2 * len(self._history_data_),) + \
                self._history_data_.shape[1:]
            self._history_data_.resize(shape_, refcheck=True)

        self._history_data_[self._history_size_] = X
        self._history_size_ += 1

    def history_get(self, writeable=False):
        """Get a view into the current history."""
        view_ = self._history_data_[:self._history_size_]
        view_.setflags(write=writeable)
        return view_

    @staticmethod
    def history_sliding_window(array, n_dim):
        """A read-only view into a sliding window over the first dimension."""
        if n_dim < 1:
            raise ValueError("""Zero-dimensional embedding is not allowed.""")

        shape_ = (array.shape[0] - n_dim + 1, n_dim,) + array.shape[1:]
        strides_ = (array.strides[0], array.strides[0],) + array.strides[1:]
        return as_strided(array, writeable=False,
                          shape=shape_, strides=strides_)


class EmbedderDetector(AnomalyDetector, HistoryMixin):
    """A base class for detectors, which require historical sliding windows.

    Parameters
    ----------
    n_dim: int, default 5
        The dimension to embed the data observation into.

    n_offset: int, default 0
        A non-negative offset of the train sample from the latest observation.

    n_depth: None, "auto", or int (default 500)
        The size of the sample of embedded observations to use in detecting.
        Setting to `auto` uses the `probationaryPeriod` value instead.
        Setting to `None` means that the full history is used.

    default: float (default 0.5)
        The default value to return when the history is incomplete, or
        embedding is impossible.

    Parameters of AnomalyDetector
    -----------------------------
    dataSet : corpus.DataFile
        The unvariate dataset file from the NAB corpus.

    probationaryPeriod : float
        The share of the dataset to use as burn-in or pre-training for
        the detector.

    Attributes
    ----------
    inputMin : float
        the smallest value in the dataset.

    inputMax : float
        the largest value in the dataset.
    """

    def __init__(self, n_dim=5, n_offset=0, n_depth="auto", default=0.5,
                 *args, **kwargs):
        super(EmbedderDetector, self).__init__(*args, **kwargs)
        self.n_depth = n_depth
        self.n_offset = n_offset
        self.n_dim = n_dim
        self.default = default

    def initialize(self):
        """Check the arguments for validity."""
        if isinstance(self.n_depth, int):
            if self.n_depth < 0:
                raise ValueError("""`n_depth` must specify a non-negative """
                                 """size of the training history.""")
            self.n_depth_ = self.n_depth
        elif isinstance(self.n_depth, str):
            if self.n_depth not in ("auto",):
                raise ValueError("""Any value other than 'auto' is """
                                 """prohibited in `n_depth`.""")
            self.n_depth_ = int(self.probationaryPeriod)
        elif self.n_depth is None:
            self.n_depth_ = None
        else:
            raise ValueError("""`n_depth` can be either 'auto', a positive """
                             """integer, or `None`.""")

        if isinstance(self.n_offset, float):
            if not (0 <= self.n_offset <= 1):
                raise ValueError("""The value of the fractional `n_offset` """
                                 """must be within [0, 1].""")
            self.n_offset_ = int(self.n_offset * self.n_depth)
        else:
            if not(isinstance(self.n_offset, int) and self.n_offset >= 0):
                raise ValueError("""`n_offset` must specify a non-negative """
                                 """horizon.""")
            self.n_offset_ = self.n_offset

        if not (isinstance(self.n_dim, int) and self.n_dim >= 1):
            raise ValueError("""`n_dim` must specify a valid sliding window """
                             """width in observations.""")

    def handleRecord(self, inputData):
        """The interface function to process the next element in the stream."""
        if not hasattr(self, "n_iterations_"):
            self.n_iterations_ = 0
        self.n_iterations_ += 1

        # Handle the new observation
        self.history_record(inputData["value"])

        # Check if there is enough history for embedding
        history_ = self.history_get(writeable=False)
        if len(history_) < self.n_dim:
            return (self.default,)
        history_ = self.history_sliding_window(history_, self.n_dim)

        # Ensure that the required depth of history is available
        if isinstance(self.n_depth_, int):
            if len(history_) < self.n_offset_ + self.n_depth_ + 1:
                return (self.default,)
            X = history_[-(self.n_offset_ + self.n_depth_ + 1):]
        else:
            if len(history_) < self.n_offset_ + 1:
                return (self.default,)
            X = history_

        # p-value the current observation in X[-1] over the history in X[:-1]
        return (self.get_score(X),)

    def get_score(self, X):
        """Compute the p-value score.

        Computes the p-value of `X[-1]` with respect to the history in
        `X[:-1]`. `X` has `n_offset + n_depth + 1` observations, if `n_depth`
        is not `None`, or at least `n_offset + 1` observations otherwise.

        This method is free to create and maintain extra bookeeping variables
        for scroing.
        """
        if self.n_depth is None:
            X_train = X[:-(self.n_offset_ + 1)]
        else:
            X_train = X[:self.n_depth_]

        # in fact both views are exactly the same if n_depth_ is > 0
        raise NotImplementedError("""Child classes must reimplement this.""")


def covariance_inverse(cov, f_lambda=1e-4):
    """Compute the inverse of the positive difinite matrix."""
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov.flat[::cov.shape[0] + 1] += f_lambda
        return np.linalg.inv(cov)


class CovarianceMixin(object):
    """A mixin class to compute the covariance matrix."""

    @staticmethod
    def covariance_get(X, bias=False, update=None):
        """Return the current covariance matrix."""
        return np.cov(X, rowvar=False, bias=bias)\
                 .reshape((X.shape[1], X.shape[1]))


class IterativeCovarianceMixin(object):
    """A mixin for online covariance computation."""

    def covariance_update(self, X, bias=None, update=False):
        """Update the current covariance matrix with the data in X."""
        mu_m, m = X.mean(axis=0, keepdims=True), X.shape[0]
        sigma_m = np.cov(X, rowvar=False, bias=True)\
                    .reshape((X.shape[1], X.shape[1]))

        if update and hasattr(self, "_covariance_n_samples_"):
            self._covariance_n_samples_ += m

            delta_ = mu_m - self._covariance_mean_
            q = m / float(self._covariance_n_samples_)

            self._covariance_mean_ += q * delta_

            self._covariance_ += q * (sigma_m - self._covariance_ +
                                      (1 - q) * np.dot(delta_.T, delta_))
        else:
            self._covariance_mean_ = mu_m
            self._covariance_ = sigma_m
            self._covariance_n_samples_ = m

    def covariance_get(self, X=None, bias=False, update=False):
        """Return the current covariance matrix."""
        if X is not None:
            self.covariance_update(X, update=update)

        cov_ = self._covariance_.copy()
        if not bias:
            cov_ *= self._covariance_n_samples_ / \
                (self._covariance_n_samples_ - 1.0)
        return cov_
