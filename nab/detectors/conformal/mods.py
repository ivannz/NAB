import numpy as np

from .base import covariance_inverse, EmbedderDetector

# 1. The calibration and train samples are less than stated by `n_dim`
# 2. The data shape matrix in the Mahalanobis distance is incorrect:
#    $(X'X)^{-1}$ instead of $\hat{\Sigma} = \frac1n X'(I_n - 1(1'1)^{-1}1')X$
# 3. the first covariance matrix is computed over a smaller sample
# 4. The train sample updates are delayed by `n_depth`. Setting `n_offset` to
#    `n_depth` yields almost identical results.
# 5. The returned NCM score is $\sum_{y\in N_k(x) \setminus\{x\}} d^2(x, y)$
#    instead of $\frac1{|N_k(x)\setminus\{x\}|}
#    \sum_{y\in N_k(x) \setminus\{x\}} d(x, y)$.


class IshimtsevDetector(EmbedderDetector):
    """This detector uses X'X instead of the sample covariance matrix.
    Updates the Mahalanobis matrix each half of the training data.
    """

    def __init__(self, n_neighbours=27, n_dim=19, f_lambda=1e-4,
                 *args, **kwargs):
        super(IshimtsevDetector, self).__init__(
            n_dim=n_dim, n_depth=None, n_offset=None,
            default=0.0, *args, **kwargs)
        self.n_neighbours = n_neighbours
        self.f_lambda = f_lambda

    def initialize(self):
        super(IshimtsevDetector, self).initialize()
        # copy the probationaryPeriod to n_depth
        self.n_depth = int(self.probationaryPeriod)
        self.n_offset = int(self.probationaryPeriod)
        self.inverse_ = np.eye(self.n_dim, dtype=np.float)

    def get_rho(self, data, X):
        """Get the average distance from X to its k nearest neighbours
        in the data.
        """
        # Get all distances from the new test point to the data
        delta_ = data - X[np.newaxis]
        nn_dist_ = np.einsum("ij,ik,jk->i", delta_, delta_, self.inverse_)

        # Force zero distances out of the array
        index_ = np.flatnonzero(nn_dist_ > 0)
        neighbours_ = index_[nn_dist_[index_].argsort()[:self.n_neighbours]]

        # Get the average distance to `k` neighbours
        return np.sqrt(nn_dist_[neighbours_]).mean()

    def get_score(self, X):
        # The current train set
        X_train = X[:-(self.n_offset + 1)]

        # Emulate Vlad's fuckups
        if (self.n_iterations_ % (int(self.probationaryPeriod) // 2)) == 0:
            cov_ = np.dot(X_train.T, X_train)\
                     .reshape((X_train.shape[1], X_train.shape[1]))
            self.inverse_ = covariance_inverse(cov_, f_lambda=self.f_lambda)

        # Prepare the calibration sample
        if not hasattr(self, "calibration_"):
            self.calibration_ = [self.get_rho(np.delete(X, i, axis=0), X[i])
                                 for i in range(self.n_depth)]

        # Use the calibration sample to get the p-value
        rho = self.get_rho(X_train, X[-1])

        self.calibration_.append(rho)
        p_value = np.mean([score_ <= rho for score_ in self.calibration_])
        self.calibration_.pop(0)

        return p_value


class Ishimtsevv2Detector(EmbedderDetector):
    """This detector uses X'X instead of the sample covariance matrix.
    Updates the Mahalanobis matrix each half of the training data.
    Uses signal pruning.
    """

    def __init__(self, n_neighbours=27, n_dim=19, f_lambda=1e-4,
                 *args, **kwargs):
        super(Ishimtsevv2Detector, self).__init__(
            n_dim=n_dim, n_depth=None, n_offset=None,
            default=0.0, *args, **kwargs)
        self.n_neighbours = n_neighbours
        self.f_lambda = f_lambda

    def initialize(self):
        super(Ishimtsevv2Detector, self).initialize()
        # copy the probationaryPeriod to n_depth
        self.n_depth = int(self.probationaryPeriod)
        self.n_offset = int(self.probationaryPeriod)
        self.inverse_ = np.eye(self.n_dim, dtype=np.float)

    def get_rho(self, data, X):
        """Get the average distance from X to its k nearest neighbours
        in the data.
        """
        # Get all distances from the new test point to the data
        delta_ = data - X[np.newaxis]
        nn_dist_ = np.einsum("ij,ik,jk->i", delta_, delta_, self.inverse_)

        # Force zero distances out of the array
        index_ = np.flatnonzero(nn_dist_ > 0)
        neighbours_ = index_[nn_dist_[index_].argsort()[:self.n_neighbours]]

        # Get the average distance to `k` neighbours
        return np.sqrt(nn_dist_[neighbours_]).mean()

    def get_score(self, X):
        # The current train set
        X_train = X[:-(self.n_offset + 1)]

        # Emulate Vlad's fuckups
        if (self.n_iterations_ % (int(self.probationaryPeriod) // 2)) == 0:
            cov_ = np.dot(X_train.T, X_train)\
                     .reshape((X_train.shape[1], X_train.shape[1]))
            self.inverse_ = covariance_inverse(cov_, f_lambda=self.f_lambda)

        # Prepare the calibration sample
        if not hasattr(self, "calibration_"):
            self.calibration_ = [self.get_rho(np.delete(X, i, axis=0), X[i])
                                 for i in range(self.n_depth)]

        # Use the calibration sample to get the p-value
        rho = self.get_rho(X_train, X[-1])

        self.calibration_.append(rho)
        p_value = np.mean([score_ <= rho for score_ in self.calibration_])
        self.calibration_.pop(0)

        # Muffler
        if not hasattr(self, "muffle_counter_"):
            self.muffle_counter_ = -1

        if self.muffle_counter_ > 0:
            self.muffle_counter_ -= 1
            p_value = 0.5
        else:
            if p_value >= 0.9965:
                self.muffle_counter_ = int(self.probationaryPeriod / 5)

        return p_value
