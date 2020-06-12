"""
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Particle Swarm Optimization.

Copyright (c) 2020 Gabriele Gilardi


X           (n_samples, n_inputs)       Input dataset (training)
Xe          (n_inputs, n_pf)            Expanded input dataset (training)
Y           (n_samples, n_outputs)      Output dataset (training)
Xp          (n_samples, n_inputs)       Input dataset (prediction)
Xpe         (n_inputs, n_pf)            Expanded input dataset (prediction)
Yp          (n_samples, n_labels)       Output dataset (prediction)
J           scalar                      Cost function
theta       (n_var, )                   Unrolled parameters
mu          (n_pf, )                    Mean (premise MFs)
s           (n_pf, )                    Standard deviation (premise MFs)
c           (n_pf, )                    Exponent (premise MFs)
A           (n_inputs+1, n_cf)          Coefficients (consequent MFs)
pf          (n_samples, n_pf)           Premise MFs value
W           (n_samples, n_cf)           Firing strenght value
Wr          (n_samples, n_cf)           Firing strenght ratios
cf          (n_samples, n_cf)           Consequent MFs value
f           (n_samples, n_outputs)      ANFIS output
combs       (n_inputs, n_cf)            Combinations of premise MFs

n_samples           Number of samples
n_inputs            Number of features in the original input dataset
n_outputs           Number of labels/classes in the output dataset
n_labels            Number of outputs in the original dataset
n_var               Number of variables
n_mf                Number of premise MFs of each feature
n_pf                Total number of premise MFs
n_cf                Total number of consequent MFs

Notes:
- MF stands for membership function.
- premise (membership) functions are generalize Bell function defined by mean
  <mu>, standard deviation <s>, and exponent <c>.
- consequent (membership) functions are hyperplanes defined by <n_inputs+1>
  coefficients each.
"""

import numpy as np
import itertools


def f_activation(z):
    """
    Numerically stable version of the sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z >= 0.0)
    a[idx] = 1.0 / (1.0 + np.exp(-z[idx]))

    idx = np.invert(idx)                # Same as idx = (z < 0.0)
    a[idx] = np.exp(z[idx]) / (1.0 + np.exp(z[idx]))

    return a


def logsig(z):
    """
    Numerically stable version of the log-sigmoid function (reference:
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec3.)
    """
    a = np.zeros_like(z)

    idx = (z < -33.3)
    a[idx] = z[idx]

    idx = (z >= -33.3) & (z < -18.0)
    a[idx] = z[idx] - np.exp(z[idx])

    idx = (z >= -18.0) & (z < 37.0)
    a[idx] = - np.log1p(np.exp(-z[idx]))

    idx = (z >= 37.0)
    a[idx] = - np.exp(-z[idx])

    return a


def build_class_matrix(Y):
    """
    Builds the output array <Yout> for a classification problem. Array <Y> has
    dimensions (n_samples, 1) and <Yout> has dimension (n_samples, n_classes).
    Yout[i,j] = 1 specifies that the i-th sample belongs to the j-th class.
    """
    n_samples = Y.shape[0]

    # Classes and corresponding number
    Yu, idx = np.unique(Y, return_inverse=True)
    n_classes = len(Yu)

    # Build the array actually used for classification
    Yout = np.zeros((n_samples, n_classes))
    Yout[np.arange(n_samples), idx] = 1.0

    return Yout, Yu


class ANFIS:

    def __init__(self, n_mf, n_outputs, problem=None):
        """
        n_mf        (n_inputs, )        Number of MFs in each feature/input
        n_outputs                       Number of labels/classes
        problem     C = classification problem, otherwise continuous problem
        """
        self.n_mf = np.asarray(n_mf)
        self.n_outputs = n_outputs
        self.problem = problem

        self.n_inputs = len(n_mf)               # Number of features/inputs
        self.n_pf = self.n_mf.sum()             # Number of premise MFs
        self.n_cf = self.n_mf.prod()            # Number of consequent MFs

        # Number of variables
        self.n_var = 3 * self.n_pf \
                     + (self.n_inputs + 1) * self.n_cf * self.n_outputs

        self.init_prob = True                   # Initialization flag
        self.Xe = np.array([])                  # Extended input array

        # For logistic regression only
        if (self.problem == 'C'):
            self.Yout = np.array([])            # Actual output
            self.Yu = np.array([])              # Class list

    def create_model(self, theta, args):
        """
        Creates the model for the regression problem.
        """
        # Unpack
        X = args[0]                 # Input dataset
        Y = args[1]                 # Output dataset

        # First time only
        if (self.init_prob):
            self.init_prob = False

            # Build all combinations of premise MFs
            self.build_combs()

            # Expand the input dataset to match the number of premise MFs.
            self.Xe = self.expand_input_dataset(X)

            # For classification initialize Yout (output) and Yu (class list)
            if (self.problem == 'C'):
                self.Yout, self.Yu = build_class_matrix(Y)

        # Builds the premise/consequent parameters mu, s, c, and A
        self.build_param(theta)

        # Calculate the output
        f = self.forward_steps(X, self.Xe)

        # Cost function for classification problems (the activation value is
        # calculated in the logsig function)
        if (self.problem == 'C'):
            error = (1.0 - self.Yout) * f - logsig(f)
            J = error.sum() / float(X.shape[0])

        # Cost function for continuous problems
        else:
            error = f - Y
            J = (error ** 2).sum() / 2.0

        return J

    def eval_data(self, Xp):
        """
        Evaluates the input dataset with the model created in <create_model>.
        """
        # Expand the input dataset to match the number of premise MFs.
        Xpe = self.expand_input_dataset(Xp)

        # Calculate the output
        f = self.forward_steps(Xp, Xpe)

        # Classification problem
        if (self.problem == 'C'):
            A = f_activation(f)
            idx = np.argmax(A, axis=1)
            Yp = self.Yu[idx].reshape((len(idx), 1))

        # Continuous problem
        else:
            Yp = f

        return Yp

    def build_combs(self):
        """
        Builds all combinations of premise functions.

        For example if <n_mf> = [3, 2], the MF indexes for the first feature
        would be [0, 1, 2] and for the second feature would be [3, 4]. The
        resulting combinations would be <combs> = [[0 0 1 1 2 2],
                                                   [3 4 3 4 3 4]].
        """
        idx = np.cumsum(self.n_mf)
        v = [np.arange(0, idx[0])]

        for i in range(1, self.n_inputs):
            v.append(np.arange(idx[i-1], idx[i]))

        list_combs = list(itertools.product(*v))
        self.combs = np.asarray(list_combs).T

    def expand_input_dataset(self, X):
        """
        Expands the input dataset to match the number of premise MFs. Each MF
        will be paired with the correct feature in the dataset.
        """
        n_samples = X.shape[0]
        Xe = np.zeros((n_samples, self.n_pf))       # Expanded array
        idx = np.cumsum(self.n_mf)
        i1 = 0

        for i in range(self.n_inputs):
            i2 = idx[i]
            Xe[:, i1:i2] = X[:, i].reshape(n_samples, 1)
            i1 = idx[i]

        return Xe

    def build_param(self, theta):
        """
        Builds the premise/consequent parameters  mu, s, c, and A.
        """
        i1 = self.n_pf
        i2 = 2 * i1
        i3 = 3 * i1
        i4 = self.n_var

        # Premise function parameters (generalized Bell functions)
        self.mu = theta[0:i1]
        self.s = theta[i1:i2]
        self.c = theta[i2:i3]

        # Consequent function parameters (hyperplanes)
        self.A = \
            theta[i3:i4].reshape(self.n_inputs + 1, self.n_cf * self.n_outputs)

    def forward_steps(self, X, Xe):
        """
        Calculate the output giving premise/consequent parameters and the
        input dataset.
        """
        n_samples = X.shape[0]

        # Layer 1: premise functions (pf)
        d = (Xe - self.mu) / self.s
        pf = 1.0 / (1.0 + (d * d) ** self.c)

        # Layer 2: firing strenght (W)
        W = np.prod(pf[:, self.combs], axis=1)

        # Layer 3: firing strenght ratios (Wr)
        Wr = W / W.sum(axis=1, keepdims=True)

        # Layer 4and 5: consequent functions (cf) and output (f)
        X1 = np.hstack((np.ones((n_samples, 1)), X))
        f = np.zeros((n_samples, self.n_outputs))
        for i in range(self.n_outputs):
            i1 = i * self.n_cf
            i2 = (i + 1) * self.n_cf
            cf = Wr * (X1 @ self.A[:, i1:i2])
            f[:, i] = cf.sum(axis=1)

        return f

    def param_anfis(self):
        """
        Returns the premise MFs parameters.
        """
        mu = self.mu
        s = self.s
        c = self.c
        A = self.A

        return mu, s, c, A
