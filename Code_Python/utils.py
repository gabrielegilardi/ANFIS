"""
Multivariate Regression and Classification Using an Adaptive Neuro-Fuzzy
Inference System (Takagi-Sugeno) and Particle Swarm Optimization.

Copyright (c) 2020 Gabriele Gilardi
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize_data(X, param=(), ddof=0):
    """
    If mu and sigma are not defined, returns a column-normalized version of
    X with zero mean and standard deviation equal to one. If mu and sigma are
    defined returns a column-normalized version of X using mu and sigma.

    X           Input dataset
    Xn          Column-normalized input dataset
    param       Tuple with mu and sigma
    mu          Mean
    sigma       Standard deviation
    ddof        Delta degrees of freedom (if ddof = 0 then divide by m, if
                ddof = 1 then divide by m-1, with m the number of data in X)
    """
    # Column-normalize using mu and sigma
    if (len(param) > 0):
        Xn = (X - param[0]) / param[1]
        return Xn

    # Column-normalize using mu=0 and sigma=1
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=ddof)
        Xn = (X - mu) / sigma
        param = (mu, sigma)
        return Xn, param


def scale_data(X, param=()):
    """
    If X_min and X_max are not defined, returns a column-scaled version of
    X in the interval (-1,+1). If X_min and X_max are defined returns a
    column-scaled version of X using X_min and X_max.

    X           Input dataset
    Xs          Column-scaled input dataset
    param       Tuple with X_min and X_max
    X_min       Min. value along the columns (features) of the input dataset
    X_max       Max. value along the columns (features) of the input dataset
    """
    # Column-scale using X_min and X_max
    if (len(param) > 0):
        Xs = -1.0 + 2.0 * (X - param[0]) / (param[1] - param[0])
        return Xs

    # Column-scale using X_min=-1 and X_max=+1
    else:
        X_min = np.amin(X, axis=0)
        X_max = np.amax(X, axis=0)
        Xs = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
        param = (X_min, X_max)
        return Xs, param


def get_classes(Y):
    """
    Returns the number of classes (unique values) in array Y and the
    corresponding list.
    """
    class_list = np.unique(Y)
    n_classes = len(class_list)

    return n_classes, class_list


def build_classes(Y):
    """
    Builds the output array Yout for a classification problem. Array Y has
    dimensions (n_data, ) while Yout has dimension (n_data, n_classes).
    Yout[i,j] = 1 specifies that the i-th input belongs to the j-th class.
    Y can be an array of integer or an array of strings.
    """
    n_data = Y.shape[0]

    # Classes and corresponding number
    Yu, idx = np.unique(Y, return_inverse=True)
    n_classes = len(Yu)

    # Build the output array actually used for classification
    Yout = np.zeros((n_data, n_classes))
    Yout[np.arange(n_data), idx] = 1.0

    return Yout, Yu


def regression_sol(X, Y):
    """
    Returns the closed-form solution to the continuous regression problem.

    X           (m, 1+N)        Input dataset (must include column of 1s)
    Y           (m, k)          Output dataset
    theta       (1+N, k)        Regression parameters

    m = number of data in the input dataset
    N = number of features in the (original) input dataset
    k = number of labels in the output dataset
    p = number of parameters equal to (1+N) x k

    Note: each COLUMN contains the coefficients for each output/label.
    """
    theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    return theta


def calc_rmse(a, b):
    """
    Calculates the root-mean-square-error of arrays <a> and <b>. If the arrays
    are multi-column, the RMSE is calculated as all the columns are one single
    vector.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Root-mean-square-error
    rmse = np.sqrt(((a - b) ** 2).sum() / len(a))

    return rmse


def calc_corr(a, b):
    """
    Calculates the correlation between arrays <a> and <b>. If the arrays are
    multi-column, the correlation is calculated as all the columns are one
    single vector.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    corr = np.corrcoef(a, b)[0, 1]

    return corr


def calc_accu(a, b):
    """
    Calculates the accuracy (in %) between arrays <a> and <b>. The two arrays
    must be column/row vectors.
    """
    # Convert to (n, ) dimension
    a = a.flatten()
    b = b.flatten()

    # Correlation
    accu = 100.0 * (a == b).sum() / len(a)

    return accu


def info_anfis(n_mf, n_outputs):
    """
    Returns number of premise functions <n_pf>, number of consequent functions
    <n_cf>, and number of variables <n_var> for the ANFIS defined by <n_mf>
    and <n_outputs>.
    """
    n_mf = np.asarray(n_mf)

    n_pf = n_mf.sum()
    n_cf = n_mf.prod()
    n_var = 3 * n_pf + (len(n_mf) + 1) * n_cf * n_outputs

    return n_pf, n_cf, n_var


def plot_mfs(n_mf, mu, s, c, X):
    """
    Plot the generalized Bell functions defined by mu, c, and s.

    X           (n_samples, n_inputs)       Input dataset
    n_mf        (n_inputs, )                Number of MFs in each feature/input
    mu          (n_pf, )                    Mean
    s           (n_pf, )                    Standard deviation
    c           (n_pf, )                    Exponent

    n_samples           Number of samples
    n_inputs            Number of features/inputs
    n_pf                Number of premise MFs
    """
    const = 0.1                 # Plot all values from <const> to 1
    idx = np.cumsum(n_mf)
    i1 = 0

    # Loop over all features/inputs
    for j in range(len(n_mf)):

        i2 = idx[j]
        names = []

        # Loop over all MFs in the same feature/input
        for i in range(i1, i2):

            # Point where the MF is equal to <const> (wrt the mean mu)
            t_delta = s[i] * ((1.0 - const) / const) ** (1.0 / (2.0 * c[i]))
            t = np.linspace(mu[i]-t_delta, mu[i]+t_delta, num=200)

            # MF values
            d = (t - mu[i]) / s[i]
            pf = 1.0 / (1.0 + ((d ** 2.0) ** c[i]))

            names.append(str(i+1))      # Feature/input number
            plt.plot(t, pf)

        # Min. and max. values in the feature/input
        X_min = np.amin(X[:, j])
        X_max = np.amax(X[:, j])

        # Draw vertical lines to show the dataset range for the feature/input
        plt.axvline(X_min, lw=1.5, ls='--', C='k')
        plt.axvline(X_max, lw=1.5, ls='--', C='k')

        # Format and show all MFs for this feature/input
        plt.grid(b=True)
        plt.title('Feature nr. ' + str(j+1))
        plt.title('Example: pulsar')
        plt.xlabel('$X_' + str(j+1) + '$')
        plt.ylabel('$MF$')
        plt.ylim(0, 1)
        plt.legend(names)
        plt.show()
        plt.close()

        # Next feature/input
        i1 = idx[j]


def bounds_pso(X, n_mf, n_outputs, mu_delta=0.2, s_par=[0.5, 0.2],
               c_par=[1.0, 3.0], A_par=[-10.0, 10.0]):
    """
    Builds the boundaries for the PSO using a few simple heuristic rules.

    Premise parameters:
    - Means (mu) are equidistributed (starting from the min. value) along the
      input dataset and are allowed to move by <mu_delta> on each side. The
      value of <mu_delta> is expressed as fraction of the range.
    - Standard deviations (s) are initially the same for all MFs, and are given
      using a middle value <s_par[0]> and its left/right variation <s_par[1]>.
      The middle value is scaled based on the actual range of inputs.
    - Exponents (c) are initially the same for all MFs, and are given using a
      range, i.e. a min. value <c_par[0]> and a max. value <c_par[1]>.

    Consequent parameters:
    - Coefficients (A) are given using a range, i.e. a min. value <A_par[0]>
      and a max. value <A_par[1]>.
    """
    n_inputs = len(n_mf)
    n_pf, n_cf, n_var = info_anfis(n_mf, n_outputs)

    i1 = n_pf
    i2 = 2 * i1
    i3 = 3 * i1
    i4 = n_var

    LB = np.zeros(n_var)
    UB = np.zeros(n_var)

    # Premise parameters (mu, s, c)
    idx = 0
    for i in range(n_inputs):

        # Feature/input min, max, and range
        X_min = np.amin(X[:, i])
        X_max = np.amax(X[:, i])
        X_delta = X_max - X_min

        # If there is only one MF
        if (n_mf[i] == 1):
            X_step = 0.0
            X_start = (X_min + X_max) / 2.0
            s = s_par[0]

        # If there is more than one MF
        else:
            X_step = X_delta / float(n_mf[i] - 1)
            X_start = X_min
            s = s_par[0] * X_step

        # Assign values to boundary arrays LB and UB
        for j in range(n_mf[i]):
            mu = X_start + X_step * float(j)
            LB[idx] = mu - mu_delta * X_delta           # mu lower limit
            UB[idx] = mu + mu_delta * X_delta           # mu upper limit
            LB[i1+idx] = s - s_par[1]                   # s lower limit
            UB[i1+idx] = s + s_par[1]                   # s upper limit
            LB[i2+idx] = c_par[0]                       # c lower limit
            UB[i2+idx] = c_par[1]                       # c upper limit
            idx += 1

    # Consequent parameters (A)
    LB[i3:i4] = A_par[0]                # A lower limit
    UB[i3:i4] = A_par[1]                # A upper limit

    return LB, UB
