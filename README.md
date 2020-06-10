# Adaptive Neuro-fuzzy Inference System (ANFIS) for regression problems using the Takagi–Sugeno fuzzy inference system

## Reference

- Mathematical background: Jang, Sun, and Mizutani, "[Neuro-Fuzzy and Soft Computing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=633847)".

- Datasets: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Multi-input/multi-output (multivariate) adaptive neuro-fuzzy inference system implementation for regression problems using the Takagi–Sugeno fuzzy inference system.
- Quadratic cost function for continuous problems and cross-entropy cost function for classification problems.
- Classes in classification problems are determined automatically.
- Sigmoid and cross-entropy function are computed using a numerically stable implementation.
- Generalized Bell curves depending on three parameters (mean, standard deviation, and exponent) are used as premise membership functions.
- Hyperplanes depending on the number of features are used as consequent functions.
- A particle swarm optimizer (PSO) is used to solve the minimization problem. More info about it [here](https://github.com/gabrielegilardi/PSO).
- Limits/constraints on the parameter values (similar to regularization in neural networks) can be easily done through the PSO boundary arrays.
- The *ANFIS* class in *ANFIS.py* is not constrained to the PSO solver but it can be used with any other optimizer not gradient-based.
- File *utils.py* consists of several utility functions, including an helper function to build the PSO boundary arrays.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (plant, stock, wine, pulsar.)

`problem` Defines the type of problem. Equal to C specifies a classification problem, anything else specifies a continuous problem. The default value is `None`.

`split_factor` Split value between training and test data.

`data_file` File name with the dataset (csv format.)

`n_mf` List, tuple, or array with the number of premise functions of each feature.

`nPop`, `epochs` Number of agents (population) and number of iterations (PSO.)

`mu_delta` Allowed variation (plus/minus) of the mean in the premise functions. It is given as fraction of the corresponding feature data range.

`s_par` Center value and allowed variation (plus/minus) of the standard deviation in the premise functions. The center value is scaled based on the corresponding feature data range.

`c_par` Range of allowed values of the exponent in the premise functions.

`A_par` Range of allowed values of the coefficients in the consequent functions.

For the meaning of the PSO parameters `K`, `phi`, `vel_fact`, `conf_type`, `IntVar`, `normalize`, and `rad`, see [here](https://github.com/gabrielegilardi/PSO) and file *pso.py*.

## Examples

There are four examples in *test.py*: plant, stock, wine, pulsar. The values common to (most of) all examples are:

```python
problem = None
split_factor = 0.70
K = 3
phi = 2.05
vel_fact = 0.5
conf_type = 'RB'
IntVar = None
normalize = False
rad = 0.1
mu_delta = 0.2
s_par = [0.5, 0.2]
c_par = [1.0, 3.0]
A_par = [-10.0, 10.0]
```

### Single-label continuous problem example: plant

```python
data_file = 'plant_dataset.csv'
n_mf = [1, 1, 1, 1]
nPop = 40
epochs = 500
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant>.

The dataset has 4 features (inputs), 1 label (output), and 9568 samples.

The ANFIS has a layout of [1, 1, 1, 1] and 17 variables.

Correlation predicted/actual values: 0.965 (training), 0.961 (test).

### Multi-label continuous problem example: stock

```python
data_file = 'stock_dataset.csv'
n_mf = [2, 2, 2]
nPop = 100
epochs = 500
A_par = [-1.0, 1.0]         # Supersedes the default A_par
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE>.

The dataset has 3 features (inputs), 2 labels (outputs), and 536 samples.

The ANFIS has a layout of [2, 2, 2] and 82 variables.

Correlation predicted/actual values: 0.883 (training), 0.871 (test).

### Multi-class classification problem example: wine

```python
data_file = 'wine_dataset.csv'
problem = 'C'
n_mf = [3, 2]
nPop = 40
epochs = 500
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>.

The dataset has 2 features (inputs), 6 classes (outputs), and 1599 samples.

The ANFIS has a layout of [3, 2] and 123 variables.

Accuracies predicted/actual values: 58.2% (training), 59.8% (test).

### Multi-class classification problem example: pulsar

```python
data_file = 'pulsar_dataset.csv'
problem = 'C'
n_mf = [3, 4, 2]
nPop = 40
epochs = 200
```

Original dataset: <https://archive.ics.uci.edu/ml/datasets/HTRU2>.

The dataset has 3 features (inputs), 2 classes (outputs), and 17898 samples.

The ANFIS has a layout of [3, 4, 2] and 219 variables.

Accuracies predicted/actual values: 97.9% (training), 97.7% (test).

The initial and final premise membership functions for this example can be seen [here](./Code_Python/MFs_Pulsar_Example.pdf).
