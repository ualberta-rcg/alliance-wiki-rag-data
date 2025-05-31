# Large Scale Machine Learning (Big Data)

This page is a translated version of the page [Large Scale Machine Learning (Big Data)](https://docs.alliancecan.ca/mediawiki/index.php?title=Large_Scale_Machine_Learning_(Big_Data)&oldid=147804) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Large_Scale_Machine_Learning_(Big_Data)&oldid=147804), français

In the field of deep learning, scalability in terms of data quantity is made possible by the use of very small batch processing strategies and first-order iterative solvers. In deep neural network learning, the code works more or less the same way whether it involves a few thousand or hundreds of millions of examples: a few examples are loaded from a source (disk, memory, remote source, etc.), the gradients are calculated during the iterations, which modifies the model parameters as it goes. On the other hand, with several traditional machine learning packages, notably scikit-learn, writing code to perform large-scale training is often not obvious. For several algorithms that are suitable for common models such as Generalized Linear Models (GLM) and Support Vector Machines (SVM), their default implementation requires that the entire training set be loaded into memory and offers no thread or process parallelism functionality. In addition, some of these implementations rely on rather memory-intensive solvers which, to work well, require an amount of memory several times greater than the size of the data set to be trained.

We address here options for adapting traditional machine learning methods to very large datasets in cases where a Large Memory type node is insufficient or where sequential processing is excessively long.


## Contents

1. Scikit-learn
    * Stochastic Gradient Solvers
    * Batch Learning
2. Snap ML
    * Installation
        * Recently Added Wheels
        * Installing the Wheel
    * Multithreading
    * GPU Training
    * Out-of-Core Training
    * MPI
3. Spark ML


## Scikit-learn

Scikit-learn is a Python module for machine learning based on SciPy and distributed under the BSD-3-Clause license. The package has an intuitive API that simplifies the construction of complex machine learning pipelines. However, several implementations of GLM and SVM methods assume that the training set is completely loaded into memory, which is not always desirable. In addition, some of these algorithms use very memory-intensive solvers by default. In some cases, the following suggestions will allow you to circumvent these limitations.


### Stochastic Gradient Solvers

If your dataset is small enough to be fully loaded into memory, but you get Out-Of-Memory (OOM) errors during training, the problem is probably due to a memory-intensive solver. With scikit-learn, several methods offer optional variations of the stochastic gradient algorithm, and replacing the default solver with a stochastic gradient solver is often an easy solution.

In the following examples, a ridge regression uses the default solver and a stochastic gradient solver. To observe memory usage, run the `htop` command in the terminal while the Python program is running.

**File: `ridge-default.py`**

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
X, y = make_regression(n_samples=100000, n_features=10000, n_informative=50)
model = Ridge()
model.fit(X, y)
```

**File: `ridge-saga.py`**

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
X, y = make_regression(n_samples=100000, n_features=10000, n_informative=50)
model = Ridge(solver='saga')
model.fit(X, y)
```

Another option that further reduces memory usage is to work with `SGDRegressor` rather than `Ridge`. This class implements several types of generalized linear models (GLM) for regressions using the stochastic gradient algorithm (SGD) as a solver. However, it should be noted that `SGDRegressor` only works if the result is one-dimensional (scalar).

**File: `ridge-sgd_regressor.py`**

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
X, y = make_regression(n_samples=100000, n_features=10000, n_informative=50)
model = SGDRegressor(penalty='l2')
# set penalty='l2' to perform a ridge regression
model.fit(X, y)
```


### Batch Learning

In cases where your dataset is too large for the available memory, or just large enough to not leave enough memory for training, it is possible to keep the data on disk and load it in batches, as is the case with deep learning packages. Scikit-learn calls this technique out-of-core learning and it is a viable option when the estimator offers the `partial_fit` method. In the examples below, out-of-core learning is done by iterating over datasets stored on disk.

In the first example, we use `SGDClassifier` to fit a linear SVM classifier with batches of data from a pair of numpy vectors. The vectors are stored on disk in npy files that will be memory-mapped. Since `SGDClassifier` has the `partial_fit` method, iterations can be done in large memory files by loading only small batches at a time from the vectors. Each call to `partial_fit` will then execute one epoch of the stochastic gradient algorithm on a batch of data.

**File: `svm-sgd-npy.py`**

```python
import numpy as np
from sklearn.linear_model import SGDClassifier

def batch_loader(X, y, batch_size):
    return ((X[idx:idx+batch_size], y[idx:idx+batch_size]) for idx in range(0, len(X), batch_size))
# function returns a Generator

inputs = np.memmap('./x_array.npy', dtype='float64', shape=(100000, 10000))
targets = np.memmap('./y_array.npy', dtype='int8', shape=(100000,))

model = SGDClassifier(loss='hinge')
# Using loss='hinge' is equivalent to fitting a Linear SVM

for batch in batch_loader(inputs, targets, batch_size=512):
    X, y = batch
    model.partial_fit(X, y)
```

Another method of storing data is to use CSV files. In the next example, training a lasso regression model is done by batch reading data from a CSV file with the pandas package.

**File: `lasso-sgd-csv.py`**

```python
import pandas as pd
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(penalty='l1')

for batch in pd.read_csv("./data.csv", chunksize=512, iterator=True):
    X = batch.drop('target', axis=1)
    y = batch['target']
    model.partial_fit(X, y)
```


## Snap ML

Snap ML is a proprietary machine learning library developed by IBM that supports several classic models and easily adapts to datasets containing billions of examples and/or variables. It allows distributed training, GPU acceleration, and the use of sparse structures. One of its APIs is very similar to that of scikit-learn and can replace it in the case of massive datasets.


### Installation

#### Recently Added Wheels

To find out the most recent version of Snap ML that we have built, run:

```bash
[name@server ~]$ avail_wheels "snapml"
```

For more information, see [Available Wheels](link_to_available_wheels).


#### Installing the Wheel

The preferred option is to use the Python wheel as follows:

1. Load a Python module with `module load python`.
2. Create and launch a Python virtual environment.
3. Install Snap ML in the virtual environment with `pip install`.

```bash
(venv) [name@server ~]$ pip install --no-index snapml
```


### Multithreading

All Snap ML estimators support thread parallelism, which is controlled with the `n_jobs` parameter. By setting this parameter equal to the number of cores available for your task, one can typically observe an acceleration compared to the implementation of the same estimator with scikit-learn. Here's how the performance of Ridge compares between scikit-learn and Snap ML.

**File: `ridge-snap-vs-sklearn.py`**

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from snapml import LinearRegression
import time

X, y = make_regression(n_samples=100000, n_features=10000, n_informative=50)

model_sk = Ridge(solver='saga')
print("Running Ridge with sklearn...")
tik = time.perf_counter()
model_sk.fit(X, y)
tok = time.perf_counter()
print(f"sklearn took {tok - tik:0.2f} seconds to fit.")

model_snap = LinearRegression(penalty='l2', n_jobs=4)
print("Running Ridge with SnapML...")
tik = time.perf_counter()
model_snap.fit(X, y)
tok = time.perf_counter()
print(f"SnapML took {tok - tik:0.2f} seconds to fit.")
```


### GPU Training

All Snap ML estimators support acceleration of one or more GPUs. For training with a GPU, the parameter is `use_gpu=True`. For training with multiple GPUs, the parameter is also `use_gpu`, and the list of IDs of the available GPUs is passed to `device_ids`. For example, for a task that requires two GPUs, `device_ids=[0,1]` will use both GPUs. The next example makes the same comparison as in the previous section, but for training an SVM classifier with a non-linear kernel.

**File: `ridge-snap-vs-sklearn2.py`**

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from snapml import SupportVectorMachine
import time

X, y = make_classification(n_samples=100000, n_features=10000, n_classes=3, n_informative=50)

model_sk = SVC(kernel='rbf')
#sklearn's SVM fit-time scales at least quadratically with the number of samples... this will take a loooong time.
print("Running SVM Classifier with sklearn...")
tik = time.perf_counter()
model_sk.fit(X, y)
tok = time.perf_counter()
print(f"sklearn took {tok - tik:0.2f} seconds to fit.")

model_snap = SupportVectorMachine(kernel='rbf', n_jobs=4)
print("Running SVM Classifier with SnapML without GPU...")
tik = time.perf_counter()
model_snap.fit(X, y)
tok = time.perf_counter()
print(f"SnapML took {tok - tik:0.2f} seconds to fit without GPU.")

model_snap_gpu = SupportVectorMachine(kernel='rbf', n_jobs=4, use_gpu=True)
print("Running SVM Classifier with SnapML with GPU...")
tik = time.perf_counter()
model_snap_gpu.fit(X, y)
tok = time.perf_counter()
print(f"SnapML took {tok - tik:0.2f} seconds to fit with GPU.")
```


### Out-of-Core Training

All Snap ML estimators use first-order iterative solvers like SGD by default. It is therefore possible to perform batch training without having to load the entire datasets into memory. However, Snap ML accepts numpy vector inputs by memory mapping, unlike scikit-learn.

**File: `snap-npy.py`**

```python
import numpy as np
from snapml import LogisticRegression

X = np.memmap('./x_array.npy', dtype='float64', shape=(100000, 10000))
y = np.memmap('./y_array.npy', dtype='int8', shape=(100000,))

model = LogisticRegression(n_jobs=4)
model.fit(X, y)
```


### MPI

Snap ML offers distributed implementations of several estimators. To use distributed mode, call a Python script with `mpirun` or `srun`.


## Spark ML

Spark ML is a library based on Apache Spark that allows the scalability of several machine learning methods to enormous amounts of data and across multiple nodes, without having to distribute datasets or create distributed or parallel code. It includes several useful tools in linear algebra and statistics. Before reproducing the examples in the Spark ML documentation, see our tutorial on how to submit a Spark task.


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Large_Scale_Machine_Learning_(Big_Data)/fr&oldid=147804](https://docs.alliancecan.ca/mediawiki/index.php?title=Large_Scale_Machine_Learning_(Big_Data)/fr&oldid=147804)"
