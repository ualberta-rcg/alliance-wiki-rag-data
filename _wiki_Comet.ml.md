# Comet

Comet is a meta machine learning platform designed to help AI practitioners and teams build reliable machine learning models for real-world applications by streamlining the machine learning model lifecycle. By using Comet, users can track, compare, explain and reproduce their machine learning experiments. Comet can also greatly accelerate hyperparameter search, by providing a module for the Bayesian exploration of hyperparameter space.

## Using Comet on our clusters

### Availability

Since it requires an internet connection, Comet has restricted availability on compute nodes, depending on the cluster:

| Cluster | Availability | Note |
|---|---|---|
| Béluga | Yes ✅ | Comet can be used after loading the `httpproxy` module:  `module load httpproxy` |
| Narval | Yes ✅ | internet access is enabled |
| Cedar | Yes ✅ | internet access is enabled |
| Graham | No ❌ | internet access is disabled on compute nodes. Workaround: Comet OfflineExperiment |

### Best practices

Avoid logging metrics (e.g., loss, accuracy) at a high frequency. This can cause Comet to throttle your experiment, which can make your job duration harder to predict. As a rule of thumb, please log metrics (or request new hyperparameters) at an interval >= 1 minute.
