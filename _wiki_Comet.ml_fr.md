# Comet.ml

Comet is a machine learning meta-platform that allows you to build models for concrete applications and facilitates their development and maintenance. The platform allows you to track, compare, describe and reproduce experiments, and greatly accelerates hyperparameter search thanks to its Bayesian exploration module.


## Usage on our Clusters

### Availability

Since an internet connection is required, the use of Comet is restricted to certain clusters.

| Cluster | Availability | Comment |
|---|---|---|
| Béluga | Yes ✅ | Before using Comet, load the `httpproxy` module with `module load httpproxy`. |
| Narval | Yes ✅ | Internet connection is enabled. |
| Cedar | Yes ✅ | Internet connection is enabled. |
| Graham | No ❌ | The internet connection is disabled for the compute nodes. To work around this, see [Comet OfflineExperiment](Link-to-Comet-OfflineExperiment-page-needed). |


## Best Practices

Avoid making requests to the Comet server too frequently, as Comet may limit the throughput and make the task duration unpredictable. Interact with Comet at intervals of >= 1 minute.
