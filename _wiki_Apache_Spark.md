# Apache Spark

## Introduction

Apache Spark is an open-source distributed computing framework initially developed by AMPLab at the University of Berkeley, and now a project of the Apache foundation. Unlike the MapReduce algorithm implemented by Hadoop, which uses disk storage, Spark uses in-memory primitives, allowing it to achieve performance up to 100 times faster for certain applications. Loading data into memory allows for frequent querying, making Spark a particularly suitable framework for machine learning and interactive data analysis.


## Utilisation

### PySpark

**File:** `pyspark_submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=00:01:00
#SBATCH --nodes=4
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
module load spark/2.3.0
module load python/3.7
# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $(( ${SLURM_MEM_PER_NODE} * 95 / 100 )))
start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)
NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
SPARK_NO_DAEMONIZE=1
srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!
srun -n 1 -N 1 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M $SPARK_HOME/examples/src/main/python/pi.py
kill $slaves_pid
stop-master.sh
```

### Java Jars

**File:** `pyspark_java_submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=00:01:00
#SBATCH --nodes=4
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
module load spark/2.3.0
# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $(( ${SLURM_MEM_PER_NODE} * 95 / 100 )))
start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)
NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
SPARK_NO_DAEMONIZE=1
srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!
SLURM_SPARK_SUBMIT="srun -n 1 -N 1 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M"
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkPi $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkLR $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
kill $slaves_pid
stop-master.sh
```

## Monitoring

The logs of the executed Spark application can be saved and viewed later using a web application provided with Spark. The following instructions show how to enable log saving and start the web application.

### Configuration

First, create a directory that will contain the application logs:

```bash
[name@server ~]$ mkdir ~/.spark/<spark version>/eventlog
```

Then, if it doesn't already exist, create a directory that will contain the Spark configuration parameters:

```bash
[name@server ~]$ mkdir ~/.spark/<spark version>/conf
```

In this directory, create the following file or add the content shown to the `spark-defaults.conf` file if it already exists.

**File:** `spark-defaults.conf`

```
spark.eventLog.enabled true
spark.eventLog.dir /home/<userid>/.spark/<spark version>/eventlog
spark.history.fs.logDirectory  /home/<userid>/.spark/<spark version>/eventlog
```

### Visualisation

Create a tunnel between your computer and the compute cluster.

Load the Spark module:

```bash
[name@server ~]$ module load spark/2.3.0
```

Start the log visualization web application:

```bash
[name@server ~]$ SPARK_NO_DAEMONIZE=1 start-history-server.sh
```

The following output will be displayed in the terminal:

```
starting org.apache.spark.deploy.history.HistoryServer, logging to /home/<userid>/.spark/<spark version>/log/spark-<userid>-org.apache.spark.deploy.history.HistoryServer-1-<server>.computecanada.ca.out
Spark Command: /cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/java/1.8.0_121/bin/java -cp /home/<userid>/.spark/<spark version>/conf/:/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/spark/2.2.0/jars/* -Xmx1g org.apache.spark.deploy.history.HistoryServer
========================================
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
17/10/13 04:28:56 INFO HistoryServer: Started daemon with process name: 71616@<server>.computecanada.ca
17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for TERM
17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for HUP
17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for INT
17/10/13 04:28:56 INFO SecurityManager: Changing view acls to: <userid>
17/10/13 04:28:56 INFO SecurityManager: Changing modify acls to: <userid>
17/10/13 04:28:56 INFO SecurityManager: Changing view acls groups to: 
17/10/13 04:28:56 INFO SecurityManager: Changing modify acls groups to: 
17/10/13 04:28:56 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(<userid>); groups with view permissions: Set(); users with modify permissions: Set(<userid>); groups with modify permissions: Set()
17/10/13 04:28:56 INFO FsHistoryProvider: History server ui acls disabled; users with admin permissions: ; groups with admin permissions
17/10/13 04:29:01 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
17/10/13 04:29:02 INFO FsHistoryProvider: Replaying log path: file:/home/<userid>/.spark/<spark version>/eventlog/app-20171013040359-0000
17/10/13 04:29:02 INFO Utils: Successfully started service on port 18080.
17/10/13 04:29:02 INFO HistoryServer: Bound HistoryServer to 0.0.0.0, and started at http://<server ip address>:18080
```

Copy the URL displayed in the terminal and paste it into your web browser.

To stop the visualization application, press Ctrl-C in the terminal used to launch the application.
