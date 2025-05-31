# Machine Learning Tutorial

This page is a getting started guide for running a machine learning task on one of our clusters.

## Step 1: Remove all graphical displays

Modify your program so that it does not use any graphical display. Any graphical output should be written to a file on disk and visualized on your personal computer once the task is complete. For example, if you are displaying graphs with matplotlib, you should save the graphs as files instead of displaying them on the screen.

## Step 2: Archiving a dataset

The shared storage on our clusters is not optimized to handle a large number of small files (it is optimized for very large files). Make sure that the dataset you will need for your training is in an archive file (such as "tar"), which you will transfer to your compute node at the beginning of your task.

If you don't do this, you risk causing high-frequency file reads from the storage node to your compute node, thus harming the overall performance of the system. If you want to learn more about managing large sets of files, we recommend reading [this page](link_to_page_here).

Assuming the files you need are in the `mydataset` directory:

```bash
$ tar cf mydataset.tar mydataset/*
```

The above command does not compress the data. If you think it would be appropriate, you can use `tar czf`.

## Step 3: Preparing the virtual environment

Create a virtual environment in your home space. For installation and usage details of different machine learning frameworks, refer to our documentation:

* [PyTorch](link_to_pytorch_docs)
* [TensorFlow](link_to_tensorflow_docs)

## Step 4: Interactive task (`salloc`)

We recommend that you try your task in an interactive task before submitting it with a script (next section). This will allow you to diagnose problems more quickly. Here is an example of the command to submit an interactive task:

```bash
$ salloc --account=def-someuser --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00
```

Once in the task:

1. Activate your Python virtual environment.
2. Try to run your program.
3. Install missing packages if necessary. Since the compute nodes do not have Internet access, you will need to install them from a login node. Refer to our [documentation on Python virtual environments](link_to_venv_docs) for more details.
4. Note the steps that were necessary to make your program work.
5. Now is a good time to check that your task reads and writes as much as possible to the local storage on the compute node (`$SLURM_TMPDIR`), and as little as possible on the shared file systems (home, scratch, project).

## Step 5: Scripted task (`sbatch`)

You must submit your tasks using `sbatch` scripts so that they can be fully automated. Interactive tasks are only used to prepare and debug tasks that will then be run fully and/or at large scale using `sbatch`.

### Important elements of an `sbatch` script

* **Account:** The account on which resources will be "billed".
* **Requested resources:**
    * Number of CPUs, suggestion: 6
    * Number of GPUs, suggestion: 1 (Use only one (1) GPU unless you are certain that your program uses multiple ones. By default, TensorFlow and PyTorch use a single GPU.)
    * Amount of memory, suggestion: 32000M
    * Duration (Maximum Beluga: 7 days, Graham and Cedar: 28 days)
* **Bash commands:**
    * Environment preparation (modules, virtualenv)
    * Data transfer to the compute node
    * Launching the executable

### Example script

**File:** `ml-test.sh`

```bash
#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Beluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS
module load python/3.6 cuda cudnn
SOURCEDIR=~/ml-test
# Prepare virtualenv
source ~/my_env/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.
# Prepare data
mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data
# Start training
python $SOURCEDIR/train.py $SLURM_TMPDIR/data
```

### Chunking a long task

We recommend that you chunk your tasks into 24-hour blocks. Requesting shorter tasks improves your priority. By creating a chain of tasks, it is possible to exceed the 7-day limit on Beluga.

Modify your submission script (or your program) so that your task can be interrupted and continued. Your program must be able to access the most recent checkpoint. (See the example script below.)

1. Check how many epochs (or iterations) can be performed within 24 hours.
2. Calculate how many 24-hour blocks you will need: `n_blocks = n_epochs_total / n_epochs_per_24h`
3. Use the argument `--array 1-<n_blocks>%1` to request a chain of `n_blocks` tasks.

The submission script will look like this:

**File:** `ml-test-chain.sh`

```bash
#!/bin/bash
#SBATCH --array=1-10%1   # 10 is the number of jobs in the chain
#SBATCH ...
module load python/3.6 cuda cudnn
# Prepare virtualenv
...
# Prepare data
...
# Get most recent checkpoint
CHECKPOINT_EXT='*.h5'
# Replace by *.pt for PyTorch checkpoints
CHECKPOINTS=~/scratch/checkpoints/ml-test
LAST_CHECKPOINT=$(find $CHECKPOINTS -maxdepth 1 -name "$CHECKPOINT_EXT" -print0 | xargs -r -0 ls -1 -t | head -1)
# Start training
if [ -z "$LAST_CHECKPOINT" ]; then
  # $LAST_CHECKPOINT is null; start from scratch
  python $SOURCEDIR/train.py --write-checkpoints-to $CHECKPOINTS ...
else
  python $SOURCEDIR/train.py --load-checkpoint $LAST_CHECKPOINT --write-checkpoints-to $CHECKPOINTS ...
fi
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Tutoriel_Apprentissage_machine&oldid=133700](https://docs.alliancecan.ca/mediawiki/index.php?title=Tutoriel_Apprentissage_machine&oldid=133700)"
