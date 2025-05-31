# Open Babel

This page is a translated version of the page Open Babel and the translation is 100% complete.

Other languages: English, français

## Description

Open Babel is a toolbox designed to speak the many languages of chemical data. It is an open and collaborative project allowing anyone to search, convert, analyze, or store data from molecular modeling, chemistry, solid materials, biochemistry, or related fields.

See the [Open Babel User Guide](link_to_user_guide_here).


Two types of modules are installed on our clusters:

### `openbabel`

This sequential version can be safely used even on login nodes to convert the formats of chemical structure files. In most cases, this is the right module.

#### Example

```bash
[user@login1]
$ module load openbabel
[user@login1]
$ wget "https://www.chemspider.com/FilesHandler.ashx?type=str&3d=yes&id=171" -O acetic_acid.mol
[user@login1]
$ obabel -i mol acetic_acid.mol -o pdb -O acetic_acid.pdb
```

**Remarks:**

* The `wget` command downloads the `acetic_acid.mol` file.
* The `obabel` command converts the molecule described in `acetic_acid.mol` from the `.mol` format to the `.pdb` format.


### `openbabel-omp`

This version offers parallelization with OpenMP.  Do not use this module on login nodes, as even for simple tasks it will create as many threads as it detects CPUs on the machine, thus causing load spikes that will disrupt other users.

The parallel version is useful for converting a very large number of molecular structures or calculating a large number of chemoinformatics descriptors for multiple molecules. Make sure to set the environment variable `OMP_NUM_THREADS` to tell Open Babel how many CPUs it can use.

#### Example

The next task uses the SDF file `many_molecules.sdf` which should contain a database of several molecules and generates canonical SMILES representations for each of them, using two CPU cores.

**File: `parallel_openbabel_job.sh`**

```bash
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
module load openbabel-omp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
obabel -i sdf many_molecules.sdf -o can -O many_canonical_smiles.txt
```

## Python

Open Babel functionalities can be used from other languages such as Python. The Python interface for Open Babel is added to the `openbabel` e.g. `openbabel-omp` modules as extensions. Therefore, the `openbabel` and `pybel` packages can be used after loading `openbabel` and a compatible Python module.

#### Example

```bash
$ module load python/3.11 openbabel/3.1.1
$ python
Python 3.11.5 (main, Sep 19 2023, 19:49:15) [GCC 11.3.0] on linux
>>> import openbabel
>>> print(openbabel.__version__)
3.1.1.1
>>> from openbabel import pybel
>>>
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Open_Babel/fr&oldid=150870](https://docs.alliancecan.ca/mediawiki/index.php?title=Open_Babel/fr&oldid=150870)"
