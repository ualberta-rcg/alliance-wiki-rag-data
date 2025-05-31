# ACPYPE: AnteChamber PYthon Parser interfacE

This is a draft, a work in progress that is intended to be published into an article, which may or may not be ready for inclusion in the main wiki. It should not necessarily be considered factual or authoritative.

## General

ACPYPE (pronounced as ace + pipe), is a tool written in Python to use Antechamber to generate topologies for chemical compounds and to interface with other Python applications like CCPN and ARIA.  It will generate topologies for CNS/XPLOR, GROMACS, CHARMM, and AMBER, that are based on General Amber Force Field (GAFF) and should be used only with compatible force fields like AMBER and its variants.

We provide Python wheels for ACPYPE for StdEnv/2020 and StdEnv/2023 in our wheelhouse that you should install into a virtual environment.

Please note that you need to load the `openbabel` module before installing ACPYPE and anytime you want to use it.


## Creating a virtual environment for ACPYPE

```bash
module load python openbabel
virtualenv ~/venv_acpype
source ~/venv_acpype/bin/activate
pip install --no-index acpype
```

Now the `acpype` command can be used:

```bash
(venv_acpype) [user@login1]$ acpype --help
```

## Using ACPYPE

### Running as a non-interactive job

You can run ACPYPE as a short job with a job script similar to the one shown below. If you have already a file with the 3D-coordinates of your molecule, you can delete the lines that use `obabel` to generate the file `adp.mol2` from the SMILES string.

**File: `job_acpype_ADP.sh`**

```bash
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
module load python openbabel
source ~/venv_acpype/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# generate "adp.mol2" file from SMILES string:
obabel -:"c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)O)O)O)N" -i smi -o mol2 -O adp.mol2 -h --gen3d

acpype -i adp.mol2
```

### Running on a login node

As part of the topology generation, ACPYPE will run a short QM calculation to optimize the structure and determine the partial charges. For small molecules, this should take less than two minutes and can therefore be done on a login-node; however, in this case, the number of threads should be limited by running ACPYPE with:  `OMP_NUM_THREADS=2 acpype ...`

For larger molecules or generating topologies for several molecules, you should submit a job as shown above. First, `python` and `openbabel` need to be loaded. We also download a structure file of Adenosine triphosphate (ATP) from PubChem:

```bash
module load python openbabel
source ~/venv_acpype/bin/activate
# download a test file for ATP:
wget "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/5957/record/SDF?record_type=3d&response_type=save&response_basename=ATP" -O atp.sdf
```

We run ACPYPE, restricting it to using a maximum of two threads:

```bash
(venv_acpype) [user@login1]$ OMP_NUM_THREADS=2 acpype -i atp.sdf
```

```
============================================================================
| ACPYPE: AnteChamber PYthon Parser interfacE v. 2023.10.27 (c) 2025 AWSdS |
============================================================================
WARNING: no charge value given, trying to guess one...
==> ... charge set to 0 == >
Executing Antechamber...
==> * Antechamber OK * == >
==> * Parmchk OK * == >
Executing Tleap...
==> * Tleap OK * == >
[ ... ]
==> Removing temporary files...
Total time of execution: 1m 32s
```

The directory `atp.acpype` is created

```bash
[name@server ~]$ ls atp.acpype/
acpype.log ANTECHAMBER_PREP.AC0 atp_CHARMM.prm atp_GMX_OPLS.top posre_atp.itp
ANTECHAMBER_AC.AC ATOMTYPE.INF atp_CHARMM.rtf atp_GMX.top rungmx.sh
ANTECHAMBER_AC.AC0 atp_AC.frcmod atp_CNS.inp atp_NEW.pdb sqm.in
ANTECHAMBER_AM1BCC.AC atp_AC.inpcrd atp_CNS.par atp.pkl sqm.out
ANTECHAMBER_AM1BCC_PRE.AC atp_AC.lib atp_CNS.top atp.sdf sqm.pdb
ANTECHAMBER_BOND_TYPE.AC atp_AC.prmtop atp_GMX.gro em.mdp
ANTECHAMBER_BOND_TYPE.AC0 atp_bcc_gaff2.mol2 atp_GMX.itp leap.log
ANTECHAMBER_PREP.AC atp_CHARMM.inp atp_GMX_OPLS.itp md.mdp
```

## Useful Links

* [Frequent Asked Questions about ACPYPE](link_to_faq)
* [Tutorial Using ACPYPE for GROMACS](link_to_gromacs_tutorial)
* [Tutorial NAMD](link_to_namd_tutorial)


**(Remember to replace `link_to_faq`, `link_to_gromacs_tutorial`, and `link_to_namd_tutorial` with the actual links.)**
