# MRCC

## Introduction

MRCC is a suite of ab initio and density functional quantum chemistry programs for high-accuracy electronic structure calculations developed and maintained by the quantum chemistry research group at the Department of Physical Chemistry and Materials Science, TU Budapest. Its special feature, the use of automated programming tools, enabled the development of tensor manipulation routines independent of the number of indices of the corresponding tensors, thus significantly simplifying the general implementation of quantum chemical methods. Applying the automated tools of the program, several quantum chemistry models and techniques of high complexity have been implemented so far, including arbitrary single-reference coupled-cluster (CC) and configuration interaction (CI) methods, multi-reference CC approaches, CC and CI energy derivatives and response functions, and arbitrary perturbative CC approaches. Many features of the package are also available with relativistic Hamiltonians, allowing for accurate calculations on heavy element systems. The developed cost-reduction techniques and local correlation approaches also enable high-precision calculations for medium-sized and large molecules.

## License Limitations

The Alliance has signed a license agreement with Prof. Dr. Mihaly Kallay, who acts for the developers of the MRCC Software.

To use the current installed version on the Alliance systems, each user must agree to certain conditions. Please contact support with a copy of the following statement:

1. I will use MRCC only for academic research.
2. I will not copy the MRCC software, nor make it available to anyone else.
3. I will properly acknowledge original papers related to MRCC and to the Alliance in my publications. For more details: [https://www.mrcc.hu/index.php/citation](https://www.mrcc.hu/index.php/citation)
4. I understand that the agreement for using MRCC can be terminated by either the MRCC developers or the Alliance.
5. I will notify the Alliance of any change in the above acknowledgment.

## Module

The MRCC version from "2023-08-28" is available on all clusters by loading a module:

```bash
module load intel/2023.2.1
module load openmpi/4.1.5
module load mrcc/20230828
```

The module was installed with OpenMP and MPI support. Once the module is loaded, you can access all the binaries and the basis. The list of binaries is:

```
$ ls $EBROOTMRCC/bin/
ccsd  ccsd_mpi  dirac_mointegral_export  dmrcc  dmrcc_mpi  drpa  drpa_mpi  goldstone  integ  minp  mp2f12  mrcc  mrcc_mpi  mulli  ovirt  orbloc  prop  qmmod  scf  scf_mpi  uccsd  xmrcc  xmrcc_mpi
```

## Examples and Job Scripts

Coming soon

## Citations

As indicated in the license, users are asked to cite the original papers in their publications. For more information, please see this page.

## Documentation

Detailed documentation about the usage of the program is available on the [MRCC website](https://www.mrcc.hu/). Another useful source of information about the program is the [MRCC forum](https://www.mrcc.hu/index.php/forum).
