# Symbolic Algebra Software

Symbolic algebra software is a program, often accessible as an interactive environment, that is able to work directly with symbolic expressions (derivatives, integrals, and so forth) and permits exact arithmetic (e.g., `exp(-i*pi/2) = -i`) as well as other formal operations that arise in domains like number theory, group theory, differential geometry, commutative algebra, and so forth.  Most such programs also permit the use of approximate numerical calculations using floating-point numbers for handling problems that are analytically intractable.

Some well-known symbolic algebra software packages are the commercial products *Mathematica* and *Maple*, neither of which is available on our clusters, but which you can install in your home directory if your license for the software allows this.  An open-source alternative, *SageMath*, can, however, be used by loading the appropriate module:

```bash
[name@server ~]$ module load sagemath/9.3
```

Afterwards, you can then run the software interactively, e.g.,

```bash
[name@server ~]$ sage
┌────────────────────────────────────────────────────────────────────┐
│ SageMath version 9.3, Release Date: 2021-05-09                     │
│ Using Python 3.8.10. Type "help()" for help.                       │
└────────────────────────────────────────────────────────────────────┘
sage:
```

Additional open-source software that may be of interest and which is available on the clusters as a module includes the Number Theory Library (NTL) (`ntl`), SINGULAR (`singular`), Macaulay2 (`m2`), and PARI/GP (`pari-gp`).


Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Symbolic_algebra_software&oldid=125363"
