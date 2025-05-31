# Symbolic Algebra Software

A symbolic algebra software application often functions as an interactive environment that can directly work with symbolic expressions (derivatives, integrals, etc.) and allows the use of exact arithmetic (e.g., `exp(-i*pi/2) = -i`) and other formal operations in areas such as number theory, group theory, differential geometry, commutative algebra, and so on.  Most of these applications also allow the use of approximate numerical computation with floating-point numbers to handle otherwise unsolvable problems.

Well-known applications like Mathematica and Maple are not available on our clusters but can be installed in your `/home` directory if your license allows it. You can use SageMath as an alternative, loading the module as follows:

```bash
[name@server ~]$ module load sagemath/9.3
```

You will then be able to run the application interactively:

```bash
[name@server ~]$ sage
┌────────────────────────────────────────────────────────────────────┐
│ SageMath version 9.3, Release Date: 2021-05-09                     │
│ Using Python 3.8.10. Type "help()" for help.                       │
└────────────────────────────────────────────────────────────────────┘
sage:
```

Other open-source software that may be of interest and are available as modules on our clusters include Number Theory Library (NTL) (`ntl`), SINGULAR (`singular`), Macaulay2 (`m2`), and PARI/GP (`pari-gp`).
