# Modules

In computing, a module is a unit of software designed to be independent, interchangeable, and contains everything necessary to provide the desired functionality.<sup>[1]</sup> The term "module" can have a more specific meaning depending on the context. This page describes a few types of modules and suggests links to further documentation.

## Disambiguation

### Lmod Modules

Also called "environment modules," Lmod modules alter your (shell) environment to enable using a particular software package or a non-default version of common software packages like compilers. See [Using modules](link-to-using-modules-page).

### Python Modules

In Python, a module is a code file (usually Python code) loadable with the `import ...` or `from ... import ...` statements to provide functionality. A Python package is a collection of Python modules; the terms "package" and "module" are often interchanged casually.<sup>[2]</sup>

Frequently used Python modules like NumPy can be imported after loading the `scipy-stack` Lmod module at the shell level. See [SciPy stack](link-to-scipy-stack-page) for details.

We maintain a large collection of Python "wheels." These are pre-compiled modules compatible with the [Standard software environments](link-to-standard-software-environments-page).  Before importing modules from our wheels, create a [virtual environment](link-to-virtual-environment-page).

Python modules not in the `scipy-stack` Lmod module or our wheels collection can be installed from the internet as described in the [Installing packages](link-to-installing-packages-page) section.


## Other Related Topics

The main [Available software](link-to-available-software-page) page is a good starting point. Other related pages are:

*   [Standard software environments](link-to-standard-software-environments-page): As of April 1, 2021, `StdEnv/2020` is the default collection of Lmod modules.
*   Lmod modules specific to Niagara (link-to-niagara-modules-page)
*   Tables of Lmod modules optimized for AVX, AVX2, and AVX512 CPU instructions (link-to-avx-modules-page)
*   [Category Software](link-to-software-category-page): A list of different software pages in this wiki, including commercial or licensed software.

## Footnotes

1.  Wikipedia, "Modular programming"
2.  Tutorialspoint.com, "What is the difference between a python module and a python package?"


**(Remember to replace the bracketed `link-to-...-page` placeholders with the actual links.)**
