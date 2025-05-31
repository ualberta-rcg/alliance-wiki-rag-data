# Modules

This page is a translated version of the page Modules and the translation is 100% complete.

Other languages: English français

In programming, a module is an independent and interchangeable software component that contains everything needed to provide a specific functionality.<sup>[1]</sup>  Depending on the context, the term *module* can have different meanings. We describe here some types of modules and suggest other documentation references.


## Contents

* [Precision](#precision)
    * [Lmod Modules](#lmod-modules)
    * [Python Modules](#python-modules)
* [Additional Information](#additional-information)
* [References](#references)


## Precision

### Lmod Modules

Also called *environment modules*, Lmod modules are used to modify your environment (shell) to allow the use of a software package or a version of software other than the one offered by default, for example for compilers (see [Using Modules](Using Modules)).


### Python Modules

A Python module is a file usually consisting of Python code that can be loaded with the statements `import ...` or `from ... import ...`. A Python package is a collection of Python modules; note that the terms *package* and *module* are often used interchangeably.<sup>[2]</sup>

Some Python modules such as Numpy can be imported if you first load the Lmod module `scipy-stack` at the shell level (see [Software Stack SpiCy](Software Stack SpiCy)).

We offer a large collection of Python wheels, which are pre-compiled modules compatible with our standard software environments.

Before importing modules from a wheel, you must create a virtual environment.

Python modules that are neither in the Lmod module `scipy-stack` nor in our collection of wheels can be installed from the internet as described in [Installing Packages](Installing Packages).


## Additional Information

* Wiki page [Available Software](Available Software)
* Standard software environments; by default, the module collection is `StdEnv/2020` (since April 1, 2021)
* Specific Lmod modules on Niagara
* Lmod modules optimized with CPU instructions for AVX, AVX2 and AVX512
* Wiki page [Category Software](Category Software): list of pages on our wiki site related to commercial or licensed software


## References

1. See [Modular Programming](https://en.wikipedia.org/wiki/Modular_programming) on Wikipedia.
2. See [What is the difference between a python module and a python package?](What is the difference between a python module and a python package?)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Modules/fr&oldid=116636](https://docs.alliancecan.ca/mediawiki/index.php?title=Modules/fr&oldid=116636)"
