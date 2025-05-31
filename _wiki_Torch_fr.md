# Torch

**Outdated**

This page or section contains obsolete information and some statements may not be valid. The technical documentation is currently being updated by our support team.

Torch is a software platform for scientific computing that primarily uses GPUs and allows working with several machine learning algorithms. Its ease of use and efficiency are due to the LuaJIT scripting language and the underlying C/CUDA implementation.

There is some resemblance between Torch and PyTorch. The referenced documents discuss their differences.<sup>[1]</sup> PyTorch offers a Python interface with software that has similar functionalities, but PyTorch does not depend on Torch. See the PyTorch page.


To use Torch you must load a CUDA module.

```bash
[name@server ~]$ module load cuda torch
```

## Installing Lua Packages

Torch includes `luarocks` for managing Lua packages. For the list of installed packages, run `luarocks list`.

If you need a package that is not in the list, use the following command to install it in your own directory:

```bash
[name@server ~]$ luarocks install --local --deps-mode=all <package name>
```

If you have difficulty finding packages at runtime, add the following command just before running your Lua program<sup>[2]</sup>:

```bash
eval $(luarocks path --bin)
```

Some packages do not install well with `luarocks`; if you need assistance, contact technical support.


<sup>[1]</sup> https://stackoverflow.com/questions/44371560/what-is-the-relationship-between-pytorch-and-torch, https://www.quora.com/What-are-the-differences-between-Torch-and-Pytorch, and https://discuss.pytorch.org/t/torch-autograd-vs-pytorch-autograd/1671/4.

<sup>[2]</sup> https://github.com/luarocks/luarocks/wiki/Using-LuaRocks#Rocks_trees_and_the_Lua_libraries_path

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Torch/fr&oldid=151102](https://docs.alliancecan.ca/mediawiki/index.php?title=Torch/fr&oldid=151102)"
