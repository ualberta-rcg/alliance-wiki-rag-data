# Torch

**Other languages:** English, français

**Outdated**

This page or section contains obsolete information and some statements may not be valid. The technical documentation is currently being updated by our support team.

Torch is a scientific computing framework with wide support for machine learning algorithms that puts GPUs first. It is easy to use and efficient, thanks to an easy and fast scripting language, LuaJIT, and an underlying C/CUDA implementation.

Torch has a distant relationship to PyTorch.<sup>[1]</sup> PyTorch provides a Python interface to software with similar functionality, but PyTorch is not dependent on Torch. See [PyTorch](PyTorch_link_here) for instructions on using it.


## Installing Lua Packages

Torch comes with the Lua package manager, named `luarocks`. Run `luarocks list` to see a list of installed packages.

If you need a package not listed, install it in your own folder using:

```bash
luarocks install --local --deps-mode=all <package name>
```

If you have trouble finding the packages at runtime, add this command<sup>[2]</sup> before running "lua your_program.lua":

```bash
eval $(luarocks path --bin)
```

We often find packages that don't install well with `luarocks`.  If you need help installing a package not in the default module, contact our [Technical support](Technical_Support_Link_Here).


## References

1. See [https://stackoverflow.com/questions/44371560/what-is-the-relationship-between-pytorch-and-torch](https://stackoverflow.com/questions/44371560/what-is-the-relationship-between-pytorch-and-torch), [https://www.quora.com/What-are-the-differences-between-Torch-and-Pytorch](https://www.quora.com/What-are-the-differences-between-Torch-and-Pytorch), and [https://discuss.pytorch.org/t/torch-autograd-vs-pytorch-autograd/1671/4](https://discuss.pytorch.org/t/torch-autograd-vs-pytorch-autograd/1671/4) for some attempts to explain the connection.
2. [https://github.com/luarocks/luarocks/wiki/Using-LuaRocks#Rocks_trees_and_the_Lua_libraries_path](https://github.com/luarocks/luarocks/wiki/Using-LuaRocks#Rocks_trees_and_the_Lua_libraries_path)


**(Note:  Please replace `PyTorch_link_here` and `Technical_Support_Link_Here` with the actual links.)**
