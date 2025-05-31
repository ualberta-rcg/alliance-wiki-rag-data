# spaCy

spaCy is a Python package that provides industrial-strength natural language processing.

## Installation

### Latest available wheels

To see the latest version of spaCy that we have built:

```bash
name@server ~] $ avail_wheels spacy thinc thinc_gpu_ops
```

For more information on listing wheels, see [listing available wheels](link_to_listing_wheels_documentation).


### Pre-build

The preferred option is to install it using the python wheel that we compile, as follows:

1. Load python 3.6 module: `python/3.6`
2. Create and activate a virtual environment.
3. Install spaCy in the virtual environment with `pip install`. For both GPU and CPU support:

```bash
(venv) [name@server ~] pip install spacy[cuda] --no-index
```

If you only need CPU support:

```bash
(venv) [name@server ~] pip install spacy --no-index
```

### GPU version

At the present time, in order to use the GPU version you need to add the CUDA libraries to `LD_LIBRARY_PATH`:

```bash
(venv) [name@server ~] module load gcc/5.4.0 cuda/9
(venv) [name@server ~] export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

If you want to use the Pytorch wrapper with thinc, you'll also need to install the `torch_cpu` or `torch_gpu` wheel.


**(Note:  Replace bracketed placeholders like `[link_to_listing_wheels_documentation]` with actual links or file paths as needed.)**
