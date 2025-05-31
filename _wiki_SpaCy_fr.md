# spaCy

spaCy is a Python package for advanced Natural Language Processing.

## Installation

### Available Wheels

The following command shows the most recent build of spaCy.

```bash
[name@server ~]$ avail_wheels spacy thinc thinc_gpu_ops
```

See Listing available wheels.


### Pre-compiled Wheels

The preferred option is to install it with a pre-compiled Python wheel.

1. Load the python/3.6 module.
2. Create and activate a virtual environment.
3. Install spaCy in the virtual environment with pip install.

For CPUs and GPUs:

```bash
(venv)[name@server ~] pip install spacy[cuda] --no-index
```

For CPUs only:

```bash
(venv)[name@server ~] pip install spacy --no-index
```

#### GPU Version

To use the GPU version, you currently need to add the CUDA libraries to the `LD_LIBRARY_PATH` variable:

```bash
(venv)[name@server ~] module load gcc/5.4.0 cuda/9
(venv)[name@server ~] export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

To use the Pytorch wrapper with thinc, you also need to install the `torch_cpu` or `torch_gpu` wheel.
