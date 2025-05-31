# Apache Arrow

Apache Arrow is a cross-language development platform for in-memory data. It uses a standardized language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations. It also provides computational libraries and zero-copy streaming messaging and interprocess communication. Languages currently supported include C, C++, C#, Go, Java, JavaScript, MATLAB, Python, R, Ruby, and Rust.

## CUDA

Arrow is also available with CUDA.

```bash
module load gcc arrow/X.Y.Z cuda
```

where `X.Y.Z` represent the desired version.


## Python Bindings

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run:

```bash
module spider arrow/X.Y.Z
```

where `X.Y.Z` represent the desired version. Or search directly `pyarrow`, by running:

```bash
module spider pyarrow
```

### PyArrow

The Arrow Python bindings (also named `PyArrow`) have first-class integration with NumPy, Pandas, and built-in Python objects. They are based on the C++ implementation of Arrow.

1. Load the required modules:

```bash
module load gcc arrow/X.Y.Z python/3.11
```

where `X.Y.Z` represent the desired version.

2. Import PyArrow:

```bash
python -c "import pyarrow"
```

If the command displays nothing, the import was successful. For more information, see the [Arrow Python documentation](link-to-python-docs).


### Fulfilling Other Python Package Dependencies

Other Python packages depend on PyArrow in order to be installed. With the `arrow` module loaded, your package dependency for `pyarrow` will be satisfied.

```bash
pip list | grep pyarrow
```

### Apache Parquet Format

The Parquet file format is available. To import the Parquet module, execute the previous steps for `pyarrow`, then run:

```bash
python -c "import pyarrow.parquet"
```

If the command displays nothing, the import was successful.


## R Bindings

The Arrow package exposes an interface to the Arrow C++ library to access many of its features in R. This includes support for analyzing large, multi-file datasets (`open_dataset()`), working with individual Parquet files (`read_parquet()`, `write_parquet()`) and Feather files (`read_feather()`, `write_feather()`), as well as lower-level access to the Arrow memory and messages.

### Installation

1. Load the required modules:

```bash
module load StdEnv/2020 gcc/9.3.0 arrow/8 r/4.1 boost/1.72.0
```

2. Specify the local installation directory:

```bash
mkdir -p ~/.local/R/$EBVERSIONR/
export R_LIBS=~/.local/R/$EBVERSIONR/
```

3. Export the required variables to ensure you are using the system installation:

```bash
export PKG_CONFIG_PATH=$EBROOTARROW/lib/pkgconfig
export INCLUDE_DIR=$EBROOTARROW/include
export LIB_DIR=$EBROOTARROW/lib
```

4. Install the bindings:

```bash
R -e 'install.packages("arrow", repos="https://cloud.r-project.org/")'
```

### Usage

After the bindings are installed, they have to be loaded.

1. Load the required modules:

```bash
module load StdEnv/2020 gcc/9.3.0 arrow/8 r/4.1
```

2. Load the library:

```bash
R -e "library(arrow)"
```

```r
library("arrow")
```

For more information, see the [Arrow R documentation](link-to-r-docs).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Arrow&oldid=157736")**
