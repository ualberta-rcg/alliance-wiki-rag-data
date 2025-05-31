# GDAL

GDAL is an open source translator library for raster geospatial data formats. It can be used as a library, presenting a single abstract data model to the calling application for all supported formats. It also comes with a variety of useful command line utilities for data translation and processing. GDAL is used by a [long list of software packages](<link_to_list_of_software_packages>) and its functionality can be used in scripts written in [Python](<link_to_python_docs>) or [R](<link_to_r_docs>).


## Using GDAL from Python

GDAL functionality can be used via the `osgeo` package, which we install as an extension to the GDAL module. In order to use it, you need to load a compatible Python module alongside the GDAL module.

### Using osgeo under StdEnv/2020

Check which Python modules are compatible with e.g. `gdal/3.5.1`:

```bash
[name@server ~]$ module whatis gdal/3.5.1
gdal/3.5.1 : Description: GDAL is a translator library for raster geospatial data formats...
gdal/3.5.1 : Homepage: https://www.gdal.org/
gdal/3.5.1 : URL: https://www.gdal.org/
gdal/3.5.1 : Compatible modules: python/3.8, python/3.9, python/3.10
```

We have the choice between Python 3.8, 3.9 and 3.10. Let's choose `python/3.10` for this.

```bash
[name@server ~]$ module load StdEnv/2020 gcc/9.3.0 python/3.10 gdal/3.5.1
```

```python
# File: osgeo_gdal.py
#!/usr/bin/env python3
from osgeo import gdal
print("osgeo.gdal version:", gdal.__version__)
# osgeo.gdal version: 3.5.1
```

### Using osgeo under StdEnv/2023

Check which Python modules are compatible with e.g. `gdal/3.7.2`:

```bash
[name@server ~]$ module whatis gdal/3.7.2
gdal/3.7.2 : Description: GDAL is a translator library for raster geospatial data formats... data translation and processing.
gdal/3.7.2 : Homepage: https://www.gdal.org/
gdal/3.7.2 : URL: https://www.gdal.org/
gdal/3.7.2 : Compatible modules: python/3.10, python/3.11
gdal/3.7.2 : Extensions: osgeo-3.7.2
```

We have the choice between Python 3.10 and 3.11. Let's choose `python/3.11` for this.

```bash
[name@server ~]$ module load StdEnv/2023 gcc/12.3 python/3.11 gdal/3.7.2
```

```python
# File: osgeo_gdal.py
#!/usr/bin/env python3
from osgeo import gdal
print("osgeo.gdal version:", gdal.__version__)
# osgeo.gdal version: 3.7.2
```

## Using GDAL from R

Several R-packages for Analysis of Spatial Data directly depend on GDAL as a System dependency. For example:

*   `sf`: Simple Features for R
*   `terra`: Spatial Data Analysis

The older package `rgdal` has been discontinued in favor of `sf` and `terra`.

### Installing sf and terra under StdEnv/2020

Installing these packages not only requires loading a `gdal` module, but also `udunits` which is required by `units`.

```bash
# File: install_sf_terra_StdEnv2020.sh
# load required modules:
module load StdEnv/2020 gcc/9.3.0 udunits/2.2.28 gdal/3.5.1 r/4.2.2
# create a local R library in $HOME:
mkdir -p $HOME/R/x86_64-pc-linux-gnu-library/4.2
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/4.2:$R_LIBS"
# install sf and terra from a Canadian CRAN mirror:
R -e "install.packages(c('sf', 'terra'), repos='https://mirror.csclub.uwaterloo.ca/CRAN/', dep=TRUE)"
```

### Installing sf and terra under StdEnv/2023

Note that under StdEnv/2023, in addition to modules `gdal` and `udunits`, also `hdf/4.3.1` is required.

```bash
# File: install_sf_terra_StdEnv2020.sh
# load required modules:
module load StdEnv/2023 gcc/12.3 udunits/2.2.28 hdf/4.2.16 gdal/3.7.2 r/4.4.0
# create a local R library in $HOME:
mkdir -p $HOME/R/x86_64-pc-linux-gnu-library/4.4
export R_LIBS="$HOME/R/x86_64-pc-linux-gnu-library/4.4:$R_LIBS"
# install sf and terra from a Canadian CRAN mirror:
R -e "install.packages(c('sf', 'terra'), repos='https://mirror.csclub.uwaterloo.ca/CRAN/', dep=TRUE)"
```

**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=GDAL&oldid=172434")**
