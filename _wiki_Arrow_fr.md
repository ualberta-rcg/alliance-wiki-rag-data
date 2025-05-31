# Apache Arrow

Apache Arrow est une plateforme de développement multilangage pour la gestion des données en mémoire. Elle utilise un format standardisé en colonnes qui organise les données hiérarchiques ou autres afin de permettre des opérations analytiques efficaces. La plateforme offre des bibliothèques de calcul, la transmission sans copie et en continu des données et la communication interprocessus. Parmi les langages pris en charge, on compte C, C++, C#, Go, Java, JavaScript, MATLAB, Python, R, Ruby et Rust.


## CUDA

Arrow est aussi disponible avec CUDA.

```bash
[name@server ~]$ module load gcc arrow/X.Y.Z cuda
```

où `X.Y.Z` désigne la version.


## Bindings Python

Le module contient des bindings pour plusieurs versions de Python. Pour connaître les versions compatibles, lancez

```bash
[name@server ~]$ module spider arrow/X.Y.Z
```

où `X.Y.Z` désigne la version.

Ou cherchez `pyarrow` directement avec

```bash
[name@server ~]$ module spider pyarrow
```

### PyArrow

Les bindings Python (appelés PyArrow) s’intègrent avec les objets de première classe NumPy, Pandas, et les objets natifs Python. Ils sont basés sur l'implémentation C++ de Arrow.

1. Chargez les modules requis.

```bash
[name@server ~]$ module load gcc arrow/X.Y.Z python/3.11
```

où `X.Y.Z` désigne la version.

2. Importez PyArrow.

```bash
[name@server ~]$ python -c "import pyarrow"
```

L’importation est réussie si rien n’est affiché.

Pour plus d'information, consultez la [documentation Python](link_to_python_docs).


### Autres paquets Python dépendants

L'installation de certains paquets Python est dépendante de PyArrow. Une fois le module `arrow` chargé, la dépendance à `pyarrow` sera satisfaite.

```bash
[name@server ~]$ pip list | grep pyarrow
pyarrow 17.0.0
```


### Format Apache Parquet

Le format de fichier Parquet est disponible. Pour importer le module Parquet, effectuez les étapes pour `pyarrow` ci-dessus et lancez ensuite

```bash
[name@server ~]$ python -c "import pyarrow.parquet"
```

L’importation est réussie si rien n’est affiché.


## Bindings R

Arrow possède une interface avec la bibliothèque Arrow C++ pour permettre l'accès en R de plusieurs de ses fonctionnalités. Ceci inclut l’analyse de grands ensembles de données multifichiers (`open_dataset()`); la capacité de travailler avec des fichiers individuels de format Parquet (`read_parquet()`, `write_parquet()`) et Feather (`read_feather()`, `write_feather()`); l'accès à la mémoire et aux messages Arrow.


### Installation

1. Chargez les modules requis.

```bash
[name@server ~]$ module load StdEnv/2020 gcc/9.3.0 arrow/8 r/4.1 boost/1.72.0
```

2. Spécifiez le répertoire d’installation local.

```bash
[name@server ~]$ mkdir -p ~/.local/R/$EBVERSIONR/
[name@server ~]$ export R_LIBS=~/.local/R/$EBVERSIONR/
```

3. Exportez les variables requises pour vous assurer d’utiliser l'installation du système.

```bash
[name@server ~]$ export PKG_CONFIG_PATH=$EBROOTARROW/lib/pkgconfig
[name@server ~]$ export INCLUDE_DIR=$EBROOTARROW/include
[name@server ~]$ export LIB_DIR=$EBROOTARROW/lib
```

4. Installez les bindings.

```bash
[name@server ~]$ R -e 'install.packages("arrow", repos="https://cloud.r-project.org/")'
```


### Utilisation

Une fois les bindings installés, il faut les charger.

1. Chargez les modules requis.

```bash
[name@server ~]$ module load StdEnv/2020 gcc/9.3.0 arrow/8 r/4.1
```

2. Chargez la bibliothèque.

```bash
[name@server ~]$ R -e "library(arrow)"
> library("arrow")
Attaching package: ‘arrow’
```

Pour plus d'information, consultez la [documentation Arrow sur R](link_to_r_docs).


**(Remember to replace `link_to_python_docs` and `link_to_r_docs` with actual links to the respective documentation.)**
