# OpenFOAM

(pour *Open Field Operation and Manipulation*) est un paquet logiciel open source gratuit pour la modélisation numérique de la dynamique des fluides. Ses nombreuses fonctions touchent autant l'électromagnétisme et la dynamique des solides que les flux liquides complexes avec réaction chimique, turbulence et transfert thermique.

## Modules

Pour une version récente, utilisez

```bash
[name@server ~]$ module load openfoam
```

La communauté OpenFOAM comprend :

*   La OpenFOAM Foundation avec ses sites web [openfoam.org](openfoam.org) et [cfd.direct](cfd.direct)
*   OpenCFD avec son site web [openfoam.com](openfoam.com)

Les versions semblent identiques jusqu'à 2.3.1 (décembre 2014). Pour les versions après 2.3.1, les modules avec des noms commençant par la lettre `v` sont dérivés de la branche `.com` (par exemple `openfoam/v1706`); les modules avec des noms commençant par un chiffre sont dérivés de la branche `.org` (par exemple, `openfoam/4.1`).

Pour plus d'information sur les commandes, consultez [Utiliser des modules](link-to-modules-page).


## Documentation

*   [documentation OpenFOAM.com](link-to-openfoam-com-docs)
*   [CFD Direct, Guide de l'utilisateur](link-to-cfd-direct-guide)


## Utilisation

Votre environnement nécessite beaucoup de préparation. Pour pouvoir exécuter les commandes OpenFOAM (`paraFoam`, `blockMesh`, etc.), vous devez charger un module.

Le script suivant est pour une tâche séquentielle avec OpenFOAM 5.0 :

**File:** `submit.sh`

```bash
#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --account=def-someuser
module purge
module load openfoam/5.0

blockMesh
icoFoam
```

Le script suivant est pour une tâche parallèle :

**File:** `submit.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --ntasks=4               # number of MPI processes
#SBATCH --mem-per-cpu=1024M      # memory; default unit is megabytes
#SBATCH --time=0-00:10           # time (DD-HH:MM)
module purge
module load openfoam/5.0

blockMesh
setFields
decomposePar
srun interFoam -parallel
```

La préparation du maillage (`blockMesh`) peut être assez rapide pour se faire en ligne de commande (voir [Exécuter des tâches](link-to-tasks-page)). L'étape la plus exigeante est habituellement celle du solveur (entre autres `icoFoam`); ces tâches devraient toujours être soumises à l'ordonnanceur, sauf pour de très petits cas ou des tutoriels.


## Erreurs « segfault » avec OpenMPI 3.1.2

Des utilisateurs ont rapporté des plantages aléatoires (« segfault ») sur Cedar lors de l’utilisation de versions d’OpenFOAM compilées avec OpenMPI 3.1.2 pour des tâches sur un seul nœud (communication par mémoire partagée). Ces problèmes semblent spécifiques à cette version. Si vous obtenez une erreur semblable, essayez d’abord d’utiliser une chaîne de compilation basée sur OpenMPI 2.1.1. Par exemple :

```bash
[name@server ~]$ module load gcc/5.4.0
[name@server ~]$ module load openmpi/2.1.1
[name@server ~]$ module load openfoam/7
```


## Performance

La fonction de débogage produit fréquemment des centaines d'opérations d'écriture par seconde, ce qui peut causer une baisse de performance des systèmes de fichiers partagés. Si vous êtes en production et que vous n'avez pas besoin de cette information, diminuez ou désactivez la fonction de débogage avec :

```bash
[name@server ~]$ mkdir -p $HOME/.OpenFOAM/$WM_PROJECT_VERSION
[name@server ~]$ cp $WM_PROJECT_DIR/etc/controlDict $HOME/.OpenFOAM/$WM_PROJECT_VERSION/
```

Plusieurs autres paramètres peuvent diminuer la quantité et la fréquence des écritures sur disque; voir la documentation pour la [version 6](link-to-version-6-docs) et la [version 7](link-to-version-7-docs).

Par exemple, le dictionnaire `debugSwitches` dans `$HOME/.OpenFOAM/$WM_PROJECT_VERSION/controlDict` peut être modifié pour que les valeurs des indicateurs qui sont plus grandes que zéro soient égales à zéro. Une autre solution serait d'utiliser l'espace scratch local (`$SLURM_TMPDIR`) qui est un disque attaché directement au nœud de calcul; voir la [section Disque local dans la page Travailler avec un grand nombre de fichiers](link-to-large-files-page).


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=OpenFOAM/fr&oldid=175232")**
