# Installer des logiciels dans votre répertoire personnel

La plupart des logiciels utilisés en recherche sont disponibles gratuitement sur Internet. Vous pouvez nous demander d'installer des logiciels que vous pourrez ensuite utiliser avec la commande `module load` (voir [Utiliser les modules](Utiliser_les_modules)); pour ce faire, écrivez à support@tech.alliancecan.ca en joignant l'adresse URL pour l'installation. Si les clauses de la licence et les exigences techniques le permettent, le logiciel sera rendu disponible le plus tôt possible.

Vous avez le droit d'installer des logiciels dans votre propre espace `/project` ou dans votre espace `/home`, par exemple pour :

* apporter vous-même des modifications au code,
* évaluer le logiciel rapidement.

Prenez connaissance des directives pour l'installation du logiciel. Il s'agit souvent des instructions décrites ci-après.


## `configure; make; make install`

```bash
[name@server ~]$ ./configure
[name@server ~]$ make
[name@server ~]$ make install
```

Cette syntaxe de commande est fréquemment utilisée, avec des variantes telles `cmake .` au lieu de `./configure` et `sudo make install` au lieu de `make install`.

Ces instructions fonctionnent comme prévu à l'occasion, mais `make install` crée quelquefois un obstacle, car le logiciel s'attend à pouvoir écrire dans `/usr/local` ou dans un autre espace partagé du système de fichiers. La commande `sudo make install` causera toujours l'arrêt de la procédure parce que `sudo` exige les permissions d'administrateur (`root`). La solution consiste à placer un indicateur `--prefix` à l'étape `configure` pour que l'installation soit dirigée vers le répertoire de votre choix, par exemple :

```bash
[name@server ~]$ ./configure --prefix=/my/project/directory/some-package && make && make install
```

Si d'autres erreurs surviennent, contactez support@calculcanada.ca. Pour les détails, consultez les pages [Make](Make), [Autotools](Autotools) et [CMake](CMake).


## Utiliser les bibliothèques

Le moyen le plus simple pour utiliser une bibliothèque est habituellement de charger d'abord le module correspondant avec :

```bash
[name@server ~]$ module load library_name/x.y.z
```

Une fois le module chargé, vous pouvez modifier les liens établis au cours du processus de `build` pour inclure la bibliothèque, par exemple :

```bash
[name@server ~]$ gcc -o my_prog file1.o file2.o -lnetcdf
```

pour lier avec la bibliothèque NetCDF.

Sur la ligne pour le lien, le nom de la bibliothèque doit être préfixé par `-l`; il s'agit d'un fichier de type `.a` ou `.so`. Vous trouverez dans la documentation relative à la bibliothèque le nom de ce fichier et l'ordre dans lequel les liens doivent être établis dans les cas où vous avez plusieurs de ces fichiers. Le module pour la bibliothèque doit être chargé pour effectuer le `build`, mais aussi pour exécuter l'application compilée à l'aide de la bibliothèque.

Le chargement du module d'une bibliothèque configure les variables d'environnement `CPATH` et `LIBRARY_PATH` pour qu'elles pointent sur la bibliothèque elle-même et ses fichiers d’en-tête (voir [Utiliser des modules](Utiliser_les_modules)). La plupart des compilateurs, dont Intel et GCC peuvent traiter ces variables; aux étapes de compilation et de construction des liens, les compilateurs iront automatiquement aux bibliothèques indiquées par les variables d'environnement. Ceci permet de facilement établir un lien avec une bibliothèque sans devoir en indiquer le chemin avec les options `-I` et `-L`. Si votre fichier `make` - ou `config-` demande l'endroit spécifique où se trouve la bibliothèque avec `-I` et `-L`, vous pouvez habituellement omettre d’indiquer le chemin en laissant les lignes vides.

Dans certains cas cependant, particulièrement avec `cmake`, il peut être nécessaire d'indiquer de manière explicite la localisation de la bibliothèque fournie par le module. La solution préférée et la plus robuste est d'utiliser la variable d'environnement EasyBuild `EBROOT...` plutôt que d'entrer manuellement le chemin. Ceci permet de facilement utiliser différentes chaînes de compilation (`toolchains`) sans modifier les instructions de compilation, en plus de minimiser le risque de lier une bibliothèque non apparentée. Par exemple, pour indiquer la localisation de la bibliothèque GSL, l'option pour `cmake` pourrait ressembler à `-DGSL_DIR=$EBROOTGSL`. Les variables d'environnement `EBROOT...` utilisent la même syntaxe, soit `EBROOT` suivi par le nom du paquet, par exemple `EBROOTGCC`.


## BLAS/LAPACK et MKL

Voyez notre page wiki [BLAS et LAPACK](BLAS_et_LAPACK).


## `apt-get` et `yum`

Si le logiciel fait appel à `apt-get` ou `yum`, il est peu probable que vous puissiez l'installer avec ces instructions. Repérez les instructions `to build from source` ou contactez le [soutien technique](soutien_technique).


## Paquets Python, R et Perl

Les langages Python, R, et Perl offrent d'importantes bibliothèques d'extensions; presque toutes peuvent être facilement installées dans votre répertoire `/home`. Consultez les pages [Python](Python), [R](R) et [Perl](Perl) pour savoir si le paquet dont vous avez besoin est disponible; si ce n'est pas le cas, vous trouverez aussi dans cette documentation l'information nécessaire pour l’installer par vous-même.


## Installer des paquets binaires

L'installation de binaires précompilés dans votre espace `/home` pourrait générer une erreur comme `/lib64/libc.so.6: version 'GLIBC_2.18' not found`. Le script `setrpaths.sh` peut souvent éliminer ce problème avec la syntaxe :

`setrpaths.sh --path path [--add_origin]`

où `path` représente le répertoire dans lequel vous avez installé le logiciel. Le script fait en sorte que les binaires utilisent le bon interpréteur et trouvent les bibliothèques auxquelles ils sont dynamiquement liés, dans le bon répertoire. L'option `--add_origin` ajoute aussi `$ORIGIN` au RUNPATH, ce qui peut s'avérer utile si la bibliothèque est incapable de trouver d'autres bibliothèques dans le répertoire où elle est située.

**Note:** Certains fichiers d'archive comme `.jar` (Java) ou `.whl` (`Python wheels`) peuvent contenir des objets qui devront être corrigés. Le script `setrpaths.sh` extrait ces objets, les corrige et met à jour le fichier d'archive.


## Environnement logiciel

Le système de fichiers CVMFS (`shared software distribution system`) rend presque tous les logiciels disponibles sur les nouvelles grappes. Sous Linux, les logiciels seraient typiquement installés dans `/usr/bin`, `/usr/include` et ainsi de suite, alors que dans notre cas, ils sont installés de manière identique sur toutes les nouvelles grappes sous `/cvmfs/soft.computecanada.ca`.

Le module `gentoo/2020` est chargé par défaut et agit comme cœur pour la pile logicielle gérée par le gestionnaire de paquets Gentoo situé sous `/cvmfs/soft.computecanada.ca/nix/var/nix/profiles/16.09`. Indiquez ce chemin avec la variable d'environnement `$EBROOTGENTOO`. À cet endroit se trouvent tous les paquets usuels fournis dans un environnement Linux dont `make`, `ls`, `cat`, `grep`. À la compilation d'un logiciel, le compilateur et l'éditeur de liens cherchent typiquement les fichiers d'en-tête et les bibliothèques à l'endroit approprié (avec les variables d'environnement `$CPATH` et `$LIBRARY_PATH` respectivement).

Cependant, dans le cas de certains logiciels, `/usr` est explicitement indiqué; si c'est le cas, la compilation s'arrête et vous devrez spécifier `$EBROOTGENTOO`. Il faudra quelquefois ajuster un Makefile, passer un indicateur `--with-` via le script de compilation ou éditer un fichier de configuration. Si vous ne savez pas comment procéder, contactez le [soutien technique](soutien_technique).

De la même manière, si un paquet dépend d'une bibliothèque provenant d'un module autre que `gentoo`, vous devrez peut-être spécifier où se trouvent les fichiers d'en-tête et les bibliothèques du module. Ces autres modules ont aussi une variable d'environnement commençant par `EBROOT` et se terminant par le nom du module en majuscules. Par exemple, après avoir exécuté la commande `module load hdf5`, son installation se trouvera dans `$EBROOTHDF5`, ses fichiers d'en-tête dans `$EBROOTHDF5/include`, ses fichiers de bibliothèque dans `$EBROOTHDF5/lib` et ainsi de suite.

Si un fichier d'en-tête ou une bibliothèque habituellement offerts dans une distribution de type Linux par un RPM ou autre gestionnaire de paquets ne se trouvent ni par `gentoo`, ni par un autre module, veuillez nous en informer; nous pourrons très probablement l'ajouter.


### Notes

Tous les binaires sous `/cvmfs/soft.computecanada.ca` utilisent un RUNPATH; les répertoires des bibliothèques d'exécution desquels dépendent ces binaires sont placés dans le binaire. Il *n'est donc pas nécessaire* d'utiliser `$LD_LIBRARY_PATH`. En fait, `$LD_LIBRARY_PATH` a préséance sur le RUNPATH et cette variable d'environnement ne devrait pas se trouver dans des endroits comme `/usr/lib64` ou `$EBROOTGENTOO/lib`. Si vous procédez ainsi, plusieurs binaires ne fonctionneront pas.

En dernier recours, utilisez `module --force purge` pour éliminer l'environnement CVMFS. Vous obtiendrez ainsi une installation CentOS-7 brute, sans modules. Ceci peut servir dans des cas spéciaux où vous compilez GCC par vous-même ou quand vous utilisez des chaînes d'outils de compilation comme MESA SDK. Il ne serait nécessaire de purger des modules qu'à la compilation et ils peuvent être chargés à nouveau au lancement du logiciel.


## Compiler avec un nœud de calcul

Dans la plupart des cas, vous pouvez compiler avec un nœud de connexion. Toutefois, si le code doit être développé à l'aide d'un nœud :

* avec un GPU, ou
* avec un CPU Skylake,

vous devriez démarrer une [tâche interactive](tâche_interactive) dans un serveur qui possède le matériel approprié et compiler de cet endroit.

