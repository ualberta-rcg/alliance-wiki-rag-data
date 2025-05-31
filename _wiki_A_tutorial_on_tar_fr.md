# Tutoriel sur 'tar'

Cette page est une version traduite de la page [A tutorial on 'tar'](https://docs.alliancecan.ca/mediawiki/index.php?title=A_tutorial_on_%27tar%27&oldid=93417) et la traduction est complète à 100 %.

Autres langues :

*   [Anglais](https://docs.alliancecan.ca/mediawiki/index.php?title=A_tutorial_on_%27tar%27&oldid=93417)
*   Français

## Archiver des fichiers et des répertoires

La commande `tar` est l’utilitaire d’archivage principal sous Linux et autres systèmes de type Unix. La commande rassemble plusieurs fichiers ou répertoires et génère un *fichier archive* (nommé aussi *fichier tar* ou *tarball*). Par convention, un fichier archive possède le suffixe `.tar`. Le fichier archive d’un répertoire contient par défaut tous les fichiers et sous-répertoires avec leurs sous-répertoires, sous-sous-répertoires et ainsi de suite. Par exemple, la commande

```bash
tar --create --file project1.tar project1
```

rassemble le contenu du répertoire `project1` dans le fichier `project1.tar` ; le fichier d’origine est conservé, ce qui peut doubler l’espace disque utilisé.

Pour extraire des fichiers du fichier archive, utilisez la commande avec une option différente, soit

```bash
tar --extract --file project1.tar
```

S’il n’existe pas de répertoire avec le nom d’origine, il sera créé. S’il existe un répertoire avec le nom d’origine et qu’il contient des fichiers portant le même nom que ceux du fichier archive, ils seront remplacés. Il y a aussi une option pour spécifier le répertoire de destination pour le contenu extrait du fichier archive.

## Compression et décompression

L’utilitaire `tar` peut compresser un fichier archive en même temps que ce fichier est créé. Parmi les méthodes de compression, nous recommandons `xz` ou `gzip` qui s’utilisent comme suit :

```bash
[user_name@localhost]$ tar --create --xz --file project1.tar.xz project1
[user_name@localhost]$ tar --extract --xz --file project1.tar.xz
[user_name@localhost]$ tar --create --gzip --file project1.tar.gz project1
[user_name@localhost]$ tar --extract --gzip --file project1.tar.gz
```

De façon générale, `--xz` produit un fichier compressé plus petit (c’est-à-dire avec un meilleur taux de compression), mais utilise plus de mémoire RAM<sup>[1]</sup>. `--gzip` ne compresse pas autant, mais vous pouvez l’utiliser si vous avez des problèmes de manque de mémoire ou de durée d’exécution avec `tar --create`.

Vous pouvez aussi lancer `tar --create` d’abord sans compression et utiliser ensuite la commande `xz` ou `gzip` dans une étape distincte, mais il est rarement utile de procéder ainsi. De même, vous pouvez lancer `xz -d` ou `gzip -d` pour décompresser un fichier archive avant de lancer `tar --extract`, mais ceci est aussi rarement utile.

Une fois que le fichier tar est créé, il est aussi possible d’utiliser `gzip` ou `bzip2` pour compresser l’archive et diminuer sa taille :

```bash
[user_name@localhost]$ gzip project1.tar
[user_name@localhost]$ bzip2 project1.tar
```

Ces commandes produisent les fichiers `project1.tar.gz` et `project1.tar.bz2`.

## Options fréquemment employées

Il y a deux formes pour chaque option.

*   `-c` ou `--create` pour créer une nouvelle archive
*   `-f` ou `--file=` précède le nom du fichier archive
*   `-x` ou `--extract` pour extraire des fichiers d’une archive
*   `-t` ou `--list` pour lister le contenu d’un fichier archive
*   `-J` ou `--xz` pour compresser ou décompresser avec `xz`
*   `-z` ou `--gzip` pour compresser ou décompresser avec `gzip`

Les options de la forme simple peuvent être combinées en les faisant précéder d’un seul tiret; par exemple `tar -cJf project1.tar.zx project1` équivaut à `tar --create --xz --file=project1.tar.xz project1`.

Plusieurs autres options sont disponibles, dépendant de la version que vous utilisez. Pour obtenir la liste de toutes les options dont vous disposez, lancez `man tar` ou `tar --help`. Notez que certaines versions moins récentes peuvent ne pas supporter la compression avec `--xz`.

## Exemples

Dans les exemples qui suivent, nous supposons un répertoire qui contient les sous-répertoires et fichiers (`bin/ documents/ jobs/ new.log.dat programs/ report/ results/ tests/ work`). Comparez ces exemples avec le contenu de votre propre répertoire.

### Archivage

#### Répertoires particuliers

On utilise `tar` le plus fréquemment pour créer une archive d’un répertoire. Dans cet exemple, nous créons le fichier archive `results.tar` avec le répertoire `results`.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  tests/  work/
[user_name@localhost]$ tar -cvf results.tar results
results
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
```

Avec la commande `ls`, nous voyons le nouveau fichier tar :

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  results.tar  tests/  work/
```

Nous avons utilisé la commande `tar` avec les options `-c` (pour *create*), `-v` (pour *verbosity*) et `-f` (pour *file*). Nous avons nommé l’archive `results.tar` ; le nom pourrait être différent, mais il est préférable qu’il soit semblable à celui du répertoire pour que vous puissiez plus facilement le reconnaître.

Vous pouvez placer plusieurs répertoires ou fichiers dans un fichier tar; par exemple, pour placer les répertoires `results`, `report` et `documents` dans le fichier archive `full_results.tar`, nous utilisons :

```bash
[user_name@localhost]$ tar -cvf full_results.tar results report documents/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
report/
report/report-2016.pdf
report/report-a.pdf
documents/
documents/1504.pdf
documents/ff.doc
```

L’option `v` permet de voir les fichiers qui ont été ajoutés; pour les cacher, omettez cette option.

Pour vérifier l’archive créée, utilisez `ls` :

```bash
[user_name@localhost]$ ls
bin/  documents/  full_results.tar  jobs/  new.log.dat  programs/  report/  results/  results.tar  tests/  work/
```

#### Fichiers et répertoires dont le nom commence par une lettre en particulier

Dans notre répertoire de travail se trouvent deux répertoires qui commencent par la lettre `r` (`reports` et `results`). Dans cet exemple, nous rassemblons le contenu de ces répertoires dans une seule archive (`archive.tar`).

```bash
[user_name@localhost]$ tar -cvf archive.tar r*
report/
report/report-2016.pdf
report/report-a.pdf
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
```

Ici nous avons rassemblé tous les répertoires qui commencent par la lettre `r`. Il est aussi possible de rassembler des fichiers ou des répertoires avec une chaîne de caractères, par exemple *r*, *.dat, etc.

#### Ajouter (*append*) des fichiers à la fin d’une archive

L’option `-r` est utilisée pour ajouter des fichiers à une archive existante sans avoir à en créer une autre ou à décompresser l’archive puis lancer `tar` à nouveau pour créer une nouvelle archive. Dans le prochain exemple, nous ajoutons le fichier `new.log.dat` à l’archive `results.tar`.

```bash
[user_name@localhost]$ tar -rf results.tar new.log.dat
```

La commande `tar` a ajouté le fichier `new.log.dat` à la fin de l’archive `results.tar`.

Pour vérifier, utilisez les options précédentes pour lister les fichiers du fichier tar :

```bash
[user_name@localhost]$ tar -tvf results.tar
drwxrwxr-x name name        0 2016-11-20 11:02 results/
-rw-r--r-- name name    10905 2016-11-16 16:31 results/log1.dat
drwxrwxr-x name name        0 2016-11-16 19:36 results/Res-01/
-rw-r--r-- name name    11672 2016-11-16 15:10 results/Res-01/log.15Feb16.1
-rw-r--r-- name name    11682 2016-11-16 15:10 results/Res-01/log.15Feb16.4
drwxrwxr-x name name        0 2016-11-16 19:37 results/Res-02/
-rw-r--r-- name name    34117 2016-11-16 15:10 results/Res-02/log.15Feb16.balance.b.4
-rw-r--r-- name name    10905 2016-11-20 11:16 new.log.dat
```

**Note :** Il n’est pas possible d’ajouter des fichiers à une archive compressée (*.gz ou *.bz2). Les fichiers peuvent être ajoutés uniquement à une archive tar ordinaire. L’option `-r` est aussi utilisée avec la commande `tar` pour ajouter un ou plusieurs répertoires à un fichier tar existant. Nous allons maintenant ajouter le répertoire `report` à l’archive `results.tar` de l’exemple précédent :

```bash
[user_name@localhost]$ tar -rf results.tar report/
```

Voyons maintenant le fichier tar créé :

```bash
[user_name@localhost]$ tar -tvf results.tar
drwxrwxr-x name name        0 2016-11-20 11:02 results/
-rw-r--r-- name name    10905 2016-11-16 16:31 results/log1.dat
drwxrwxr-x name name        0 2016-11-16 19:36 results/Res-01/
-rw-r--r-- name name    11672 2016-11-16 15:10 results/Res-01/log.15Feb16.1
-rw-r--r-- name name    11682 2016-11-16 15:10 results/Res-01/log.15Feb16.4
drwxrwxr-x name name        0 2016-11-16 19:37 results/Res-02/
-rw-r--r-- name name    34117 2016-11-16 15:10 results/Res-02/log.15Feb16.balance.b.4
-rw-r--r-- name name    10905 2016-11-20 11:16 new.log.dat
drwxrwxr-x name name        0 2016-11-20 11:02 report/
-rw-r--r-- name name   924729 2015-11-20 04:14 report/report-2016.pdf
-rw-r--r-- name name   924729 2015-11-20 04:14 report/report-a.pdf
```

Rappelez-vous que l’option `-v` n’est pas nécessaire si vous n’avez pas besoin de voir les détails pour les fichiers.

#### Combiner deux archives

Comme on peut ajouter un fichier à une archive, on peut aussi ajouter une archive à une autre archive avec l’option `-A`. Ajoutons l’archive `report.tar` (pour le rapport du répertoire) à l’archive `results.tar` existante :

Pour vérifier l’archive existante :

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  report.tar  results/  results.tar  tests/  work/
[user_name@localhost]$ ar -tvf results.tar
drwxr-xr-x name name        0 2016-11-20 16:16 results/
-rw-r--r-- name name    10905 2016-11-20 16:16 results/log1.dat
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-01/
-rw-r--r-- name name    11682 2016-11-20 16:16 results/Res-01/log.15Feb16.4
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-02/
-rw-r--r-- name name    34117 2016-11-20 16:16 results/Res-02/log.15Feb16.balance.b.4
```

Ajoutons maintenant l’archive et vérifions la nouvelle archive :

```bash
[user_name@localhost]$ tar -A -f results.tar report.tar
[user_name@localhost]$ tar -tvf results.tar
drwxr-xr-x name name        0 2016-11-20 16:16 results/
-rw-r--r-- name name    10905 2016-11-20 16:16 results/log1.dat
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-01/
-rw-r--r-- name name    11682 2016-11-20 16:16 results/Res-01/log.15Feb16.4
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-02/
-rw-r--r-- name name    34117 2016-11-20 16:16 results/Res-02/log.15Feb16.balance.b.4
drwxrwxr-x name name        0 2016-11-20 11:02 report/
-rw-r--r-- name name   924729 2015-11-20 04:14 report/report-2016.pdf
-rw-r--r-- name name   924729 2015-11-20 04:14 report/report-a.pdf
```

Dans l’exemple précédent, nous avons utilisé l’option `-A` (pour *Append*) dans `tar -A -f results.tar report.tar` pour ajouter l’archive `report.tar` à l’archive `results.tar` comme vous pouvez voir en comparant le résultat de la commande `tar -tvf results.tar` avant et après l’opération.

**Note :** Les options `-A`, `--catenate` et `--concatenate` sont équivalentes; dépendant du système que vous utilisez, certaines options pourraient ne pas être disponibles. La commande précédente peut aussi être utilisée comme suit :

```bash
[user_name@localhost]$ tar -A -f full-results.tar report.tar
[user_name@localhost]$ tar -A --file=full-results.tar report.tar
[user_name@localhost]$ tar --list --file=full-results.tar
```

**Note :** Il existe deux possibilités pour ajouter l’archive `archive_2.tar` à l’archive `archive_1.tar`. La première est d’utiliser `-r` comme nous avons vu précédemment quand on ajoute un fichier à une archive existante. Dans ce cas, l’archive ajoutée `archive_2.tar` paraîtra comme un fichier ajouté à une archive existante. L’option `-tvf` montrera que l’archive sera ajoutée comme un fichier à la fin de l’archive. La deuxième possibilité est d’utiliser l’option `-A`. Dans ce cas, l’archive ajoutée ne paraîtra pas comme une archive; la commande créera une nouvelle archive.

#### Exclure certains fichiers

À partir de l’exemple précédent, créons l’archive `results.tar` pour y enregistrer les résultats, mais en y ajoutant l’option `--exclude=*.dat` pour exclure les fichiers avec le suffixe `.dat`.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  tests/  work/
[user_name@localhost]$ ls results/
log1.dat  log5.dat  Res-01/  Res-02/
[user_name@localhost]$ tar -cvf results.tar --exclude=*.dat results/
results/
results/Res-01/
results/Res-01/log.15Feb16.4|results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ tar -tvf results.tar
drwxr-xr-x name name        0 2016-11-20 16:16 results/
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-01/
-rw-r--r-- name name    11682 2016-11-20 16:16 results/Res-01/log.15Feb16.4
drwxr-xr-x name name        0 2016-11-20 16:16 results/Res-02/
-rw-r--r-- name name    34117 2016-11-20 16:16 results/Res-02/log.15Feb16.balance.b.4
```

#### Conserver les liens symboliques

Si vous avez des liens symboliques dans votre répertoire et que vous voulez les préserver, ajouter l’option `-h` à la commande `tar`.

```bash
[user_name@localhost]$ tar -cvhf results.tar results/
```

### Compression

#### Compresser un fichier, des fichiers, un fichier archive tar

La compression et l’archivage sont deux processus différents. L’archivage ou la création d’un fichier tar rassemble plusieurs fichiers ou répertoires dans un même fichier. Le processus de compression s’effectue sur un seul fichier ou une seule archive pour en diminuer la taille, avec des utilitaires comme gzip ou bzip2. Dans l’exemple suivant, nous compressons `new.log.dat` et `results.tar`.

Avec gzip :

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar  tests/  work/
[user_name@localhost]$ gzip new.log.dat
[user_name@localhost]$ gzip results.tar
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat.gz  new_results/  programs/  report/  results/  results.tar.gz  tests/  work/
```

Avec bzip2 :

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar  tests/  work/
[user_name@localhost]$ bzip2 new.log.dat
[user_name@localhost]$ bzip2 results.tar
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat.bz2  new_results/  programs/  report/  results/  results.tar.bz2  tests/  work/
```

**Note :** Pour compresser en même temps que l’archive est créée, utilisez les options `z` ou `j` pour gzip ou bzip2 respectivement. L’extension du nom du fichier n’a pas vraiment d’importance. Pour les fichiers compressés avec gzip, `.tar.gz` et `.tgz` sont des extensions communes; pour les fichiers compressés avec bzip2, `.tar.bz2` et `.tbz` sont des extensions communes.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  tests/  work/
[user_name@localhost]$ tar -cvzf results.tar.gz results/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.4|results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  results.tar.gz  tests/  work/
[user_name@localhost]$ tar -cvjf results.tar.bz2 results/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  programs/  report/  results/  results.tar.bz2  results.tar.gz  tests/  work/
```

#### Ajouter des fichiers à une archive compressée (tar.gz/tar.bz2)

Nous avons déjà mentionné qu’il n’est pas possible d’ajouter des fichiers à des archives compressées. Si nous devons le faire, il faut décompresser les fichiers avec `gunzip` ou `bunzip2`. Une fois que nous avons obtenu le fichier tar, nous ajoutons les fichiers à cette archive en invoquant l’option `r`. Nous pouvons ensuite compresser à nouveau avec gzip ou bzip2.

### Décompression

#### Extraire l’archive au complet

Pour décompresser ou extraire une archive, utilisez l’option `-x` (pour *extract*) avec `-f` (pour *file*) ; vous pouvez aussi ajouter `-v` (pour *verbosity*). Nous allons maintenant extraire l’archive `results.tar`. Pour extraire dans le même répertoire, il faut s’assurer qu’aucun autre répertoire ne possède ce même nom. Pour éviter d’avoir à ré-écrire les données s’il existe déjà un répertoire avec ce nom, nous allons rediriger l’extraction vers un autre répertoire avec l’option `-C`, en s’assurant que le répertoire de destination existe ou est créé avant de décompresser l’archive. Par exemple, créons le répertoire `moved_results` pour y extraire les données de l’archive `results.tar`.

```bash
[user_name@localhost]$ tar -xvf results.tar -C new_results/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
new.log.dat
report/
report/report-2016.pdf
report/report-a.pdf
[user_name@localhost]$ ls new_results/
new.log.dat  report/  results/
```

**Note :** L’option `v` fait afficher uniquement les noms des fichiers qui ont été extraits de l’archive. Invoquez cette option deux fois pour obtenir plus de détails (utilisez `-xvvf` au lieu de `-xvf`).

#### Décompresser des fichiers gz et bz2

Pour les fichiers avec l’extension `.gz`, utilisez gunzip.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat.gz  new_results/  programs/  report/  results/  results.tar.gz  tests/  work/
[user_name@localhost]$ gunzip new.log.dat.gz
[user_name@localhost]$ gunzip results.tar.gz
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar  tests/  work/
```

Pour les fichiers avec l’extension `.bz2`, utilisez `bunzip2`.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat.bz2  new_results/  programs/  report/  results/  results.tar.bz2  tests/  work/
[user_name@localhost]$ bunzip2 new.log.dat.bz2
[user_name@localhost]$ bunzip2 results.tar.bz2
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar  tests/  work/
```

#### Extraire un fichier archive compressé vers un autre répertoire

Comme c’est le cas avec un fichier tar, un fichier *compressed tar* peut être extrait dans un autre répertoire avec l’option `-C` pour indiquer le répertoire de destination et l’option `z` pour les fichiers *.gz ou `j` pour les fichiers *.bz2. Avec le même exemple que précédemment, nous allons extraire l’archive `results.tar.gz` (ou `results.tar.bz2`) dans le répertoire `new_results` en une ou deux étapes.

##### Extraire le fichier archive compressé en une étape

Avec gz

```bash
[user_name@localhost]$ tar -xvzf results.tar.gz -C new_results/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/
results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ tar -xzf results.tar.gz -C new_results/
[user_name@localhost]$ ls new_results/
results/
```

Avec l’extension bz2

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar.bz2  tests/  work/
[user_name@localhost]$ tar -xvjf results.tar.bz2 -C new_results/
results/
results/log1.dat
results/Res-01/
results/Res-01/log.15Feb16.1
results/Res-01/log.15Feb16.4
results/Res-02/|results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ ls new_results/
results/
```

**Notes :**

*   Dans l’exemple précédent, il est possible de commencer avec l’option `-C` (le répertoire de destination), cependant, assurez-vous d’abord que le répertoire de destination existe puisque tar ne va pas le créer pour vous et s’il n’existe pas, tar va échouer. La commande est
    ```bash
    [user_name@localhost]$ tar -C new_results/ -xzf results.tar.gz
    ```
    ou
    ```bash
    [user_name@localhost]$ tar -C new_results/ -xvjf results.tar.bz2
    ```
*   Si l’option `-C` (répertoire de destination) n’est pas invoquée, les fichiers seront extraits dans le même répertoire.

L’option `v` (pour *verbosity*) fait afficher les fichiers et répertoires comme ils sont extraits vers le nouveau répertoire.

Pour faire afficher plus de détails (comme la date, la permission, etc.), ajoutez une autre option `v` comme suit :

```bash
tar -C new_results/ -xvvzf results.tar.gz
```

ou

```bash
tar -C new_results/ -xvvjf results.tar.bz2
```

L’extraction du fichier archive compressé se fait en deux étapes.

Nous utilisons ici les mêmes commandes qu’auparavant, mais sans les options `z` ou `j`. D’abord, gunzip ou bunzip2 décompresse le fichier et ensuite `tar -xvf` pour *dé-tarer* l’archive comme suit :

En supposant que nous avons le fichier compressé `results.tar.bz2` :

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar.bz2  tests/  work/
[user_name@localhost]$ bunzip2 results.tar.bz2
[user_name@localhost]$ tar -C ./new_results/ -xvvf results.tar
drwxrwxr-x name name        0 2016-11-20 11:02 results/
-rw-r--r-- name name    10905 2016-11-16 16:31 results/log1.dat
drwxrwxr-x name name        0 2016-11-20 15:16 results/Res-01/
-rw-r--r-- name name    11682 2016-11-16 15:10 results/Res-01/log.15Feb16.4
drwxrwxr-x name name        0 2016-11-16 19:37 results/Res-02/
-rw-r--r-- name name    34117 2016-11-16 15:10 results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ ls new_results/results/
log1.dat  log5.dat  Res-01/   Res-02/
[user_name@localhost]$ ls new_results/results/
log1.dat  Res-01/  Res-02/
```

Pour les fichiers *.gz

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar.gz  tests/  work/
[user_name@localhost]$ gunzip results.tar.gz
[user_name@localhost]$ tar -C ./new_results/ -xvvf results.tar
drwxrwxr-x name name        0 2016-11-20 11:02 results/
-rw-r--r-- name name    10905 2016-11-16 16:31 results/log1.dat
drwxrwxr-x name name        0 2016-11-20 15:16 results/Res-01/
-rw-r--r-- name name    11682 2016-11-16 15:10 results/Res-01/log.15Feb16.4
drwxrwxr-x name name        0 2016-11-16 19:37 results/Res-02/
-rw-r--r-- name name    34117 2016-11-16 15:10 results/Res-02/log.15Feb16.balance.b.4
[user_name@localhost]$ ls new_results/results/
log1.dat  Res-01/  Res-02/
```

#### Extraire un fichier d’une archive ou d’une archive compressée

Avec l’exemple précédent, nous allons d’abord créer l’archive `results.tar` pour archiver le répertoire et lister tous les fichiers qu’il contient et ensuite extraire un fichier vers le répertoire `new_results`.

```bash
[user_name@localhost]$ ls
bin/  documents/  jobs/  new.log.dat  new_results/  programs/  report/  results/  results.tar  tests/  