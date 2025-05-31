# Politique de purge de l'espace /scratch

Cette page est une traduction complète de la page [Scratch purging policy](https://docs.alliancecan.ca/mediawiki/index.php?title=Scratch_purging_policy&oldid=175861).

Autres langues : [English](https://docs.alliancecan.ca/mediawiki/index.php?title=Scratch_purging_policy&oldid=175861), français

## Sommaire

1. Procédure
2. Fichiers à être purgés
3. Connaître l'âge d'un fichier
4. Mauvaises pratiques
5. Copier un dossier avec des symlinks de manière sécuritaire


Sur nos grappes, le système de fichiers `/scratch` sert au stockage rapide et temporaire des données utilisées en cours d'exécution. Pour leur part, les données qui doivent être stockées à long terme et les données de référence sont enregistrées dans l'espace `/project` ou dans une des zones d'archivage. Pour toujours garder suffisamment d'espace `/scratch`, les fichiers de plus de 60 jours sont supprimés périodiquement en fonction de leur âge. Notez que c'est l'âge d'un fichier qui détermine s'il sera purgé et non l'endroit où il se trouve dans `/scratch`; règle générale, le fait de déplacer un fichier dans un autre répertoire de `/scratch` ne l'empêchera pas d'être supprimé.

Remarque : le système de fichiers `/scratch` de Graham n'expire pas comme tel, mais utilise plutôt un quota pour forcer l'utilisation temporaire.


## Procédure

À la fin de chaque mois, les fichiers susceptibles d'être supprimés le 15 du mois suivant sont repérés. Si vous possédez au moins un de ces fichiers, un message d'avertissement s'affiche au début du mois et vous recevez un avis par courriel; cet avis contient aussi une liste de tous les fichiers susceptibles d'être supprimés. Vous avez donc deux semaines pour copier les fichiers que vous voulez conserver.

Le 12 du mois, un dernier avis est envoyé avec une liste à jour des fichiers susceptibles d'être supprimés le 15, ce qui vous laisse 72 heures pour déplacer vos fichiers. Le 15 du mois en fin de journée, tous les fichiers dans l'espace `/scratch` pour lesquels `ctime` et `atime` sont de plus de 60 jours sont supprimés.

Ces fichiers ne doivent pas se trouver dans l'espace `/scratch` et cet avertissement est émis uniquement à titre de courtoisie.

Prenez note que le fait de copier un fichier ou d'utiliser la commande `rsync` pour le déplacer modifie `atime` et fait que le fichier ne sera pas considéré lors de la purge. Une fois les données déplacées, veuillez supprimer les fichiers et répertoires d'origine plutôt que d'attendre qu'ils soient supprimés par la procédure de purge.


## Fichiers à être purgés

Sur Cedar, Béluga et Narval, allez à `/scratch/to_delete/` et localisez le fichier à votre nom.

Sur Niagara, allez à `/scratch/t/to_delete/` ou établissez un lien symbolique (`symlink`) vers `/scratch/t/todelete/current`.

S'il y a un fichier à votre nom, certains de vos fichiers sont susceptibles d'être purgés. Ce fichier contient la liste des noms de fichiers avec le chemin complet et possiblement d'autres renseignements comme la taille, `atime`, `ctime`, `size`, etc. Ce fichier est seulement mis à jour le 1er et le 12e jour de chaque mois.

Si vous accédez à un ou plusieurs fichiers ou les lisez, les déplacez ou les supprimez entre le 1er et le 11 du mois, aucune modification ne sera faite à la liste avant le 12.

Si un fichier avec votre nom existe avant le 11 mais pas le 12, aucun de vos fichiers n'est susceptible d'être purgé.

Si vous accédez à un ou plusieurs fichiers ou les lisez, les déplacez ou les supprimez après le 12 du mois, vous devrez confirmer que les fichiers peuvent ou non être purgés le 15 (voir ci-dessous).


## Connaître l'âge d'un fichier

L'âge d'un fichier est déterminé par : `atime`, le moment du dernier accès et `ctime`, le moment de la dernière modification.

Pour trouver `ctime` utilisez :

```bash
[name@server ~]$ ls -lc <filename>
```

Pour trouver `atime` utilisez :

```bash
[name@server ~]$ ls -lu <filename>
```

Le paramètre (`mtime`) n'est pas utilisé parce que sa valeur peut être modifiée par l'utilisateur ou par un autre programme pour afficher une fausse information.

Il serait suffisant de n'utiliser que `atime` étant donné que sa valeur est mise à jour par le système en synchronisation avec `ctime`. Par contre, les programmes à l'intérieur de l'espace d'un utilisateur peuvent potentiellement modifier `atime` pour situer sa valeur dans le passé. Le fait d'utiliser aussi `ctime` ajoute un deuxième niveau de contrôle.


## Mauvaises pratiques

Il demeure cependant possible de fausser l'âge des fichiers avec l'exécution périodique de la commande récursive `touch`. Notre équipe technique dispose toutefois de moyens pour détecter ce genre de pratique et les utilisateurs qui s'y prêtent seront priés de retirer les fichiers trafiqués de l'espace `/scratch`.


## Copier un dossier avec des symlinks de manière sécuritaire

Dans la plupart des cas, `cp` ou `rsync` seront suffisants pour copier des données de `/scratch` vers votre projet. Mais si vous avez des liens symboliques (`symlink`) dans `/scratch`, les copier posera problème car ils continueront de pointer vers `/scratch`. Pour éviter cela, vous pouvez utiliser `tar` pour faire une archive de vos fichiers sur `/scratch`, et ensuite l'extraire dans votre projet. Vous pouvez le faire d'un seul coup avec :

```bash
cd /scratch/.../vos_donnees
mkdir project/.../vos_donnees
tar cf - ./* | (cd /project/.../vos_donnees && tar xf -)
```
