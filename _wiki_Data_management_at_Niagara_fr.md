# Gestion des données sur Niagara

Cette page est une traduction de la page [Data management at Niagara](https://docs.alliancecan.ca/mediawiki/index.php?title=Data_management_at_Niagara&oldid=177146) et la traduction est complète à 79 %.

Autres langues :

*   Anglais
*   Français

Pour travailler de façon optimale et faire bon usage des ressources, il faut bien connaître les divers systèmes de fichiers. Nous donnons ici des renseignements sur comment les utiliser correctement.

## Contenu

1.  Performance
2.  Utilisation des systèmes de fichiers
    *   `/home` (`$HOME`)
    *   `/scratch` (`$SCRATCH`)
    *   `/project` (`$PROJECT`)
    *   `/bb` (`$BBUFFER`)
    *   `/archive` (`$ARCHIVE`)
    *   `/dev/shm` (RAM)
    *   `$SLURM_TMPDIR` (RAM)
    *   Espace de mémoire tampon d'éclatement temporaire par tâche (`$BB_JOB_DIR`)
3.  Quotas et purge
    *   Combien d'espace disque me reste-t-il ?
    *   Politique de purge du disque de travail
4.  Déplacer des données
    *   Avec `rsync`/`scp`
    *   Utiliser Globus
    *   Déplacer des données vers HPSS/Archive/Nearline
5.  Gestion de la propriété des fichiers et listes de contrôle d'accès
    *   Utilisation de `mmputacl`/`mmgetacl`
    *   Script ACL récursif


## Performance

À l'exception de `/archive`, les systèmes de fichiers hautement performants de SciNet sont de type GPFS ; ils permettent des opérations parallèles de lecture et d'écriture rapides avec de grands ensembles de données, à partir de plusieurs nœuds. Par contre, de par sa conception, sa performance laisse beaucoup à désirer quand il s'agit d'accéder à des ensembles de données composés de plusieurs petits fichiers. En effet, il est beaucoup plus rapide de lire un fichier de 16 Mo que 400 fichiers de 40 Ko. Rappelons que dans ce dernier cas, autant de petits fichiers n'est pas une utilisation efficace de l'espace puisque la capacité des blocs est de 16 Mo pour les systèmes `/scratch` et `/project`. Tenez compte de ceci dans votre stratégie de lecture/écriture.

Par exemple, pour exécuter une tâche multiprocessus, le fait que chaque processus écrive dans son propre fichier n'est pas une solution I/O flexible ; le répertoire est bloqué par le premier processus qui l'accède et les suivants doivent attendre. Non seulement cette solution rend-elle le code considérablement moins parallèle, mais le système de fichiers sera arrêté en attendant le prochain processus et votre programme se terminera mystérieusement.

Utilisez plutôt MPI-IO (partie du standard MPI-2) qui permet à des processus différents d'ouvrir simultanément des fichiers, ou encore un processus I/O dédié qui écrit dans un seul fichier toutes les données envoyées par les autres processus.


## Utilisation des systèmes de fichiers

Certains des systèmes de fichiers ne sont pas disponibles à tous les utilisateurs.

### `/home` (`$HOME`)

Utilisé d'abord pour les fichiers d'un utilisateur, les logiciels communs ou les petits ensembles de données partagés par le groupe d'utilisateurs, pourvu que le quota de l'utilisateur ne soit pas dépassé ; dans le cas contraire utilisez plutôt `/scratch` ou `/project`.

En lecture seulement sur les nœuds de calcul.

### `/scratch` (`$SCRATCH`)

Utilisé d'abord pour les fichiers temporaires, les résultats de calcul et de simulations, et le matériel qui peut être obtenu ou recréé facilement. Peut aussi être utilisé pour une étape intermédiaire dans votre travail pourvu que cela ne cause pas trop d'opérations I/O ou trop de petits fichiers pour cette solution de stockage sur disque, auquel cas vous devriez utiliser `/bb` (burst buffer). Une fois que vous avez obtenu les résultats que vous voulez conserver à long terme, vous pouvez les transférer à `/project` ou `/archive`.

Purgé régulièrement ; aucune copie de sauvegarde n'est faite.

### `/project` (`$PROJECT`)

`/project` est destiné aux logiciels de groupe courants, aux grands ensembles de données statiques ou à tout contenu très coûteux à acquérir ou à régénérer par le groupe.

Le contenu de `/project` est censé rester relativement immuable au fil du temps.

Les fichiers temporaires ou transitoires doivent être conservés sur `/scratch` plutôt que sur `/project`. Les mouvements fréquents de données engendrent une surcharge et une consommation inutile des bandes sur le système de sauvegarde TSM, longtemps après leur suppression, et ceci en raison des politiques de conservation des sauvegardes et des versions supplémentaires conservées du même fichier. Le simple fait de renommer les répertoires principaux suffit à tromper le système et à lui faire croire qu'une arborescence de répertoires entièrement nouvelle a été créée et que l'ancienne a été supprimée. Réfléchissez donc soigneusement à vos conventions de nommage et respectez-les. Si vous abusez du système de fichiers `/projet` et l'utilisent comme `/scratch`, nous vous demanderons de procéder autrement. Notez que sur Niagara, `/project` est uniquement accessible aux groupes disposant de ressources allouées par concours.

### `/bb` (`$BBUFFER`)

`Burst buffer` est une alternative très rapide et performante à `/scratch`, is a very fast, sur disque SSD. Utilisez cette ressource si vous prévoyez beaucoup d'opérations I/O ou si vous remarquez une faible performance d'une tâche sur `/scratch` ou `/project` en raison d'un goulot d'étranglement des opérations I/O.

Plus d'information sur la [page wiki de SciNet](https://docs.alliancecan.ca/mediawiki/index.php?title=Burst_buffer).

### `/archive` (`$ARCHIVE`)

Espace de stockage nearline pour une copie temporaire de matériel semi-actif du contenu des systèmes de fichiers décrits plus haut. En pratique, les utilisateurs déchargent et rappellent du matériel dans le cours de leur travail ou quand les quotas sont atteints sur `/scratch` ou `/project`. Ce matériel peut demeurer sur HPSS de quelques mois jusqu'à quelques années.

Réservé aux groupes qui ont obtenu une allocation par suite des concours.

### `/dev/shm` (RAM)

Les nœuds ont un ramdisk plus rapide qu'un disque réel et que burst buffer. Jusqu'à 70 % du RAM du nœud (202 Go) peut être utilisé comme système de fichiers local temporaire. Ceci est très utile dans les premières étapes de migration de programmes d'un ordinateur personnel vers une plateforme de CHP comme Niagara, particulièrement quand le code utilise beaucoup d'opérations I/O. Dans ce cas, un goulot d'étranglement se forme, surtout avec les systèmes de fichiers parallèles comme GPFS (utilisé sur Niagara), puisque les fichiers sont synchronisés sur l'ensemble du réseau.

### `$SLURM_TMPDIR` (RAM)

Comme c'est le cas avec les grappes d'usage général Cedar et Graham, la variable d'environnement `$SLURM_TMPDIR` sera utilisée pour les tâches de calcul. Elle pointera sur RAMdisk et non sur les disques durs locaux. Le répertoire `$SLURM_TMPDIR` est vide lorsque la tâche commence et son contenu est supprimé après que la tâche est complétée.

### Espace de mémoire tampon d'éclatement temporaire par tâche (`$BB_JOB_DIR`)

Pour chaque tâche sur Niagara, le planificateur crée un répertoire temporaire sur le tampon d'éclatement appelé `$BB_JOB_DIR`. Le répertoire `$BB_JOB_DIR` sera vide lorsque votre tâche commence et son contenu sera supprimé une fois la tâche terminée. Ce répertoire est accessible depuis tous les nœuds d'une tâche.

`$BB_JOB_DIR` est un emplacement pour les applications qui génèrent plusieurs petits fichiers temporaires ou encore des fichiers qui sont fréquemment utilisés (c'est-à-dire avec des IOPS élevées), mais qui ne peuvent pas être contenus sur disque virtuel.

Il est important de noter que si ramdisk peut contenir les fichiers temporaires, c'est généralement un meilleur endroit que le burst buffer parce que la bande passante et le IOPS y sont beaucoup plus grands. Pour utiliser ramdisk, vous pouvez soit accéder `/dev/shm` directement, soit utiliser la variable d'environnement `$SLURM_TMPDIR`.

Les nœuds de calcul de Niagara n'ont pas de disques locaux ; `$SLURM_TMPDIR` est en mémoire (ramdisk), contrairement aux grappes généralistes Cedar et Graham où cette variable pointe vers un répertoire sur le disque SSD d'un nœud local.


## Quotas et purge

Familiarisez-vous avec les différents systèmes de fichiers, leur utilité et leur utilisation correcte. Ce tableau récapitule les points principaux.

| Système de fichiers | Quota                                                         | Taille des blocs | Durée | Sauvegarde | Sur nœuds de connexion | Sur nœuds de calcul |
|----------------------|-----------------------------------------------------------------|-----------------|-------|-------------|------------------------|-----------------------|
| `$HOME`              | 100 Go par utilisateur                                          | 1 Mo             | oui   | oui         | oui                     | lecture seule         |
| `$SCRATCH`           | 25 To par utilisateur (quota groupe non atteint)                | 16 Mo            | 2 mois | non         | oui                     | oui                    |
|                      | Groupes de 4 utilisateurs ou moins : 50 To pour le groupe       |                 |       |             |                        |                       |
|                      | Groupes de 11 utilisateurs ou moins : 125 To pour le groupe     |                 |       |             |                        |                       |
|                      | Groupes de 28 utilisateurs ou moins : 250 To pour le groupe     |                 |       |             |                        |                       |
|                      | Groupes de 60 utilisateurs ou moins : 400 To pour le groupe     |                 |       |             |                        |                       |
|                      | Groupes de plus de 60 utilisateurs : 500 To pour le groupe     |                 |       |             |                        |                       |
| `$PROJECT`           | Allocation de groupe                                           | 16 Mo            | oui   | oui         | oui                     | oui                    |
| `$ARCHIVE`           | Allocation de groupe                                           |                 |       | non         | non                     | non                    |
| `$BBUFFER`           | 10 To par utilisateur                                           | 1 Mo             | très court | non         | oui                     | oui                    |


**Quota Inode vs. Espace (PROJECT et SCRATCH)**

**Quota dynamique par groupe (SCRATCH)**

**Les nœuds de calcul n'ont pas de stockage local.**

**L'espace d'archivage est sur HPSS et n'est pas accessible sur les nœuds de connexion, de calcul ou de transfert de données de Niagara.**

**Sauvegarde signifie un instantané récent, pas une archive de toutes les données qui ont jamais existé.**

`$BBUFFER` signifie `Burst Buffer`, un niveau de stockage parallèle plus rapide pour les données temporaires.


### Combien d'espace disque me reste-t-il ?

La commande `/scinet/niagara/bin/diskUsage`, disponible sur les nœuds de connexion et les transferts de données, fournit des informations de plusieurs manières sur les systèmes de fichiers home, scratch, project et archive. Par exemple, combien d'espace disque est utilisé par vous-même et votre groupe (avec l'option -a), ou combien votre utilisation a changé sur une certaine période ("informations delta") ou vous pouvez générer des graphiques de votre utilisation au fil du temps. Veuillez consulter l'aide d'utilisation ci-dessous pour plus de détails.

Utilisation : `diskUsage [-h\|-?\| [-a] [-u <user>]`

`-h\|-?` : aide

`-a` : liste les utilisations de tous les membres du groupe

`-u <user>` : en tant qu'un autre utilisateur de votre groupe


Utilisez les commandes suivantes pour vérifier l'espace qui vous reste :

`/scinet/niagara/bin/topUserDirOver1000list` pour identifier les répertoires qui contiennent plus de 1000 fichiers,

`/scinet/niagara/bin/topUserDirOver1GBlist` pour identifier les répertoires qui contiennent plus de 1 Go de matériel.

**REMARQUE :** L'information sur l'utilisation et les quotas est mise à jour toutes les trois heures.


### Politique de purge du disque de travail

Afin de garantir qu'il y a toujours un espace important disponible pour les tâches en cours d'exécution, nous supprimons automatiquement les fichiers dans `/scratch` qui n'ont pas été accessibles ou modifiés depuis plus de 2 mois à compter de la date de suppression effective le 15 de chaque mois. Notez que nous avons récemment modifié la référence de découpage à `MostRecentOf(atime,ctime)`. Cette politique est sujette à révision en fonction de son efficacité. Plus de détails sur le processus de purge et sur la manière dont les utilisateurs peuvent vérifier si leurs fichiers seront supprimés suivent. Si vous avez des fichiers programmés pour suppression, vous devez les déplacer vers des emplacements plus permanents tels que votre serveur départemental ou votre espace `/project` ou dans HPSS (pour les PI qui ont soit été alloués de l'espace de stockage par le RAC sur le projet ou HPSS).

Le 1er de chaque mois, une liste des fichiers programmés pour la purge est produite, et une notification par courriel est envoyée à chaque utilisateur sur cette liste. Vous recevez également une notification sur le shell à chaque fois que vous vous connectez à Niagara. De plus, vers/ou autour du 12 de chaque mois, une 2e analyse produit une évaluation plus actuelle et une autre notification par courriel est envoyée. De cette façon, les utilisateurs peuvent vérifier qu'ils ont effectivement pris soin de tous les fichiers qu'ils devaient relocaliser avant la date limite de purge. Ces fichiers seront automatiquement supprimés le 15 du même mois, à moins qu'ils n'aient été accessibles ou relocalisés entre-temps. Si vous avez des fichiers programmés pour suppression, ils seront répertoriés dans un fichier dans `/scratch/t/todelete/current`, qui contient votre identifiant utilisateur et votre identifiant de groupe dans le nom de fichier. Par exemple, si l'utilisateur xxyz souhaite vérifier s'il a des fichiers programmés pour suppression, il peut exécuter la commande suivante sur un système qui monte `/scratch` (par exemple, un nœud de connexion scinet) :

`ls -1 /scratch/t/todelete/current \|grep xxyz`

Dans l'exemple ci-dessous, le nom de ce fichier indique que l'utilisateur xxyz fait partie du groupe abc, a 9 560 fichiers programmés pour suppression et qu'ils occupent 1,0 To d'espace :

```bash
[xxyz@nia-login03 ~]$ ls -1 /scratch/t/todelete/current |grep xxyz
 -rw-r----- 1 xxyz     root       1733059 Jan 17 11:46 3110001___xxyz_______abc_________1.00T_____9560files
```

Le fichier lui-même contient une liste de tous les fichiers programmés pour la suppression (dans la dernière colonne) et peut être visualisé avec des commandes standard comme `more`/`less`/`cat` - par exemple `more /scratch/t/todelete/current/3110001___xxyz_______abc_________1.00T_____9560files`

De même, vous pouvez vérifier tous les autres membres de votre groupe en utilisant la commande `ls` avec `grep` pour votre groupe. Par exemple, `ls -1 /scratch/t/todelete/current \|grep abc` listera les autres membres dont fait partie xxyz et dont les fichiers doivent être purgés le 15 du mois. Les membres d'un même groupe ont accès au contenu des autres.

**REMARQUE :** La préparation de ces évaluations prend plusieurs heures. Si vous modifiez l'heure d'accès/de modification d'un fichier entre-temps, cela ne sera pas détecté avant le cycle suivant. Une façon d'obtenir un retour immédiat est d'utiliser la commande `ls -lu` sur le fichier pour vérifier le ctime et `ls -lc` pour le mtime. Si l'heure d'accès/de création du fichier a été mise à jour entre-temps, à la date de purge le 15, il ne sera plus supprimé.


## Déplacer des données

Les données pour l'analyse et les résultats finaux doivent être déplacées vers et depuis Niagara. Il existe plusieurs façons d'y parvenir.


### Avec `rsync`/`scp`

Déplacer moins de 10 Go par les nœuds de connexion

Les nœuds de connexion et de copie sont visibles de l'extérieur de SciNet.

Utilisez `scp` ou `rsync` pour vous connecter à `niagara.scinet.utoronto.ca` ou `niagara.computecanada.ca` (aucune différence).

Il y aura interruption dans le cas de plus d'environ 10 Go.

Déplacer plus de 10 Go par les nœuds de copie

À partir d'un nœud de connexion, utilisez `ssh` vers `nia-datamover1` ou `nia-datamover2` ; de là, vous pouvez transférer de ou vers Niagara.

Vous pouvez aussi aller aux nœuds de copie de l'extérieur en utilisant login/scp/rsync.

`nia-datamover1.scinet.utoronto.ca`

`nia-datamover2.scinet.utoronto.ca`

Si vous faites souvent ceci, considérez utiliser Globus, un outil web pour le transfert de données.


### Utiliser Globus

Pour la documentation, consultez la [page wiki de Calcul Canada](https://docs.alliancecan.ca/mediawiki/index.php?title=Globus) et la [page wiki de SciNet](https://docs.alliancecan.ca/mediawiki/index.php?title=Globus).

Le point de chute Globus est `computecanada#niagara` pour Niagara et `computecanada#hpss` pour HPSS.


### Déplacer des données vers HPSS/Archive/Nearline

HPSS est conçu pour le stockage de longue durée.

HPSS est une solution de stockage sur bandes employée comme espace nearline par SciNet.

L'espace de stockage sur HPSS est alloué dans le cadre du concours d'allocation de ressources.


## Gestion de la propriété des fichiers et listes de contrôle d'accès

Par défaut, chez SciNet, les utilisateurs du même groupe ont déjà l'autorisation de lecture des fichiers des autres (pas d'écriture).

Vous pouvez utiliser une liste de contrôle d'accès (ACL) pour permettre à votre superviseur (ou à un autre utilisateur de votre groupe) de gérer les fichiers pour vous (c'est-à-dire créer, déplacer, renommer, supprimer), tout en conservant votre accès et vos autorisations en tant que propriétaire d'origine des fichiers/répertoires. Vous pouvez également laisser les utilisateurs d'autres groupes ou des groupes entiers accéder (lire, exécuter) à vos fichiers en utilisant le même mécanisme.


### Utilisation de `mmputacl`/`mmgetacl`

Vous pouvez utiliser les commandes natives de gpfs `mmputacl` et `mmgetacl`. Les avantages sont que vous pouvez définir l'autorisation "contrôle" et que les ACL de style POSIX ou NFS v4 sont prises en charge. Vous devrez d'abord créer un fichier `/tmp/supervisor.acl` avec le contenu suivant :

```
user::rwxc
group::----
other::----
mask::rwxc
user:[owner]:rwxc
user:[supervisor]:rwxc
group:[othegroup]:r-xc
```

Lancez ensuite les deux commandes :

1.  `$ mmputacl -i /tmp/supervisor.acl /project/g/group/[owner]`
2.  `$ mmputacl -d -i /tmp/supervisor.acl /project/g/group/[owner]`
    (chaque nouveau fichier/répertoire à l'intérieur de `[owner]` héritera également de la propriété `[supervisor]` par défaut ainsi que de la propriété `[owner]`, c'est-à-dire la propriété des deux par défaut, pour les fichiers/répertoires créés par `[supervisor]`)

`$ mmgetacl /project/g/group/[owner]` (pour déterminer les attributs ACL actuels)

`$ mmdelacl -d /project/g/group/[owner]` (pour supprimer toute ACL précédemment définie)

`$ mmeditacl /project/g/group/[owner]` (pour créer ou modifier une liste de contrôle d'accès GPFS) (pour que cette commande fonctionne, définissez la variable d'environnement EDITOR : `export EDITOR=/usr/bin/vi`)

**REMARQUES :**

Il n'y a pas d'option pour ajouter ou supprimer récursivement des attributs ACL à l'aide d'une commande intégrée de gpfs aux fichiers existants. Vous devrez utiliser l'option -i comme ci-dessus pour chaque fichier ou répertoire individuellement. Voici un exemple de script bash que vous pouvez utiliser à cette fin.

`mmputacl` ne remplacera pas les autorisations de groupe Linux d'origine pour un répertoire lorsqu'il est copié dans un autre répertoire déjà doté d'ACL, d'où la note "#effective:r-x" que vous pouvez voir de temps en temps avec `mmgetacf`. Si vous souhaitez donner des autorisations rwx à tous les membres de votre groupe, vous devez simplement utiliser la commande unix simple `chmod g+rwx`. Vous pouvez le faire avant ou après avoir copié le matériel d'origine dans un autre dossier avec les ACL.

Dans le cas de PROJECT, la personne responsable de votre groupe devra définir l'ACL appropriée au niveau `/project/G/GROUP` afin de permettre aux utilisateurs d'autres groupes d'accéder à vos fichiers.

ACL ne vous permet pas d'accorder des permissions pour des fichiers ou des répertoires qui ne vous appartiennent pas.

Nous vous recommandons vivement de ne jamais accorder d'autorisation d'écriture à d'autres personnes au niveau supérieur de votre répertoire personnel (`/home/G/GROUP/[owner]`), car cela compromettrait gravement votre confidentialité, et de désactiver l'authentification par clé SSH, entre autres. Si nécessaire, créez des sous-répertoires spécifiques sous votre répertoire personnel afin que d'autres puissent y accéder et manipuler les fichiers.

N'oubliez pas : `setfacl`/`getfacl` ne fonctionne que sur cedar/graham, car ils ont lustre. Sur niagara, vous devez utiliser la commande `mm*` uniquement pour GPFS : `mmputacl`, `mmgetacl`, `mmdelacl`, `mmeditacl`.

Pour plus d'information, consultez `mmputacl` et `mmgetacl`.


### Script ACL récursif

Vous pouvez utiliser et adapter [cet exemple de script bash](http://csngwinfo.in2p3.fr/mediawiki/index.php/GPFS_ACL) pour ajouter ou supprimer récursivement des attributs ACL à l'aide des commandes intégrées de GPFS.

Gracieuseté de Agata Disks ([http://csngwinfo.in2p3.fr/mediawiki/index.php/GPFS_ACL](http://csngwinfo.in2p3.fr/mediawiki/index.php/GPFS_ACL)).


Récupéré de "[https://docs.alliancecan.ca/mediawiki/index.php?title=Data\_management\_at\_Niagara/fr&oldid=177147](https://docs.alliancecan.ca/mediawiki/index.php?title=Data_management_at_Niagara/fr&oldid=177147)"
