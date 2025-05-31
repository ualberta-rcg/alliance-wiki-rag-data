# Foire aux questions

## Mot de passe oublié

Pour réinitialiser votre mot de passe pour accéder aux grappes nationales de l'Alliance, cliquez sur [https://ccdb.computecanada.ca/security/forgot](https://ccdb.computecanada.ca/security/forgot). Il vous est impossible de réinitialiser votre mot de passe tant que votre premier rôle n'est pas approuvé.

## Copier-coller

Dans un terminal Linux, vous ne pouvez pas utiliser [Ctrl]+C pour copier du texte parce que [Ctrl]+C signifie Annuler (Cancel) ou Interrompre (Interrupt) et met fin au programme en cours d'exécution.

Sous Windows et Linux dans la plupart des cas, vous pouvez utiliser plutôt [Ctrl]+[Insert] pour copier et [Shift]+[Insert] pour coller, selon votre programme de terminal.

Sous macOS, utilisez [Cmd]+C et [Cmd]+V comme à l'habitude.

Selon le logiciel de terminal utilisé, sélectionnez simplement le texte pour l'entrer dans le presse-papiers et collez-le ensuite avec un clic de droite ou un clic du milieu (dépendant du paramétrage par défaut).

## Fichiers texte : caractère en fin de ligne

Les systèmes d'exploitation ont des conventions différentes pour signaler la fin de ligne dans un fichier texte brut ASCII. Sous Windows, chacune des lignes se termine par un caractère *retour de chariot* invisible, ce qui entraîne certains problèmes lorsque le fichier est lu dans un environnement Linux. Pour contrer ceci, vous pouvez :

* créer et éditer les fichiers texte sur la grappe en utilisant un éditeur standard Linux comme emacs, vim ou nano ;
* avec des fichiers texte Windows, lancer la commande `dos2unix <filename>` sur un nœud de connexion pour convertir les caractères de fin de ligne au format approprié.

## Lenteur de sauvegarde avec mon éditeur

### Emacs

Pour sauvegarder les fichiers, Emacs utilise l'appel système `fsync` pour diminuer le risque de perte de données en cas de panne du système. Cet ajout de fiabilité a cependant un inconvénient : plusieurs secondes sont quelquefois nécessaires pour sauvegarder un petit fichier dans un système de fichiers partagé (par exemple `/home`, `/scratch`, `/project`) sur une de nos grappes. Si ce ralentissement nuit à votre travail, vous pouvez améliorer la performance en ajoutant la ligne suivante à votre fichier `~/.emacs` :

```
(setq write-region-inhibit-fsync t)
```

Pour plus d'information, voir [Customize save in Emacs](Customize save in Emacs).


## Transferts entre les systèmes de fichiers /scratch, /home et /project

Sur nos grappes d'usage général, les quotas pour `/scratch` et `/home` sont par utilisateur, alors qu'ils sont par projet pour le système de fichiers `/project`. Comme les quotas pour le système de fichiers Lustre sont présentement basés sur le groupe propriétaire des fichiers, il faut s'assurer que les fichiers soient associés au bon groupe.

Pour `/scratch` et `/home`, le bon groupe est typiquement le groupe qui possède le même nom que votre nom d'utilisateur.

Pour `/project`, le nom du groupe a la forme `prefix-piusername`, où `prefix` est un de `def`, `rrg` ou `rpp`.

### De /scratch vers /home

Puisque les quotas de ces systèmes de fichiers sont basés sur votre groupe personnel, vous devriez pouvoir effectuer les transferts avec :

```bash
[name@server ~]$ mv /scratch/some_file $HOME/some_file
```

### De /scratch ou /home vers /project

De `/scratch` ou `/home` vers `/project`, n'utilisez pas la commande `mv`. Utilisez plutôt les commandes `cp` ou `rsync`.

Il est très important que `cp` et `rsync` s'exécutent correctement pour que les fichiers copiés dans l'espace `/project` possèdent la propriété de groupe correcte.

Avec `cp`, n'utilisez pas l'option d'archivage `-a`.

Avec `rsync`, assurez-vous d'utiliser les options `--no-g --no-p` ainsi :

```bash
[name@server ~]$ rsync -axvH --no-g --no-p /scratch/some_directory /projects/<project>/some_other_directory
```

Une fois les fichiers copiés correctement, ils peuvent être supprimés de votre espace `/scratch`.

### De /project à /scratch ou /home

Évitez la commande `mv` : nous recommandons plutôt `cp`, ou `rsync`.

Il est très important que ces commandes s'effectuent correctement pour que les fichiers conservent la même propriété de groupe qu'avec `/project`.

Avec `cp`, n'utilisez pas l'option d'archivage `-a`.

Avec `rsync`, entrez les options comme suit :

```bash
[name@server ~]$ rsync -axvH --no-g --no-p /projects/<project>/some_other_directory /scratch/some_directory
```

### De /project à /project

N'utilisez jamais la commande `mv` ; nous vous recommandons plutôt `cp` ou `rsync`.

Il est très important de faire exécuter correctement `cp` ou `rsync` pour faire en sorte que la propriété de groupe des fichiers copiés soit correcte.

Avec `cp`, n'utilisez pas l'option d'archivage `-a`.

Avec `rsync`, spécifiez les options `--no-g --no-p` comme suit :

```bash
[name@server ~]$ rsync -axvH --no-g --no-p /projects/<project>/some_other_directory /projects/<project2>/some_directory
```

Quand vos données auront été copiées, n'oubliez pas de supprimer celles de la source.

## Message "Disk quota exceeded"

Voir aussi la page [Espace projet](Espace projet).

Certains reçoivent ce message d'erreur ou un autre message similaire au sujet du quota en rapport avec leurs répertoires de l'espace `/project`. Des difficultés ont aussi été rapportées lors du transfert de fichiers vers leur répertoire `/project` à partir d'une autre grappe. Plusieurs de ces cas sont dus à des problèmes de propriété des fichiers.

Pour savoir si vous avez atteint ou dépassé le quota, utilisez `diskusage_report`.

```bash
[ymartin@cedar5 ~]$ diskusage_report
```

Cet exemple illustre un problème fréquent : l'espace projet de l'utilisateur `ymartin` contient trop de données dans des fichiers qui appartiennent au groupe `ymartin`. Ces données devraient se trouver dans des fichiers appartenant à `def-zrichard`. Pour connaître les groupes de projets que vous pouvez utiliser, lancez la commande `stat -c %G $HOME/projects/*/`.

En ce qui a trait aux deux dernières lignes, `Project (group ymartin)` décrit les fichiers qui appartiennent au groupe `ymartin` ; notez que le nom du groupe est le même que celui de l’utilisateur. Ce dernier est le seul membre du groupe et le quota de 2048Ko pour son groupe est très bas.  `Project (group def-zrichard)` décrit les fichiers qui appartiennent au groupe du projet. Il est possible que votre compte soit associé à plusieurs groupes de projet, dont les noms sont sous la forme `def-zrichard`, `rrg-someprof-ab`, ou `rpp-someprof`.

Dans cet exemple, les fichiers ont été associés au propriétaire du groupe `ymartin` plutôt qu’au propriétaire du groupe `def-zrichard`, ce qui est inattendu et non souhaitable.

Les nouveaux fichiers et répertoires créés dans `/project` sont automatiquement associés à un groupe du projet. Les raisons les plus fréquentes pour lesquelles cette association est fautive sont que :

* les fichiers et répertoires sont déplacés d’un espace `/home` à un espace `/project` en utilisant la commande `mv` ; pour éviter ceci, voyez ci-dessus la section De `/scratch` vers `/home` ;
* les fichiers sont transférés à partir d’une autre grappe à l’aide de `rsync` ou de `scp` avec une option forçant de conserver les mêmes caractéristiques de propriété ; vérifiez donc les options que vous avez sélectionnées dans votre application de transfert de données ;
* le bit `setgid` n'est pas activé pour vos répertoires `/project`.

### Solution

Si votre répertoire `/project` contient des données qui appartiennent au mauvais groupe, utilisez `find` pour afficher les fichiers.

```bash
lfs find ~/projects/*/ -group $USER
```

Changez ensuite la propriété de `$USER` au groupe de projet, par exemple :

```bash
chown -h -R $USER:def-professor -- ~/projects/def-professor/$USER/
```

Activez le bit `setgid` pour tous les répertoires pour que les nouveaux fichiers héritent de l'appartenance au groupe du répertoire, par exemple :

```bash
lfs find ~/projects/def-professor/$USER -type d -print0 | xargs -0 chmod g+s
```

Enfin, assurez-vous que les permissions des répertoires de l'espace projet sont correctes avec :

```bash
chmod 2770 ~/projects/def-professor/
chmod 2700 ~/projects/def-professor/$USER
```

### Autre explication

Chaque fichier sous Linux appartient à la fois à un utilisateur et à un groupe. Par défaut, tout fichier que vous créez vous appartient (`username`) et votre groupe porte le même nom. Le propriétaire est donc `username:username`. Votre groupe a été créé au même moment où votre compte a été créé et vous êtes le seul utilisateur dans ce groupe. La propriété du fichier vaut pour votre répertoire `/home` et pour l'espace `/scratch`.

Les quotas pour `/home` et `/scratch` sont établis sur la base du `username`. Les deux autres lignes sont pour les groupes `username` et `def-professor` dans l'espace `/project`. Il n'est pas important que les utilisateurs soient propriétaires des fichiers dans cet espace ; par contre le groupe propriétaire des fichiers déterminera la limite du quota.

Vous remarquez que vos fichiers occupent déjà 34Go dans l'espace `/project` et que la limite pour le groupe `username` est fixée à 2Mo, ce qui est très peu. Pour cette raison, vous ne pouvez y enregistrer plus de données parce que le groupe propriétaire n'a pas reçu une allocation pour plus d'espace. Par contre, l'allocation pour le groupe `def-professor` (le groupe de votre professeur) n'utilise presque pas d'espace, mais dispose en fait de 1To. Les fichiers ici devraient appartenir à `username:def-professor`.

Dépendant du logiciel et de la méthode de copie des fichiers, le logiciel respectera la propriété du répertoire et désignera le bon groupe ou il conservera la propriété du fichier source ; ce dernier comportement n'est pas souhaitable. Vos données initiales appartiennent probablement à `username:username` ; en les déplaçant, elles devraient appartenir à `username:def-professor`, mais le logiciel insiste pour ne pas changer la propriété, ce qui est un problème.


## Message d'erreur "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"

Vous pourriez recevoir ce message d'erreur si l'ordonnanceur est surchargé (voir la page [Exécuter des tâches](Exécuter des tâches)). Nous tentons toujours d'augmenter la tolérance de Slurm à cet effet et d'éliminer les sources de surcharge ponctuelle, mais ceci est un projet de longue haleine. Notre recommandation est d'attendre environ une minute, puis d'utiliser `squeue -u $USER` pour voir si la tâche soumise paraît. Si la tâche n'est pas listée, soumettez-la de nouveau. Notez que ce message survient dans certains cas même lorsque Slurm a accepté la tâche.

## Temps d’attente des tâches

Vous pouvez savoir pourquoi vos tâches ont le statut PD (*pending* pour en attente), en exécutant la commande `squeue -u <username>`. La colonne `(REASON)` contient `Resources` ou `Priority`.

*   **Resources**: la grappe est très occupée ; vous pouvez soit attendre, soit soumettre une tâche qui exige moins de ressources en termes de CPU/nœud, GPU, mémoire ou temps d’exécution.
*   **Priority**: la tâche est en attente en raison de sa basse priorité, ce qui survient lorsque vous et les autres membres du groupe avez utilisé plus que votre juste part des ressources récemment ; vous pouvez faire le suivi de votre utilisation des ressources avec la commande `sshare` (voir [Politique d’ordonnancement des tâches](Politique d’ordonnancement des tâches)).

## Messages "Nodes required for job are DOWN, DRAINED or RESERVED for jobs in higher priority partitions" ou "ReqNodeNotAvailable"

Il est possible qu'un de ces messages s’affiche dans le champ *Reason* du fichier de résultat de `squeue` pour une tâche en attente ; ceci se produit avec Slurm 19.05 et signifie qu'un ou plusieurs des nœuds que Slurm pouvait utiliser sont en panne, ont été mis hors service, ou encore sont réservés pour d’autres tâches. Ces cas sont fréquents avec les grappes de grande capacité qui connaissent un fort achalandage. Ces messages signifient effectivement la même chose que la raison *Ressources* de la version Slurm 17.11. Ce ne sont pas des messages d'erreur ; les tâches sont dans la queue et seront éventuellement traitées.

## Précision de START_TIME avec squeue

Par défaut, la commande `squeue` ne montre pas le moment où une tâche doit être lancée, mais il est possible de le savoir avec une option. Comme les conditions sont constamment en changement, le moment prévu par Slurm pour le lancement d’une tâche n’est jamais exact et donc pas très utile.

L'ordonnanceur Slurm calcule un moment de début (START\_TIME) dans le futur pour les tâches en attente qui sont de haute priorité sur la base :

*   des ressources qui seront libérées à la fin des tâches en cours, et
*   des ressources qui seront demandées par les tâches en attente dont la priorité est plus élevée.

La valeur de START\_TIME n'est plus valide si :

*   certaines tâches se terminent plus tôt que prévu et libèrent des ressources, ou
*   l'ordre de priorité est modifié, par exemple quand des tâches sont annulées ou qu'une nouvelle tâche avec une priorité plus élevée est soumise.

Sur nos grappes d'usage général, de nouvelles tâches sont soumises toutes les cinq secondes environ et de 30 à 50 % des tâches se terminent plus tôt que prévu ; Slurm revoit donc souvent l'ordre d'exécution des tâches qui lui sont soumises.

Pour la plupart des tâches en attente, la valeur de START\_TIME est N/A (*not available*), ce qui signifie que Slurm n'essaie pas d'en fixer le moment du début. Pour les tâches qui sont en cours, la valeur de START\_TIME obtenue par `squeue` est précise.

## Fichiers .core

Dans certains cas, un fichier binaire avec l'extension `.core` est créé quand un programme se termine anormalement ; il contient un instantané de l'état du programme au moment où il s'est terminé. Ce fichier est utile pour le débogage, mais n'est d'aucun intérêt pour les utilisateurs ; il n'est qu'un signe d'un problème dans le déroulement du programme, ce qui est généralement indiqué dans le résultat en sortie de la tâche. Les fichiers `.core` peuvent être supprimés ; ajoutez la ligne `ulimit -c 0` à la fin du fichier `$HOME/.bashrc` pour que ces fichiers ne soient plus créés.

## Bibliothèque introuvable

À l'exécution, les paquets binaires précompilés qui sont installés dans votre répertoire `$HOME` peuvent produire une erreur semblable à `/lib64/libc.so.6: version 'GLIBC_2.18' not found`. Pour la solution, voir [Installer des paquets binaires](Installer des paquets binaires).

## Quelles sont vos mesures pour la gestion des données sensibles ?

Nos grappes ne sont pas spécifiquement conçues pour garantir la sécurité des données personnelles, privées ou sensibles. Nous appliquons cependant les meilleures pratiques pour les systèmes partagés et accordons beaucoup d’attention à l’intégrité, la confidentialité et la disponibilité des données. Toutefois, certains ensembles de données doivent être traités avec des ressources qui sont certifiées selon des standards de sécurité particuliers et il est de la responsabilité des chercheuses et chercheurs de voir à ce que ces exigences soient respectées. À cet effet, veuillez prendre connaissance de l’article 5.2 de notre [Politique de protection des données et des renseignements personnels](Politique de protection des données et des renseignements personnels) et du paragraphe 3.2 des [Conditions d’utilisation](Conditions d’utilisation).

Pour plus d'information, voir [Protection des données, confidentialité et respect de la vie privée](Protection des données, confidentialité et respect de la vie privée).
