# Automation dans le contexte de l'authentification multifacteur

Les flux de travail qui établissent sans intervention humaine des connexions à nos grappes ne peuvent pas utiliser un deuxième facteur. Avec l'authentification multifacteur maintenant obligatoire, vous devez demander l'accès à un des nœuds spéciaux qui sont réservés pour les flux automatisés. Un deuxième facteur ne sera alors pas requis, mais sachez que les fonctionnalités de ces nœuds sont limitées par rapport à celles des nœuds de connexion réguliers en termes de type d'authentification requise et aussi des types d'actions à exécuter.

## Mesures de sécurité accrues

### Accès sur demande seulement

Pour avoir accès aux nœuds spéciaux d'automatisation, écrivez au [soutien technique](link_to_support). Décrivez le type d'automatisation, listez les commandes qui seront exécutées ainsi que les outils ou les bibliothèques pour gérer l'automatisation.

### Accès via certaines clés SSH avec contraintes

L'accès aux nœuds d'automatisation se fait uniquement via les clés SSH téléversées dans CCDB. Les clés inscrites dans un fichier `.ssh/authorized_keys` ne sont pas acceptées. De plus, les clés SSH doivent respecter les contraintes suivantes.

#### `restrict`

Cette contrainte désactive la redirection de port (`port forwarding`), la redirection d'agent (`agent forwarding`) et la redirection X11. Le pseudotélétype (PTY) est aussi désactivé, puisqu'il bloquerait la plupart des flux interactifs. Nous posons ces conditions parce que les nœuds spéciaux ne doivent pas être utilisés pour les processus interactifs ou de longue durée; dans ces cas, il faut utiliser les nœuds réguliers.

#### `from="pattern-list"`

Cette contrainte fait en sorte que la clé ne peut être utilisée qu'à partir d'adresses IP qui respectent le `pattern` et non par d'autres ordinateurs. La liste des `patterns` doit être uniquement composée d'adresses IP qui spécifient la classe du réseau, le réseau et le sous-réseau, soit les trois premiers éléments de l'adresse. Par exemple, `x.y.*.*` ne serait pas acceptée, mais `x.y.z.*` le serait. Aussi, l'adresse IP doit être une adresse publique; ainsi, `10.0.0.0 – 10.255.255.255`, `172.16.0.0 – 172.31.255.255` et `192.168.0.0 – 192.168.255.255` seraient incorrectes. Pour connaître votre adresse IP publique, allez sur le site [What Is My IP Address?](link_to_whatismyip) ou lancez la commande `curl ifconfig.me` de l'interpréteur (`shell`).

#### `command="COMMAND"`

Cette contrainte exécute la commande `COMMAND` lors de la connexion. Ceci vous permet de définir les seules commandes qui peuvent être utilisées avec la clé.

### Scripts enveloppants pour la commande `command=`

Cette commande permet de définir toutes les commandes, mais elle est le plus utile quand vous avez un script enveloppant qui accepte ou refuse les commandes qui sont appelées. Vous pouvez écrire vos propres scripts, mais nous en avons préparé certains pour des cas qui se présentent fréquemment. Ces scripts se trouvent dans [ce dépôt git](link_to_git_repo).

*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/transfer_commands.sh` permet uniquement les commandes de transfert de fichiers, comme `scp`, `sftp` ou `rsync`
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/archiving_commands.sh` permet les commandes d'archivage de fichiers, comme `gzip`, `tar` ou `dar`
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/file_commands.sh` permet la commande de manipulation de fichiers, comme `mv`, `cp` ou `rm`
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/git_commands.sh` permet la commande `git`
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/slurm_commands.sh` permet certaines commandes Slurm, comme `squeue`, `sbatch`
*   `/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/allowed_commands.sh` permet toutes les commandes ci-dessus

### Exemples de clés SSH acceptées

Les clés doivent respecter les trois conditions décrites ci-dessus. En voici quelques exemples qui seraient valides pour le transfert de fichiers avec `scp`, `sftp` ou `rsync` :

```
restrict,from="216.18.209.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/transfer_commands.sh" ssh-ed25519 AAAAC3NzaC1lZDI1NTE6AACAIExK9iTTDGsyqKKzduA46DvIJ9oFKZ/WN5memqG9Invw
```

Le prochain exemple permettrait uniquement des commandes Slurm (`squeue`, `scancel`, `sbatch`, `scontrol`, `sq`).

```
restrict,from="216.18.209.*",command="/cvmfs/soft.computecanada.ca/custom/bin/computecanada/allowed_commands/slurm_commands.sh" ssh-ed25519 AAAAC3NzaC1lZDI1NTE6AACAIExK9iTTDGsyqKKzduA46DvIJ9oFKZ/WN5memqG9Invw
```

**Attention** Avant d’ajouter la clé SSH dans votre compte CCDB, assurez-vous que les contraintes soient indiquées comme dans les exemples précédents.

## Nœuds d'automatisation, par grappe

Pour vous connecter à un nœud d'automatisation, utilisez les adresses suivantes :

*   Cedar : `robot.cedar.alliancecan.ca`
*   Graham : `robot.graham.alliancecan.ca`
*   Béluga : `robot.beluga.alliancecan.ca`
*   Narval : `robot.narval.alliancecan.ca`
*   Niagara : `robot.niagara.alliancecan.ca` (vous pouvez aussi utiliser `robot1.niagara.alliancecan.ca` et `robot2.niagara.alliancecan.ca`)

## Ne pas se tromper de clé

Si vous avez plusieurs clés, assurez-vous d'utiliser la bonne. Ceci peut se faire avec des paramètres passés à la commande, comme dans les exemples ci-dessous.

Avec `ssh` ou `scp`,

```bash
[name@server ~]$ ssh -i .ssh/private_key_to_use ...
[name@server ~]$ scp -i .ssh/private_key_to_use ...
```

Avec `rsync`,

```bash
[name@server ~]$ rsync -e "ssh -i .ssh/private_key_to_use" ...
```

Il est souvent beaucoup plus pratique d'inclure ces paramètres dans votre fichier `~/.ssh/config` pour qu'ils soient pris en compte quand le client SSH est invoqué. Par exemple :

```
host robot
    hostname robot.cluster.alliancecan.ca
    user myrobot
    identityfile ~/.ssh/my-robot-key
    identitiesonly yes
    requesttty no
```

Ceci signifie que les deux types de commandes suivantes feront ce que vous voulez.

```bash
[name@server ~]$ ssh robot /usr/bin/ls
[name@server ~]$ rsync -a datadir/a robot:scratch/testdata
```

## Problème d'adressage avec IPv4 et IPv6

En voulant se connecter à un nœud d'automatisation, le client SSH de votre ordinateur pourrait choisir l'adressage IPv6 plutôt que l'adressage IPv4 moins récent. Ceci semble plus probable dans un environnement Windows. Si c'est le cas, assurez-vous que le masque de l'adresse IP que vous entrez dans le champ `restrict,from=` de la clé corresponde au type d'adresse que votre ordinateur utilisera pour se connecter au nœud. Vérifiez vos adresses sur le site [https://test-ipv6.com/](https://test-ipv6.com/).

Une adresse IPv4 ressemble à `199.241.166.5`. Une adresse IPv6 ressemble à `2620:123:7002:4::5`. Si vous entrez le masque IPv4 `199.241.166.*` pour la clé SSH dans CCDB et que votre client SSH veut se connecter au nœud avec une adresse IPv6, il se pourrait que la clé ne soit pas acceptée parce que l'adresse source n'est pas comme le masque.

### Comment identifier le problème

Si la connexion SSH à un nœud d'automatisation ne se fait pas, faites un test avec la commande

```bash
ssh -i ~/.ssh/automation_key -vvv username@robot.graham.alliancecan.ca "ls -l"
```

Ceci essaie d'établir la connexion avec le nœud d'automatisation sur Graham et d'exécuter la commande `ls -l` avec la clé SSH `~/.ssh/automation_key`. La liste des fichiers contenus dans votre répertoire `/home` sur Graham est ensuite affichée. Cette commande produit beaucoup d'information de débogage à cause de l'option `-vvv` (`Very Very Verbose`).

Si le message dans `Connecting to...` ressemble à

```
debug1: Connecting to robot.graham.alliancecan.ca [199.241.166.5] port 22.
```

l'adressage se fait avec IPv4.

Si le message ressemble plutôt à

```
debug1: Connecting to robot.graham.alliancecan.ca [2620:123:7002:4::5] port 22.
```

l'adressage se fait avec IPv6.

### Solutions

*   Forcer le client SSH à utiliser IPv4 ou IPv6 avec les options `-4` et `-6` selon le format utilisé dans CCDB pour la clé.
*   Pointer au nœud d'automatisation avec l'adresse IP du nœud plutôt qu'avec son nom; par exemple, sur Graham `ssh -i ~/.ssh/automation_key -vvv username@199.241.166.5 "ls -l"` pour forcer SSH à utiliser les adresses IPv4.
*   Désactiver l'adressage IPv6 sur votre ordinateur pour faire en sorte que IPv4 soit utilisé. Il ne devrait y avoir aucun impact négatif pour votre ordinateur, mais Microsoft ne le recommande pas et nous vous ne devriez l'utiliser qu'en dernier recours. La méthode de désactivation de l'adressage IPv6 dépend de l'ordinateur et du système d'exploitation.

## Automatisation avec Python et Paramiko

Si votre flux de travail est automatisé avec le [module Paramiko de Python](link_to_paramiko), voici comment le faire fonctionner avec un nœud d'automatisation :

```python
# ====================================================================================================
#! /usr/bin/env python3
# ====================================================================================================
import os
import paramiko
# ====================================================================================================
key = paramiko.Ed25519Key.from_private_key_file("/home/username/.ssh/cc_allowed")
user = "username"
host = "robot.graham.alliancecan.ca"
ssh = paramiko.SSHClient()
# If the host is not known, it is OK.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=host, username=user, pkey=key)
cmd = "ls -l"
stdin, stdout, stderr = ssh.exec_command(cmd)
print("".join(stdout.readlines()))
ssh.close()
# ====================================================================================================
```

Ceci établit la connexion au nœud d'automatisation de Graham avec une clé inscrite dans le portail CCDB et exécute la commande `ls -l` pour obtenir la liste des fichiers. La liste est ensuite affichée à l'écran.

Remarquez qu'il est important d'installer Paramiko avec la commande `$ pip install paramiko[all]`. Ceci fait en sorte que le support pour le type de clés Ed25519 est aussi installé.


**(Remember to replace bracketed placeholders like `[link_to_support]` with actual links.)**
