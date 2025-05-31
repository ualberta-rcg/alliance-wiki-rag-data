# Multi-Instance GPU

Plusieurs logiciels sont incapables d'exploiter pleinement les GPU modernes tels que les A100 et H100 de NVidia. La technologie multi-instances (MIG pour Multi-Instance GPU) permet de partitionner un seul GPU en plusieurs instances, faisant ainsi de chacune un GPU virtuel complètement indépendant. Chacune des instances de GPU dispose alors d'une portion des ressources de calcul et de la mémoire du GPU d'origine, le tout détaché des autres instances par des protections sur puce.

Les instances d'un GPU sont moins gourmandes, ce qui se reflète par une utilisation moins rapide de votre priorité de calcul. Les tâches soumises sur une instance plutôt que sur un GPU entier utilisent moins de la priorité qui vous est allouée et vous pourrez exécuter plus de tâches avec un temps d'attente plus court.

## Pourquoi choisir un GPU entier ou une instance de GPU

Les tâches qui utilisent moins de la moitié de la puissance de calcul d'un GPU entier et moins de la moitié de la mémoire GPU disponible doivent être évaluées et testées sur une instance. Dans la plupart des cas, ces tâches s'exécutent tout aussi rapidement sur une instance et consomment moins de la moitié des ressources de calcul.

Voir [Quand migrer une tâche sur une instance](#quand-migrer-une-tâche-sur-une-instance) ci-dessous.

## Limites de la technologie

La technologie MIG ne prend pas en charge la communication interprocessus CUDA qui optimise le transfert de données via NVLink et NVSwitch. Cette limite diminue aussi l'efficacité de la communication entre les instances. En conséquence, le lancement d'un exécutable sur plusieurs MIG à la fois n'améliore pas la performance et doit être évité.

Veuillez noter que les API graphiques ne sont pas prises en charge (par exemple OpenGL, Vulkan, etc.); voir *Application Considerations*.

Les tâches avec GPU qui nécessitent de nombreux cœurs CPU par GPU peuvent également nécessiter un GPU entier au lieu d'une instance. Le nombre maximum de cœurs CPU par instance dépend du nombre maximum de cœurs CPU par GPU entier et des profils MIG qui sont configurés. Ces deux caractéristiques varient d'une grappe à l'autre et d'un nœud GPU à l'autre.

## Configurations disponibles

Depuis décembre 2024, tous les types d'instances GPU sont offerts sur Narval. Plusieurs configurations et profils MIG sont possibles, mais les suivantes sont présentement disponibles :

*   `1g.5gb`
*   `2g.10gb`
*   `3g.20gb`
*   `4g.20gb`

Le nom du profil indique la taille de l'instance, par exemple `3g.20gb` est dotée de 20Go de mémoire vive et sa performance est égale à ⅜ de la performance de calcul d’un A100-40gb entier. Le fait de nécessiter moins de puissance diminue l’impact sur votre allocation et sur la priorité assignée à vos tâches.

Sur Narval, le maximum recommandé de cœurs CPU et de mémoire système par instance est de :

*   `1g.5gb` : maximum 2 cores et 15Go
*   `2g.10gb` : maximum 3 cores et 31Go
*   `3g.20gb` : maximum 6 cores et 62Go
*   `4g.20gb` : maximum 6 cores et 62Go

Pour demander une instance d’un profil particulier, ajoutez le paramètre `--gres`.

*   `1g.5gb` : `--gres=gpu:a100_1g.5gb:1`
*   `2g.10gb` : `--gres=gpu:a100_2g.10gb:1`
*   `3g.20gb` : `--gres=gpu:a100_3g.20gb:1`
*   `4g.20gb` : `--gres=gpu:a100_4g.20gb:1`

Remarque : Pour l'ordonnanceur de Narval, ajoutez le préfixe `a100_` au nom du profil.

## Exemples

Pour demander une instance de 20Go à 3/8 de la puissance pour une tâche interactive d’une durée d’une (1) heure :

```bash
[name@server ~]$ salloc --account=def-someuser --gres=gpu:a100_3g.20gb:1 --cpus-per-task=2 --mem=40gb --time=1:0:0
```

Pour demander une instance de 20Go à 4/8 de la puissance pour un script de tâches en lot d’une durée de 24 heures qui utilise le maximum recommandé de cœurs et de mémoire système :

**File :** `a100_4g.20gb_mig_job.sh`

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=6    # There are 6 CPU cores per 3g.20gb and 4g.20gb on Narval.
#SBATCH --mem=62gb           # There are 62GB GPU RAM per 3g.20gb and 4g.20gb on Narval.
#SBATCH --time=24:00:00
hostname
nvidia-smi
```

## Quand migrer une tâche sur une instance

L'historique de vos tâches est disponible sur le portail d'utilisation de Narval (préparation en cours).

La consommation d’énergie est un bon indicateur de la puissance de calcul totale demandée au GPU. Par exemple, le travail suivant nécessitait un GPU A100 entier avec une enveloppe thermique (TDP) maximale de 400 W, mais n'utilisait que 100 W en moyenne, soit seulement 50 W de plus que la consommation au repos :

*(Image: Consommation d'énergie sur un GPU A100)*

Les fonctionnalités du GPU peuvent également fournir des informations sur son utilisation dans les cas où la consommation d'énergie n'est pas suffisante. Dans ce prochain exemple de tâche, le graphique d'utilisation du GPU confirme la conclusion du graphique de consommation d'énergie du GPU selon laquelle la tâche utilise moins de 25 % de la puissance de calcul disponible d'un GPU A100 entier :

*(Image: Utilisation d'un GPU A100)*

Il faut aussi tenir compte de la quantité maximale de mémoire GPU et de la quantité moyenne de cœurs CPU nécessaires pour exécuter la tâche. Dans le prochain exemple, la tâche utilise un maximum de 3Go de mémoire GPU sur les 40Go d'un GPU A100.

*(Image: Utilisation de la mémoire du GPU A100)*

La tâche a aussi été lancée en utilisant un seul cœur de processeur. En tenant compte de ces trois métriques, nous voyons que la tâche pourrait facilement s'exécuter sur une instance de 3g.20Go ou de 4g.20Go avec de la puissance et de la mémoire à revendre.

Un autre moyen de surveiller l'utilisation d'une tâche en cours d'exécution consiste à se connecter au nœud sur lequel la tâche se trouve et utiliser `nvidia-smi` pour lire les métriques du GPU en temps réel. Cela ne fournira pas de valeurs maximales et moyennes pour la mémoire et la puissance de toute la tâche, mais pourrait être utile pour identifier une sous-utilisation du GPU.

## Puis-je utiliser plusieurs instances sur le même GPU?

Non. En principe, il est possible de le faire, mais nos systèmes ne prennent pas en charge cette fonctionnalité. Si vous voulez exécuter plusieurs tâches indépendantes sur un GPU, utilisez MPS plutôt que MIG.
