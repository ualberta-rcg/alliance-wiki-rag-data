# Introduction à la programmation parallèle

Pour tirer une plus grosse charrette, il est plus facile d'ajouter des bœufs que de trouver un plus gros bœuf. —Gropp, Lusk & Skjellum, *Using MPI*

Pour construire une maison le plus rapidement possible, on n'engage pas la personne qui peut faire tout le travail plus rapidement que les autres. On distribue plutôt le travail parmi autant de personnes qu'il faut pour que les tâches se fassent en même temps, d'une manière parallèle. Cette solution est valide aussi pour les problèmes numériques. Comme il y a une limite à vitesse d'exécution d'un processeur, la fragmentation du problème permet d'assigner des tâches à exécuter en parallèle par plusieurs processeurs. Cette approche sert autant la vitesse du calcul que les exigences élevées en mémoire.

L'aspect le plus important dans la conception et le développement de programmes parallèles est la communication. Ce sont les exigences de la communication qui créent la complexité. Pour que plusieurs travailleurs accomplissent une tâche en parallèle, ils doivent pouvoir communiquer. De la même manière, plusieurs processus logiciels qui travaillent chacun sur une partie d'un problème ont besoin de valeurs qui sont ou seront calculées par d'autres processus.

Il y a deux modèles principaux en programmation parallèle : les programmes à mémoire partagée et les programmes à mémoire distribuée.

Dans le cas d'une parallélisation avec mémoire partagée (SMP pour *shared memory parallelism*), les processeurs voient tous la même image mémoire, c'est-à-dire que la mémoire peut être adressée globalement et tous les processeurs y ont accès. Sur une machine SMP, les processeurs communiquent de façon implicite; chacun des processeurs peut lire et écrire en mémoire et les autres processeurs peuvent y accéder et les utiliser. Le défi ici est la cohérence des données puisqu'il faut veiller à ce que les données ne soient modifiées que par un seul processus à la fois.

**Figure 1: Architecture à mémoire partagée**

Pour sa part, la parallélisation avec mémoire distribuée s'apparente à une grappe, un ensemble d'ordinateurs reliés par un réseau de communication dédié. Dans ce modèle, les processus possèdent chacun leur propre mémoire et ils peuvent être exécutés sur plusieurs ordinateurs distincts. Les processus communiquent par messages : un processus utilise une fonction pour envoyer un message et l'autre processus utilise une autre fonction pour recevoir le message. Le principal défi ici est d'avoir le moins de communications possible. Même les réseaux avec les connexions physiques les plus rapides transmettent les données beaucoup plus lentement qu'un simple ordinateur : l'accès mémoire se mesure habituellement en centièmes de nanosecondes alors que les réseaux y accèdent généralement en microsecondes.

**Figure 2: Architecture à grappes, mémoire distribuée**

Nous discuterons ici uniquement de la programmation avec mémoire distribuée sur grappe, avec MPI.


# Qu'est-ce que MPI?

MPI (*message passing interface*) est en réalité une norme avec des sous-routines, fonctions, objets et autres éléments pour développer des programmes parallèles dans un environnement à mémoire distribuée. MPI est implémentée dans plusieurs bibliothèques, notamment Open MPI, Intel MPI, MPICH et MVAPICH. La norme décrit comment MPI est appelé par Fortran, C et C++, mais il existe aussi des interfaces pour plusieurs autres langages (Boost.MPI, mpi4py, Rmpi, etc.). La version MPI 3.0 ne prend plus en charge les interfaces C++, mais vous pouvez utiliser les interfaces C de C++ ou Boost MPI. Les exemples avec Python utilisent le MPI du paquet Python MPI4py.

Puisque MPI est une norme ouverte sans droits exclusifs, un programme MPI peut facilement être porté sur plusieurs ordinateurs différents. Les programmes MPI peuvent être exécutés concurremment sur plusieurs cœurs à la fois et offrent une parallélisation efficace, permettant une bonne scalabilité. Puisque chaque processus possède sa propre plage mémoire, certaines opérations de débogage s'en trouvent simplifiées; en ayant des plages mémoire distinctes, les processus n’auront aucun conflit d’accès à la mémoire comme c'est le cas en mémoire partagée. Aussi, en présence d'une erreur de segmentation, le fichier *core* résultant peut être traité par des outils standards de débogage série. Le besoin de gérer la communication et la synchronisation de façon explicite donne par contre l'impression qu'un programme MPI est plus complexe qu'un autre programme où la gestion de la communication serait implicite. Il est cependant recommandé de restreindre les communications entre processus pour favoriser la vitesse de calcul d'un programme MPI.

Nous verrons plus loin quelques-uns de ces points et proposerons des stratégies de solution; les références mentionnées au bas de cette page sont aussi à consulter.


# Principes de base

Dans ce tutoriel, nous présenterons le développement d'un code MPI en C, C++, Fortran et Python, mais les différents principes de communication s'appliquent à tout langage qui possède une interface avec MPI. Notre but ici est de paralléliser le programme simple "Hello World" utilisé dans les exemples.

**C**

```c
#include <stdio.h>
int main() {
  printf("Hello, world!\n");
  return (0);
}
```

**C++**

```cpp
#include <iostream>
using namespace std;
int main() {
  cout << "Hello, world!" << endl;
  return 0;
}
```

**Fortran**

```fortran
program hello
  print *, 'Hello, world!'
end program hello
```

**Python**

```python
print('Hello, world!')
```

Pour compiler et exécuter le programme :

```bash
[~]$ vi hello.c
[~]$ cc -Wall hello.c -o hello
[~]$ ./hello
Hello, world!
```


## Modèle SPMD

La parallélisation MPI utilise le modèle d'exécution SPMD (*single program multiple data*), où plusieurs instances s'exécutent en même temps. Chacune des instances est un processus auquel est assigné un numéro unique qui représente son rang; l'instance peut obtenir son rang lorsqu'elle est lancée. Afin d'attribuer un comportement différent à chaque instance, on utilisera habituellement un énoncé conditionnel `if`.

**Figure 3: Contrôle de comportements divergents**


## Cadre d'exécution

Un programme MPI doit comprendre le fichier d'en-tête approprié ou utiliser le modèle approprié (`mpi.h` pour C/C++, `mpif.h`, `use mpi`, ou `use mpi_f08` pour Fortran, sachant que `mpif.h` est fortement déconseillé et `mpi_f08` recommandé pour Fortran 2008). Il peut donc être compilé puis relié à l'implémentation MPI de votre choix. Dans la plupart des cas, l'implémentation possède un script pratique qui enveloppe l'appel au compilateur (*compiler wrapper*) et qui configure adéquatement `include` et `lib`, entre autres pour relier les indicateurs. Nos exemples utilisent les scripts de compilation suivants :

* pour le C, `mpicc`
* pour le Fortran, `mpifort` (recommandé) ou `mpif90`
* pour le C++, `mpiCC` ou `mpicxx`

Une fois les instances lancées, elles doivent se coordonner, ce qui se fait en tout premier lieu par l'appel d'une fonction d'initialisation :

**C**

```c
int MPI_Init(int *argc, char **argv[]);
```

**Boost (C++)**

```cpp
boost::mpi::environment(int &, char **&, bool = true);
```

**Fortran**

```fortran
MPI_INIT(IERR)
INTEGER :: IERR
```

**Fortran 2008**

```fortran
MPI_Init(ierr)
INTEGER, OPTIONAL, INTENT(OUT) :: ierr
```

**Python (mpi4py)**

```python
# importing automatically initializes MPI with mpi4py
MPI.Init()
```

En C, les arguments de `MPI_Init` pointent vers les variables `argc` et `argv` qui sont les arguments en ligne de commande. Comme pour toutes les fonctions MPI en C, la valeur retournée représente l'erreur de la fonction. En Fortran, les routines MPI retournent l'erreur dans l'argument `IERR`, ce qui est optionnel avec `use mpi_f08`.

On doit aussi appeler la fonction `MPI_Finalize` pour faire un nettoyage avant la fin du programme, le cas échéant :

**C**

```c
int MPI_Finalize(void);
```

**Boost (C++)**

```cpp
Nothing needed
```

**Fortran**

```fortran
MPI_FINALIZE(IERR)
INTEGER :: IERR
```

**Fortran 2008**

```fortran
MPI_Finalize(ierr)
INTEGER, OPTIONAL, INTENT(OUT) :: ierr
```

**Python (mpi4py)**

```python
# mpi4py installs a termination hook so there is no need to explicitly call MPI.Finalize.
MPI.Finalize()
```

Règle générale, il est recommandé d'appeler `MPI_Init` au tout début du programme et `MPI_Finalize` à la toute fin.

**C**

```c
#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  printf("Hello, world!\n");
  MPI_Finalize();
  return (0);
}
```

**Boost (C++)**

```cpp
#include <iostream>
#include <boost/mpi.hpp>
using namespace std;
using namespace boost;
int main(int argc, char *argv[]) {
  mpi::environment env(argc, argv);
  cout << "Hello, world!" << endl;
  return 0;
}
```

**Fortran**

```fortran
program phello0
  use mpi
  implicit none
  integer :: ierror
  call MPI_INIT(ierror)
  print *, 'Hello, world!'
  call MPI_FINALIZE(ierror)
end program phello0
```

**Fortran 2008**

```fortran
program phello0
  use mpi_f08
  implicit none
  call MPI_Init()
  print *, 'Hello, world!'
  call MPI_Finalize()
end program phello0
```

**Python (mpi4py)**

```python
from mpi4py import MPI
print('Hello, world!')
```


## Fonctions `rank` et `size`

Le programme pourrait être exécuté tel quel, mais le résultat ne serait pas très convaincant puisque chacun des processus produirait le même message. Nous allons plutôt faire en sorte que chaque processus fasse afficher la valeur de son rang et le nombre total de processus en opération.

**C**

```c
int MPI_Comm_size(MPI_Comm comm, int *nproc);
int MPI_Comm_rank(MPI_Comm comm, int *myrank);
```

**Boost (C++)**

```cpp
int mpi::communicator::size();
int mpi::communicator::rank();
```

**Fortran**

```fortran
MPI_COMM_SIZE(COMM, NPROC, IERR)
INTEGER :: COMM, NPROC, IERR
MPI_COMM_RANK(COMM, RANK, IERR)
INTEGER :: COMM, RANK, IERR
```

**Fortran 2008**

```fortran
MPI_Comm_size(comm, size, ierr)
TYPE(MPI_Comm), INTENT(IN) :: comm
INTEGER, INTENT(OUT) :: size
INTEGER, OPTIONAL, INTENT(OUT) :: ierr
MPI_Comm_rank(comm, rank, ierr)
TYPE(MPI_Comm), INTENT(IN) :: comm
INTEGER, INTENT(OUT) :: rank
INTEGER, OPTIONAL, INTENT(OUT) :: ierr
```

**Python (mpi4py)**

```python
MPI.Intracomm.Get_rank(self)
MPI.Intracomm.Get_size(self)
```

Le paramètre  de sortie `nproc` est donné à la fonction `MPI_Comm_size` afin d'obtenir le nombre de processus en opération. De même, le paramètre de sortie `myrank` est donné à la fonction `MPI_Comm_rank` afin d'obtenir la valeur du rang du processus actuel. Le rang du premier processus a la valeur de 0 au lieu de 1; pour N processus, les valeurs de rang vont donc de 0 à (N-1) inclusivement. L'argument `comm` est un communicateur, soit un ensemble de processus pouvant s'envoyer entre eux des messages. Dans nos exemples, nous utilisons la valeur de `MPI_COMM_WORLD`, soit un communicateur prédéfini par MPI et qui représente l'ensemble des processus lancés par la tâche. Nous n'abordons pas ici le sujet des communicateurs créés par programmation; voyez plutôt la liste des autres sujets en bas de page.

Utilisons maintenant ces fonctions pour que chaque processus produise le résultat voulu. Notez que, puisque les processus effectuent tous le même appel de fonction, il n'est pas nécessaire d'introduire des énoncés conditionnels.

**C**

```c
#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[]) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello, world! from process %d of %d\n", rank, size);
  MPI_Finalize();
  return (0);
}
```

**(Suite de la documentation dans le prochain message en raison de la limite de longueur.)**
