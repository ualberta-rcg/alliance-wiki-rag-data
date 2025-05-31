# Scalabilité

En programmation parallèle, la scalabilité est la capacité d'un programme à utiliser des ressources de calcul additionnelles, soit des cœurs CPU. Doubler le nombre de cœurs ne réduit pas forcément de moitié la durée d'une opération de calcul. Le gain de performance dépend de la nature du problème, de l'algorithme, du matériel (mémoire et réseau), et du nombre de cœurs utilisés.  Avant d'utiliser un programme parallèle sur une grappe, il est recommandé de faire une analyse de scalabilité en faisant varier le nombre de cœurs (ex: 2, 4, 8, 16, 32, 64 cœurs) pour déterminer le temps d'exécution dans chaque cas.

Deux raisons principales expliquent pourquoi la scalabilité n'est pas toujours optimale :

1.  **Opérations non parallélisables:** Certaines opérations sont exécutées en série, fixant un seuil à l'efficacité de la parallélisation.  Si 10% d'une opération d'une heure est non parallélisable, la durée d'exécution ne pourra jamais descendre en dessous de 6 minutes, quel que soit le nombre de cœurs.  On peut espérer que ce pourcentage diminue avec la taille du problème.

2.  **Communication et synchronisation:** La parallélisation nécessite de la communication et de la synchronisation entre les processus, ce qui augmente de façon non linéaire avec le nombre de cœurs :  `T<sub>c</sub>∝n<sup>α</sup>` où `α > 1`. Si le temps d'exécution de la partie scientifique est `T<sub>s</sub> = A + B/n`, le temps d'exécution total est `T = T<sub>s</sub> + T<sub>c</sub> = A + B/n + Cn<sup>α</sup>` (A, B, C étant des nombres réels positifs).  Lorsque `n→∞`, le coût indirect de la parallélisation domine.  Si A et B sont beaucoup plus grands que C, le temps d'exécution suit une courbe avec un minimum, puis une augmentation avec l'ajout de processus (trop de cuisiniers gâtent la sauce).  Il est crucial d'identifier le nombre optimal de cœurs.

Pour l'analyse de scalabilité, choisissez un problème assez petit pour des tests rapides, mais représentatif des cas réels (30 à 60 minutes avec 1 ou 2 cœurs est un bon choix).  Dans certains contextes où l'on veut réduire la scalabilité (voir plus loin), le test doit se faire avec un problème facilement extensible.

Les problèmes où C (le problème lui-même) est nul sont facilement parallélisables (*embarrassingly parallel*).  Par exemple, l'analyse de 500 fichiers indépendants ne nécessite aucune synchronisation ni communication entre les processus.

## Deux formes de scalabilité

Une meilleure scalabilité est généralement souhaitable, mais il peut être préférable de l'atténuer selon l'usage des cœurs :

*   **Scalabilité forte:** Le problème reste fixe tandis que le nombre de cœurs augmente. On vise une scalabilité linéaire (doubler les cœurs divise par deux le temps d'exécution).

*   **Scalabilité faible:** La taille du problème augmente proportionnellement à l'ajout de cœurs pour maintenir une durée d'exécution stable.


### Scalabilité forte

Exemple de tests sur une même grappe avec les mêmes paramètres d'entrée :

| Nombre de cœurs | Durée d’exécution (secondes) | Efficacité (%) |
|---|---|---|
| 2 | 2765 | s.o. |
| 4 | 1244 | 111,1 |
| 8 | 786 | 87,9 |
| 16 | 451 | 76,6 |
| 32 | 244 | 70,8 |
| 64 | 197 | 44,0 |
| 128 | 238 | 18,2 |

L'efficacité est le rapport de la durée d'exécution avec 2 cœurs et *n* cœurs, divisé par *n*/2, puis multiplié par 100.  Une efficacité de 100% correspond à une scalabilité linéaire.  Passer de 2 à 4 cœurs donne une efficacité > 100% (*superlinear scaling*), un cas rare dû à une meilleure utilisation du cache.  Avec 128 cœurs, l'efficacité est faible (18%).  Une efficacité de 75% ou plus est préférable; ici, 16 cœurs seraient recommandés.

Vous pouvez choisir le nombre et l'écart entre les points de contrôle.  Au moins 5 ou 6 valeurs sont recommandées.  Si le programme ralentit avec l'ajout de cœurs, il est inutile de poursuivre l'analyse.


### Scalabilité faible

Pour la scalabilité faible, la taille du problème augmente proportionnellement à l'augmentation du nombre de cœurs pour maintenir une durée d'exécution stable.

| Nombre de cœurs | Taille du problème | Durée d’exécution (secondes) | Efficacité (%) |
|---|---|---|---|
| 1 | 1000 | 3076 | - |
| 4 | 4000 | 3078 | 99,9 |
| 12 | 12,000 | 3107 | 99,0 |
| 48 | 48,000 | 3287 | 93,6 |
| 128 | 128,000 | 3966 | 77,6 |

L'efficacité est calculée en divisant la durée d'exécution de référence (1 cœur) par la durée d'exécution avec *n* cœurs, puis en convertissant en pourcentage.  L'objectif est une efficacité de 75%.  La scalabilité faible est appropriée pour les programmes à forte consommation mémoire et favorisant la communication avec des entités proches.


[1] https://en.wikipedia.org/wiki/Speedup#Super-linear_speedup

[2] https://fr.wikipedia.org/wiki/Transformation_de_Fourier_rapide

Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Scalability/fr&oldid=53142"
