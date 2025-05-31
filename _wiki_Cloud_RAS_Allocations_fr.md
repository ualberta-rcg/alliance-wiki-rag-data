# Allocations RAS Infonuagiques

Votre compte vous donne accès à une petite quantité de ressources de calcul, de stockage et de ressources infonuagiques. Avec le service d'accès rapide, vous pouvez utiliser immédiatement ces ressources pour expérimenter ou pour travailler. Le service d'accès rapide convient à plusieurs groupes de recherche. Si vous avez besoin d'une plus grande quantité de ressources, vous pouvez présenter une demande au [concours pour l'allocation de ressources](link-to-competition-page). Les chercheuses principales et chercheurs principaux à qui des ressources ont été allouées par suite du concours peuvent aussi demander des ressources par le service d'accès rapide.

Les ressources infonuagiques vous permettent de créer des **instances** (aussi appelées **machines virtuelles** ou **VM** pour *virtual machine*). Il existe deux options :

* **Instances de calcul**: celles-ci ont une durée de vie limitée dans le temps et font généralement un usage constant et intensif de CPU; elles sont parfois nommées *instances batch*. Dans certains cas, les activités de production exigent plusieurs instances de calcul. Ces dernières ont une durée de vie maximale d'**un mois**; une fois la limite atteinte, elles sont désactivées et vous devez faire le nettoyage de vos instances et télécharger les données qui doivent être conservées. Il est possible d'obtenir une prolongation de la durée de vie, dépendant de la disponibilité des ressources.
* **Instances persistantes**: ces instances n'ont pas une durée de vie finie et servent entre autres pour les serveurs web ou les serveurs de bases de données. Règle générale, elles offrent un service persistant et utilisent moins de capacité CPU que les instances de calcul.

**vGPU**: Arbutus a présentement des GPU V100 du gabarit `g1-8gb-c4-22gb` qui offrent 8Go de mémoire GPU, 4 vCPUs et 22Go de mémoire. D'autres gabarits seront éventuellement disponibles et nous vous invitons à suggérer les combinaisons que vous jugez utiles. Pour plus d'information sur comment configurer une machine virtuelle pour utiliser des vGPUs, voir [Utilisation de vGPU dans le cloud](link-to-vgpu-page).


## Quantité maximale de ressources

| Attributs             | Instance de calcul | Instance persistante |
|----------------------|----------------------|-----------------------|
| Demande faite par     | Chercheuse principale ou chercheur principal | Chercheuse principale ou chercheur principal |
| vCPU (voir Gabarits d'instances) | 80                    | 25                     |
| vGPUs                 | 2<sup>[1](#note1)</sup>                     | 1                      |
| Instances             | 20<sup>[2](#note2)</sup>                    | 10                     |
| Volumes               | 2<sup>[2](#note2)</sup>                     | 10                     |
| Instantanés de volume | 2<sup>[2](#note2)</sup>                     | 10                     |
| Mémoire RAM (Go)      | 300                   | 50                     |
| Adresses IP flottantes | 2                     | 2                      |
| Stockage persistant (TB) | 10                    | 10                     |
| Stockage système de fichier partagé (TB) | 2<sup>[1](#note1)</sup>                     | 10                     |
| Stockage objet (TB)    | 2<sup>[1](#note1)</sup>                     | 10                     |
| Durée par défaut      | 1 an<sup>[3](#note3)</sup>, durée d'un mois | 1 an (renouvelable)<sup>[3](#note3)</sup> |
| Renouvellement par défaut | Avril<sup>[3](#note3)</sup>                 | Avril<sup>[3](#note3)</sup>                  |


## Demander une allocation de ressources par le service d'accès rapide

Veuillez [remplir ce formulaire](link-to-form).


## Notes

<a name="note1"></a><sup>1</sup> Vous pouvez demander une allocation de calcul et une allocation persistante pour partager un même projet. Les deux allocations se partagent le stockage qui est limité à 10TB par type de stockage. Il n'y a pas de limite au nombre de renouvellements annuels qu'une chercheuse principale ou un chercheur principal peut demander via le service d'accès rapide; toutefois, les allocations sont faites sur la base des ressources disponibles et ne sont pas garanties. Les demandes faites avant le 1er janvier se terminent en mars de l'année suivante; leur durée peut donc dépasser un an. La durée des demandes faites entre mai et décembre est de moins d'un an. Les renouvellements prennent effet en avril.

<a name="note2"></a><sup>2</sup> Ceci n'est pas une limite ferme mais plutôt un quota pour les métadonnées. Vous pouvez demander plus de ces ressources sans passer par les concours.

<a name="note3"></a><sup>3</sup> Pour correspondre à la période d'allocation des ressources d'avril à mars.

<a name="note4"></a><sup>4</sup> En date de mai 2021, uniquement sur Arbutus.


