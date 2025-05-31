# Ressources infonuagiques

## Matériel

### Nuage Arbutus

Adresse : arbutus.cloud.alliancecan.ca

| Nombre de nœuds | CPU             | Mémoire (GB) | Stockage local (éphémère) | Réseautique | GPU             | Nombre de CPU | Nombre de vCPU |
|-----------------|-----------------|---------------|-----------------------------|-------------|-----------------|---------------|---------------|
| 156             | 2 x Gold 6248   | 384           | 2 x 1.92TB SSD en RAID0     | 1 x 25GbE   | s.o.            | 6,240         | 12,480        |
| 8               | 2 x Gold 6248   | 1024          | 2 x 1.92TB SSD en RAID1     | 1 x 25GbE   | s.o.            | 320           | 6,400         |
| 26              | 2 x Gold 6248   | 384           | 2 x 1.6TB SSD en RAID0      | 1 x 25GbE   | 4 x V100 32GB  | 1,040         | 2,080         |
| 32              | 2 x Gold 6130   | 256           | 6 x 900GB 10k SAS en RAID10 | 1 x 10GbE   | s.o.            | 1,024         | 2,048         |
| 4               | 2 x Gold 6130   | 768           | 6 x 900GB 10k SAS en RAID10 | 2 x 10GbE   | s.o.            | 128           | 2,560         |
| 8               | 2 x Gold 6130   | 256           | 4 x 1.92TB SSD en RAID5     | 1 x 10GbE   | s.o.            | 256           | 512           |
| 240             | 2 x E5-2680 v4  | 256           | 4 x 900GB 10k SAS en RAID5   | 1 x 10GbE   | s.o.            | 6,720         | 13,440        |
| 8               | 2 x E5-2680 v4  | 512           | 4 x 900GB 10k SAS en RAID5   | 2 x 10GbE   | s.o.            | 224           | 4,480         |
| 2               | 2 x E5-2680 v4  | 128           | 4 x 900GB 10k SAS en RAID5   | 1 x 10GbE   | 2 x Tesla K80  | 56            | 112           |


Emplacement : Université de Victoria

Nombre total de CPU : 16,008 (484 nœuds)
Nombre total de vCPU : 44,112
Nombre total de GPU : 108 (28 nœuds)
Mémoire vive : 157,184 GB
5.3 PB de stockage Ceph
12 PB de stockage objet Ceph sur des systèmes de fichiers partagés


### Nuage Cedar

Adresse : cedar.cloud.alliancecan.ca

| Nombre de nœuds | CPU             | Mémoire (GB) | Stockage local (éphémère) | Réseautique | GPU             | Nombre de CPU | Nombre de vCPU |
|-----------------|-----------------|---------------|-----------------------------|-------------|-----------------|---------------|---------------|
| 28              | 2 x E5-2683 v4  | 256           | 2 x 480GB SSD en RAID1      | 1 x 10GbE   | s.o.            | 896           | 1,792         |
| 4               | 2 x E5-2683 v4  | 256           | 2 x 480GB SSD en RAID1      | 1 x 10GbE   | s.o.            | 128           | 256           |


Emplacement : Université Simon-Fraser

Nombre total de CPU : 1,024
Nombre total de vCPU : 2,048
Mémoire vive : 7,680 GB
500 TB de stockage Ceph persistant


### Nuage Graham

Adresse : graham.cloud.alliancecan.ca

| Nombre de nœuds | CPU                  | Mémoire (GB) | Stockage local (éphémère) | Réseautique | GPU             | Nombre de CPU | Nombre de vCPU |
|-----------------|-----------------------|---------------|-----------------------------|-------------|-----------------|---------------|---------------|
| 6               | 2 x E5-2683 v4        | 256           | 2x 500GB SSD en RAID0       | 1 x 10GbE   | s.o.            | 192           |               |
| 2               | 2 x E5-2683 v4        | 512           | 2x 500GB SSD en RAID0       | 1 x 10GbE   | s.o.            | 64            |               |
| 8               | 2 x E5-2637 v4        | 128           | 2x 500GB SSD en RAID0       | 1 x 10GbE   | s.o.            | 256           |               |
| 8               | 2 x Xeon(R) Gold 6130 CPU | 256           | 2x 500GB SSD en RAID0       | 1 x 10GbE   | s.o.            | 256           |               |
| 3               | 2 x E5-2640 v4        | 256           | 2x 500GB SSD en RAID0       | 1 x 10GbE   | s.o.            | 120           |               |
| 12              | 2 x Xeon(R) Gold 6248 CPU | 768           | 2x 1TB SSD en RAID0        | 1 x 10GbE   | s.o.            | 480           |               |


Emplacement : Université de Waterloo

Nombre total de CPU : 1,368
Nombre total de vCPU :  
Mémoire vive : 15,616 GB
84 TB de stockage Ceph persistant


### Nuage Béluga

Adresse : beluga.cloud.alliancecan.ca

| Nombre de nœuds | CPU                     | Mémoire (GB) | Stockage local (éphémère)                    | Réseautique | GPU             | Nombre de CPU | Nombre de vCPU |
|-----------------|--------------------------|---------------|-------------------------------------------------|-------------|-----------------|---------------|---------------|
| 96              | 2 x Intel Xeon Gold 5218 | 256           | s.o., stockage Ceph éphémère                 | 1 x 25GbE   | s.o.            | 3,072         | 6,144         |
| 16              | 2 x Intel Xeon Gold 5218 | 768           | s.o., stockage Ceph éphémère                 | 1 x 25GbE   | s.o.            | 512           | 1,024         |


Emplacement : École de Technologie Supérieure

Nombre total de CPU : 3,584
Nombre total de vCPU : 7,168
Mémoire vive : 36,864 GiB
200 TiB de stockage répliqué SSD Ceph persistant
1.7 PiB de stockage persistant HDD Ceph HDD avec code d'effacement


## Plateforme logicielle

En date de 2021-03-11, les versions de la plateforme OpenStack sont :

* nuage Arbutus : Ussuri
* nuage Cedar : Train
* nuage Graham : Ussuri
* nuage Béluga : Victoria

Consultez [OpenStack Releases](LINK_TO_OPENSTACK_RELEASES).


## Images

Sur les nuages de l'Alliance, nous fournissons des images pour les distributions Alma, Debian, Fedora, Rocky et Ubuntu de Linux. D'autres images pour ces distributions seront ajoutées quand de nouvelles versions ou des mises à jour deviendront disponibles. Puisqu'il n'y a plus de support ni de mise à jour quand une version atteint sa fin de vie (EOL pour *end of life*), nous vous recommandons de migrer vos systèmes et plateformes à une version plus récente pour que vous puissiez recevoir les rustines et les avis de sécurité. Les images pour les distributions qui ont atteint leur fin de vie seront supprimées; même si vous ne devriez pas le faire, rien ne vous empêche d'exécuter une machine virtuelle avec une ancienne distribution Linux. Cependant, les images ne seront pas disponibles pour créer de nouvelles machines virtuelles.

Pour plus d'information, voir [Travailler avec des images](LINK_TO_WORKING_WITH_IMAGES).
