# Options de stockage infonuagique

Nos nuages offrent les types de stockage suivants :

## Volume

: unité de stockage infonuagique standard qui peut être attachée et détachée d’une instance.

## Disque de stockage éphémère

: disque local virtuel associé au cycle de vie d’une instance sur un disque local d'un hyperviseur (un disque local de gabarit C pourrait être perdu).

## Stockage objet

: stockage non hiérarchique pour les données créées ou téléversées sous forme de fichier complet.

## Système de fichiers partagé

: espace de stockage privé connecté au réseau (similaire aux partages NFS/SMB); doit être configuré sur chaque instance où il est monté.


## Comparaison des types de stockage

| Feature                      | Volume           | Disque de stockage éphémère | Stockage objet | Système de fichiers partagé |
|-------------------------------|-------------------|-----------------------------|-----------------|-----------------------------|
| Stockage par défaut           | oui               | oui                          | non              | non                          |
| Accès possible via un navigateur Web | non               | non                          | oui              | non                          |
| Restriction possible de l'accès pour un groupe d'IP sources | s.o.              | s.o.                         | oui (S3 ACL)     | s.o.                         |
| Peut être monté sur une instance | oui               | oui                          | non              | oui                          |
| Peut être monté simultanément sur plusieurs instances dans plusieurs projets | non               | non                          | non              | oui                          |
| Sauvegarde automatique         | non (manuellement pour les instantanés) | non                          | non              | oui (sur ruban tous les soirs) |
| Convient pour l’accès en écriture unique, en lecture seule et pour l’accès par le public | non               | non                          | oui              | non                          |
| Convient pour les données et/ou fichiers qui sont fréquemment modifiés | oui               | oui                          | non              | oui                          |
| Système de fichiers hiérarchique | oui               | oui                          | non              | oui                          |
| Convient au stockage à long terme | oui               | non                          | oui              | oui                          |
| Convient au stickage dédié pour serveurs individuels | oui               | uniquement pour données temporaires | non              | non                          |
| Supprimé automatiquement quand l'instance est supprimée | non               | oui                          | non              | non                          |
| Unité de mesure de l’espace alloué | Go                | Go                           | To               | To                           |
| Tolérance des pannes de plusieurs disques | oui               | non pour les gabarits C; oui pour les gabarits P | oui              | oui                          |
| Chiffrement physique des disques | non               | non                          | non              | non                          |


**(s.o. = sans objet)**

Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Cloud_storage_options/fr&oldid=157690"
