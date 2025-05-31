# Prise en main

## Que voulez-vous faire ?

Si vous ne possédez pas de compte, consultez la page [demander un compte](link-to-request-account).

Pour l'authentification multifacteur, consultez la page [Foire aux questions sur le portail CCDB](link-to-faq).

Si vous avez de l'expérience en CHP et que vous voulez vous connecter à une grappe, vous voudrez savoir :

*   quels sont les [systèmes disponibles](#quels-sont-les-systèmes-disponibles) ;
*   quels sont les [logiciels disponibles](link-to-software-availability) et comment [utiliser les modules](link-to-module-usage) ;
*   comment [soumettre une tâche](link-to-submit-job) ;
*   comment les [systèmes de fichiers](link-to-filesystem-organization) sont organisés.

Pour vous initier au CHP :

*   [Apprenez comment vous connecter par SSH](link-to-ssh-connection) à nos grappes de CHP ;
*   [Lisez cette introduction à Linux](link-to-linux-intro) ;
*   [Voyez comment transférer des données](link-to-data-transfer) soit vers nos systèmes, soit en provenance de ceux-ci.

Pour connaître les ressources qui sont disponibles pour une discipline particulière, consultez les guides spécialisés :

*   Intelligence artificielle et apprentissage machine
*   Bioinformatique
*   Simulation biomoléculaire
*   Chimie computationnelle
*   Mécanique des fluides numérique
*   Systèmes d'information géographique
*   Visualisation

Si vous avez des centaines de gigaoctets de données à transférer entre les serveurs, renseignez-vous sur le service de transfert [Globus](link-to-globus).

[Apprenez à installer des modules Python](link-to-python-installation) dans un environnement virtuel en lisant la page Python, sections « Créer et utiliser un environnement virtuel » et suivantes.

[Apprenez à installer des paquets R](link-to-r-installation).

Pour utiliser des logiciels qui ne sont pas conçus pour fonctionner sur nos systèmes traditionnels de CHP, vous pourriez utiliser l' [environnement infonuagique](link-to-cloud-environment).

Pour toute autre question, vous pouvez utiliser le champ de recherche dans le coin supérieur droit de la présente page, consulter [notre documentation technique](link-to-technical-documentation) ou encore [nous joindre](link-to-contact) par courriel.


## Quels sont les systèmes disponibles ?

Six systèmes ont été déployés de 2016 à 2018 : Arbutus, Béluga, Narval, Cedar, Graham et Niagara.

Quatre d'entre eux sont remplacés en 2015; pour plus d'informations, consultez la page [Renouvellement de l'infrastructure](link-to-infrastructure-renewal).

Arbutus est un nuage pour configurer et exécuter des instances virtuelles. Pour savoir comment y accéder, consultez la page [Service infonuagique](link-to-cloud-service).

Béluga, Cedar, Narval et Graham sont des [grappes d'usage général](link-to-general-purpose-clusters) comportant divers types de nœuds dont certains à large mémoire et d'autres avec accélérateurs comme des GPU. Pour vous y connecter, utilisez SSH. Un répertoire personnel (/home) est automatiquement créé quand vous vous connectez pour la première fois.

Niagara est une grappe conçue pour les [tâches parallèles intensives](link-to-intensive-parallel-tasks) (plus de 1000 cœurs). Pour savoir comment y accéder, consultez la page [services disponibles](link-to-available-services).

Votre [mot de passe](link-to-password-info) pour vous connecter aux nouvelles grappes est celui que vous utilisez pour vous connecter à CCDB. Votre nom d'utilisateur est affiché au haut de votre page d'accueil CCDB.


## Quelles sont les activités de formation ?

La plupart des ateliers sont organisés par nos partenaires régionaux; ils sont offerts en ligne ou en personne et pour tous les niveaux d'expertise.

*   **WestDRI** (Colombie-Britannique et provinces des Prairies) : [site web](link-to-westdri-website)  [Training Materials](link-to-westdri-training), cliquez sur l'image pour [Upcoming sessions](link-to-westdri-sessions) ou explorez le menu de navigation dans le haut de la page.
*   **UAlberta ARC Bootcamp**
*   **SHARCNET** (Ontario) : [Calendar](link-to-sharcnet-calendar), [YouTube](link-to-sharcnet-youtube), [Online Workshops](link-to-sharcnet-workshops)
*   **SciNet** (Ontario) : [Education Site](link-to-scinet-education), [YouTube](link-to-scinet-youtube)
*   **Calcul Québec** (Québec) : [Événements](link-to-calculquebec-events), [Formation](link-to-calculquebec-training)
*   **ACENET** (provinces de l'Atlantique) : [Training](link-to-acenet-training), [YouTube](link-to-acenet-youtube)

Les ateliers sont regroupés sur le [calendrier de formation de la Fédération](link-to-federation-training-calendar).


## Quels sont les systèmes qui répondent à mes besoins ?

Répondre à cette question n'est pas facile puisqu'ils peuvent subvenir à un large éventail de besoins. Si vous avez besoin de clarifications, n'hésitez pas à communiquer avec le [soutien technique](link-to-technical-support).

Les questions suivantes nous aideront à identifier les ressources pertinentes :

*   Quels sont les logiciels que vous voulez utiliser ?
*   Les logiciels doivent-ils être sous licence commerciale ?
*   Les logiciels peuvent-ils opérer sans l'intervention d'un utilisateur ? Peuvent-ils être contrôlés par un fichier de commandes ou faut-il passer par l'interface utilisateur ?
*   Les logiciels peuvent-ils fonctionner sous Linux ?
*   Pour une tâche type, quels sont les besoins en termes de mémoire, temps, puissance de traitement, accélérateurs, espace de stockage, bande passante sur le réseau, etc. ? (fournir une estimation)
*   À quelle fréquence ce type de tâche sera-t-il exécuté ?

Si vous ne connaissez pas les réponses à ces questions, notre équipe technique peut vous guider et vous indiquer les ressources appropriées.


**(Note:  All bracketed `link-to-XXX` elements need to be replaced with the actual URLs.)**
