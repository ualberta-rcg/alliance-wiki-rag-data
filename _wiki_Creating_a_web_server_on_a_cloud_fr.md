# Créer un serveur web sur un nuage

Cette page décrit une méthode simple pour créer un serveur web dans un environnement infonuagique en utilisant Ubuntu et Apache Web Server.


## Sécurité

La sécurité est un aspect crucial pour tout ordinateur accessible publiquement.  Cela inclut les connexions SSH, l'affichage de code HTML via HTTP, ou tout service utilisant un logiciel tiers (comme WordPress).  Les programmes supportant des services comme SSH ou HTTP sont appelés *daemons*.  Ces programmes fonctionnent en permanence et reçoivent des requêtes externes via des ports spécifiques.

OpenStack permet de gérer ces ports et d'en restreindre l'accès, par exemple en autorisant uniquement certaines adresses IP ou un groupe d'adresses (voir la section Groupes de sécurité).  Contrôler l'accès à une instance améliore grandement sa sécurité, mais n'élimine pas tous les risques.  Si les données envoyées ne sont pas cryptées (par exemple avec des mots de passe), une personne malveillante pourrait y accéder.  Il est donc recommandé de crypter les données avec le protocole Transport Layer Security (TLS), notamment pour les sites web accessibles publiquement (WordPress, MediaWiki, etc.).  Consultez la section Configuration du serveur Apache pour utiliser SSL.  De même, des données transmises de votre serveur web à un client peuvent être modifiées en transit par un tiers si elles ne sont pas cryptées.  Ceci n'affecte pas directement le serveur web, mais peut impacter les clients.  Dans la plupart des cas, le cryptage est recommandé.  La sécurité de vos instances est votre responsabilité.


## Installer Apache

Suivez les instructions de la page Cloud : Guide de démarrage pour créer une instance persistante sous Linux Ubuntu (voir Démarrer depuis un volume).  Pour ouvrir le port 80 et permettre l'accès HTTP à votre instance, suivez ces directives.  Sélectionnez HTTP au lieu de SSH dans le menu déroulant.

Une fois connecté à votre instance :

Mettez à jour vos répertoires apt-get avec :

```bash
sudo apt-get update
```

Mettez à jour la version d'Ubuntu avec :

```bash
sudo apt-get upgrade
```

La version la plus récente inclut les derniers correctifs de sécurité.

Installez le serveur web Apache avec :

```bash
sudo apt-get install apache2
```

![Page test Apache2](image_apache.png)  *(Cliquez pour agrandir)*

Affichez la nouvelle page Apache temporaire en entrant l'adresse IP flottante de votre instance dans la barre d'adresse de votre navigateur (la même adresse IP que celle utilisée pour la connexion SSH). Vous devriez voir une page de test similaire à celle illustrée ci-contre.

Modifiez le contenu des fichiers dans `/var/www/html` pour créer votre site web, notamment le fichier `index.html` qui définit le point d'entrée.


### Modifier le répertoire root du serveur web

Il est souvent plus facile de gérer un site web si l'utilisateur connecté à l'instance est propriétaire des fichiers. Dans l'image Ubuntu de notre exemple, le propriétaire est `ubuntu`. Les étapes suivantes indiquent à Apache d'utiliser `/home/ubuntu/public_html` (par exemple) au lieu de `/var/www/html`.

Utilisez la commande :

```bash
sudo vim /etc/apache2/apache2.conf
```

(ou un autre éditeur) pour modifier la ligne `<Directory /var/www/>` en `<Directory /home/ubuntu/public_html>`.

Utilisez la commande :

```bash
sudo vim /etc/apache2/sites-available/000-default.conf
```

pour modifier la ligne `DocumentRoot /var/www/html` en `DocumentRoot /home/ubuntu/public_html`.

Créez le répertoire dans le répertoire `/home` de l'utilisateur avec la commande :

```bash
mkdir public_html
```

Copiez la page par défaut dans le répertoire `public_html` avec la commande :

```bash
cp /var/www/html/index.html /home/ubuntu/public_html
```

Redémarrez le serveur Apache pour appliquer les modifications :

```bash
sudo service apache2 restart
```

Vous devriez pouvoir modifier le fichier `/home/ubuntu/public_html/index.html` sans utiliser `sudo`. Rafraîchissez la page dans votre navigateur pour voir les modifications.


## Limiter la bande passante

Si votre serveur web est fortement sollicité, il peut consommer beaucoup de bande passante.  Une solution est d'utiliser le module Apache `bw`.


### Installation

```bash
sudo apt install libapache2-mod-bw
sudo a2enmod bw
```


### Configuration

L'exemple suivant limite la bande passante à 100 Mbps pour tous les clients :

```apache
BandWidthModule On
ForceBandWidthModule On

#Exceptions to badwith of 100Mbps should go here above limit
#below in order to override it

#limit all connections to 100Mbps
#100Mbps *1/8(B/b)*1e6=12,500,000 bytes/s
BandWidth all 12500000
```

Ce code doit être placé entre les balises `<VirtualHost></VirtualHost>` pour votre site. La configuration Apache par défaut se trouve dans `/etc/apache2/sites-enabled/000-default.conf`.


## Pour plus d'information

* [Configuration du serveur Apache pour utiliser SSL](link_to_ssl_config)
* [Documentation Apache2](link_to_apache_doc)
* [Tutoriel HTML w3schools](link_to_w3schools)


*(Remplacez `link_to_ssl_config`, `link_to_apache_doc`, et `link_to_w3schools` par les liens appropriés)*
