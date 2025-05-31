# Guide de dépannage du nuage

Ce document fournit des solutions aux problèmes fréquemment rencontrés lors de l'utilisation d'une instance dans nos environnements cloud.  Certains problèmes peuvent être résolus par vous-même, tandis que d'autres nécessitent l'intervention d'un administrateur système. Dans ce cas, des conseils sont fournis sur la façon de soumettre une demande d'assistance et les informations à inclure.

## Problème de connexion à un nuage

Pour vous connecter à l'un de nos nuages, vous devez avoir demandé un nouveau projet cloud ou avoir demandé l'accès à un projet cloud existant. Sinon, vous recevrez le message "Données d'identification non valides".

Remplissez le formulaire de demande.

Dans les jours suivant votre demande, vous recevrez un courriel de confirmation contenant les informations d'accès à votre projet. Si vous n'avez pas reçu ce courriel après trois jours, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom d'utilisateur, le nom de votre établissement et l'adresse courriel utilisée pour soumettre votre demande de projet.

Le courriel de confirmation contient le nom du nuage où se trouve votre projet. Pour les adresses des nuages, consultez [Utiliser les ressources infonuagiques](link-to-resource-page).

Si vous avez reçu un courriel de confirmation et que vous ne pouvez toujours pas vous connecter, vérifiez s'il y a un incident signalé sur la page [État des ressources](link-to-status-page).

Utilisez le même nom d'utilisateur que celui associé à votre compte Alliance (le même que celui utilisé pour vous connecter à une grappe). N'utilisez pas votre adresse courriel. Pour vérifier vos identifiants, essayez de vous connecter à la base de données CCBD.

Au besoin, [réinitialisez votre mot de passe](link-to-password-reset).

Si vous ne pouvez toujours pas accéder à votre projet, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet et une description des étapes que vous avez suivies.

Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).


## Impossible de lancer une instance

Si votre instance ne peut pas être lancée, vérifiez si l'une des ressources demandées n'a pas atteint le quota prescrit. Votre projet est soumis à des quotas et limité en termes d'instances, de CPU et de gigaoctets de mémoire pouvant être utilisés. Une instance ne démarrera pas si elle dépasse l'un de ces quotas. Pour vérifier votre utilisation des ressources, connectez-vous au nuage où se trouve votre projet (voir [les liens de connexion aux nuages](link-to-cloud-logins)). Dans le menu de gauche, cliquez sur "Compute", puis sur "Vue d'ensemble". Si ces ressources sont insuffisantes pour votre projet, soumettez un [formulaire de demande pour des ressources supplémentaires](link-to-resource-request). Pour plus d'informations sur les quotas et les allocations de plus de 10 To, consultez [Ressources infonuagiques avec le service d'accès rapide](link-to-fast-access-resources).

Si vous recevez le message "N'a pas pu effectuer l'opération demandée sur l'instance "...", l'instance a un statut d'erreur: Veuillez essayer à nouveau ultérieurement [Error: No valid host was found. There are not enough hosts available.]", vérifiez le contenu du champ "Zone de disponibilité".

Lorsqu'une instance est lancée, l'option "Détails" vous demande d'entrer un nom, une description et une zone de disponibilité. Par défaut, la sélection est "Toute zone de disponibilité", ce qui laisse le choix à OpenStack. Si vous n'utilisez pas la valeur par défaut, mais que vous la saisissez manuellement, le message d'erreur peut s'afficher. Entrez la valeur par défaut.

Si vous recevez toujours le message ou que vous devez sélectionner une autre valeur pour la zone de disponibilité, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom d'utilisateur, le nom du nuage, le nom du projet, l'UUID du volume et la description des étapes que vous avez suivies.

Si vous ne parvenez toujours pas à lancer l'instance, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom d'utilisateur, le nom du nuage, le nom du projet, l'UUID du volume et les informations obtenues lors des étapes que vous avez suivies.

Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).


## Problème de connexion à une instance

Si vous ne pouvez pas vous connecter à votre instance ou à l'un de ses services, vérifiez s'il y a un incident signalé pour le nuage sur la page [État des ressources](link-to-status-page). Si c'est le cas, vous devrez attendre que la situation revienne à la normale.

Si aucun incident n'est signalé pour le nuage, essayez de vous y connecter via le tableau de bord OpenStack. Par exemple, si votre projet se trouve sur Arbutus, connectez-vous avec `https://arbutus.cloud.computecanada.ca`. Les liens vers les autres nuages se trouvent dans la section [Utiliser les ressources infonuagiques](link-to-resource-page).

Si vous êtes toujours incapable de vous connecter, testez votre connexion Internet en essayant de joindre une page comme `https://www.google.com` avec votre navigateur. Si votre connexion Internet n'est pas en cause, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du projet, le nom du nuage et une description des étapes que vous avez suivies. Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).

Si la page de connexion au nuage s'affiche mais que vous ne pouvez pas vous connecter, reportez-vous à la section [Problème de connexion à un nuage](#probleme-de-connexion-a-un-nuage) ci-dessus.

Si vous pouvez vous connecter au tableau de bord OpenStack, il existe plusieurs façons de savoir si votre instance fonctionne :

1. Dans le menu de gauche, sélectionnez "Compute", puis "Instances". La colonne "État de l'alimentation" devrait afficher "En fonctionnement". Si ce n'est pas le cas, sélectionnez "Démarrer une instance" (ou "Redémarrer une instance", selon votre interface) dans le menu déroulant "Actions".

2. Vérifiez le journal des actions en cliquant sur le nom de l'instance et en sélectionnant l'onglet "Journal des actions". Le journal affiche toutes les actions liées à l'instance ; si vous n'en reconnaissez pas une, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet et l'ID d'utilisateur.

3. Vérifiez également le contenu sous l'onglet "Console" qui pourrait afficher des messages d'erreur.

Si vous ne pouvez pas redémarrer votre instance, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet, l'ID de l'instance et une description de votre problème et des étapes que vous avez suivies. Pour connaître l'ID de l'instance, cliquez sur le nom de l'instance et sélectionnez l'onglet "Vue d'ensemble". Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).

4. Vérifiez si vous pouvez accéder à votre instance par SSH (protocole Secure Shell).

Si vous ne pouvez pas accéder à l'application ou au service web de votre instance, que vous avez vérifié les points 1 à 4 et que votre instance est en fonctionnement, essayez de vous connecter par SSH en suivant [ces directives](link-to-ssh-instructions).

Si une invite de connexion s'affiche, assurez-vous d'utiliser la bonne paire de clés et le bon nom d'utilisateur. Pour savoir quelle paire de clés utiliser, sélectionnez "Compute", puis "Instances" dans le menu OpenStack et regardez dans la colonne "Paire de clés".

Le nom d'utilisateur dépend du système d'exploitation :

| Système d'exploitation | Nom d'utilisateur |
|---|---|
| Debian | debian |
| Ubuntu | ubuntu |
| CentOS | centos |
| Fedora | fedora |

Ces noms ne s'appliquent pas si vous avez modifié le nom d'utilisateur avec un script CloudInit personnalisé. Dans ce cas, le nom d'utilisateur sera celui que vous avez entré.

Si une invite de connexion ne s'affiche pas, vérifiez vos paramètres de sécurité.

Assurez-vous que votre propre adresse IP n'a pas changé. Testez votre adresse IP en entrant `https://ipv4.icanhazip.com/` dans votre navigateur. Vos paramètres de sécurité doivent accepter votre adresse IP pour que vous puissiez vous connecter à votre instance. Si votre adresse a changé, ajoutez une nouvelle règle à votre groupe de sécurité comme expliqué ci-dessous.

Assurez-vous que votre adresse IP est débloquée pour SSH. Dans le tableau de bord OpenStack, sélectionnez "Réseau", puis "Groupes de sécurité". À la fin de la ligne pour le groupe relié à votre instance, sélectionnez "Gérer les règles" dans le menu déroulant "Actions". Si vous n'avez pas configuré un autre groupe, celui-ci sera le groupe par défaut. Il devrait y avoir une règle "Entrée" dans la colonne "Direction", la valeur TCP dans la colonne "Protocole IP", la valeur 22 (SSH) dans la colonne "Plage de ports" et `your-ip-address/32` dans la colonne "Préfixe IP distante". Si cette règle est absente, cliquez sur "+Ajouter une règle" et sélectionnez "SSH" dans le menu déroulant "Règle", entrez `your-ip-address/32` dans le champ "CIDR" et cliquez sur le bouton "Ajouter".

Si vous ne pouvez toujours pas vous connecter à l'instance, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet, l'UUID de l'instance et les informations obtenues lors des étapes que vous avez suivies. Pour trouver l'UUID de l'instance, sélectionnez "Compute", puis "Instances" et cliquez sur le nom de l'instance. Sous l'onglet "Vue d'ensemble", l'UUID est la longue chaîne de caractères (lettres, chiffres et tiret) qui se trouve sur la ligne ID.

Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).


## Impossible de supprimer un volume

Vous ne pouvez pas supprimer un volume qui est attaché à une instance en fonctionnement. Pour vérifier cela, connectez-vous au nuage où se trouve votre projet. Dans le menu de gauche du tableau de bord OpenStack, sélectionnez "Volumes", puis "Volumes". La colonne "Attaché à" est vide si le volume n'est attaché à aucune instance, sinon, vous devez détacher le volume de l'instance avant de pouvoir le supprimer (voir [Détacher un volume](link-to-detach-volume)).

Une fois que le volume est détaché, vérifiez son statut. Dans le menu de gauche, sélectionnez "Volumes", puis "Volumes". Si la colonne "Statut" affiche "disponible(s)", passez à l'étape 3. Sinon, si la colonne affiche "En cours d'utilisation", écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet, l'UUID de l'instance et les informations obtenues lors des étapes que vous avez suivies.

Avant de supprimer un volume, vous devez supprimer les instantanés attachés à ce volume, le cas échéant. Pour vérifier si le volume a des instantanés, sélectionnez "Volumes", puis "Instantanés". La colonne "Nom du volume" affiche le volume auquel l'instantané est attaché, le cas échéant. Dans le menu déroulant "Actions", sélectionnez "Supprimer l'instantané de volume".

Si vous ne pouvez toujours pas supprimer le volume, écrivez à nuage@tech.alliancecan.ca en indiquant votre nom, votre nom d'utilisateur, le nom du nuage, le nom du projet, l'UUID du volume et les informations obtenues lors des étapes que vous avez suivies.

Pour plus d'informations sur l'obtention d'aide, consultez la page [Soutien technique](link-to-support-page).


**(Remember to replace the bracketed `link-to-...` placeholders with actual links.)**
