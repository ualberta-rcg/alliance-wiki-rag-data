# Amélioration de la sécurité SSH

**SSH** est le protocole de connexion aux grappes de Calcul Canada. Pour protéger les communications, SSH vérifie l'identité du serveur et de l'utilisateur en les comparant aux identités connues, et effectue le chiffrement de la connexion. La sécurité étant périodiquement menacée par de nouveaux risques, Calcul Canada abandonne à l'été 2019 certaines options de SSH qui ne sont plus jugées sécuritaires. Les utilisateurs devront effectuer les modifications décrites dans l'organigramme.

## Modifications apportées en septembre et octobre 2019

Des courriels importants expliquant ces modifications ont été envoyés aux utilisateurs les 29 juillet et 16 septembre 2019.

La puissance de traitement de plus en plus forte fait en sorte que certains algorithmes et protocoles de chiffrement qui étaient suffisamment efficaces il y a 10 ou 15 ans présentent aujourd’hui un risque d’intrusion par des tierces parties. Calcul Canada modifie ses politiques et pratiques en rapport avec **SSH**, l’outil principal utilisé pour offrir des connexions sécurisées à ses grappes. Tous les utilisateurs devront mettre à jour la copie locale de la clé hôte qui identifie chacune des grappes. De plus, dans certains cas, il sera nécessaire de mettre à jour le logiciel du client SSH et/ou de générer une nouvelle paire de clés.

### Quelles sont les modifications?

Les modifications suivantes ont été apportées le 24 septembre 2019 pour Graham et une semaine plus tard pour Béluga et Cedar :

* Désactivation de certains algorithmes de chiffrement.
* Désactivation de certains types de clés publiques.
* Régénération des clés hôtes.

Même si certains de ces termes vous sont inconnus, les directives suivantes vous aideront à vous préparer adéquatement. Si les tests proposés ci-dessous indiquent que vous devez changer ou mettre à jour votre client SSH, [cette autre page](link_to_other_page) pourrait vous être utile.

Les utilisateurs d'Arbutus ne sont pas touchés puisque la connexion se fait par interface web et non via SSH.

Il est possible que quelques-uns de ces messages et erreurs aient été produits par suite de [mises à jour pour Niagara](link_to_niagara_updates) effectuées le 31 mai 2019 et pour Graham au début d'août.

### Mise à jour de la liste des hôtes de votre client

Quand les modifications seront complétées, un avertissement semblable au suivant sera probablement affiché la première fois que vous vous connecterez à une grappe.

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.
Please contact your system administrator.
Add correct host key in /home/username/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /home/username/.ssh/known_hosts:109
ED25519 host key for graham.computecanada.ca has changed and you have requested strict checking.
Host key verification failed.
Killed by signal 1.
```

Cet avertissement indique que les clés hôtes de la grappe (ici la grappe Graham) ont été modifiées et que le logiciel de votre client SSH se souvient des clés antérieures. Ceci se produit pour contrer les [attaques de l'homme du milieu](link_to_man_in_the_middle_attack). L'avertissement sera affiché sur chaque ordinateur à partir duquel vous vous connectez.

Vous pourriez aussi recevoir un avertissement de mystification ([DNS spoofing](link_to_dns_spoofing)) dû également à une telle modification.

#### MobaXterm, PuTTY, WinSCP

Sous Windows, avec un client MobaXterm, PuTTY ou WinSCP, l'avertissement sera affiché dans une fenêtre et vous serez invité à accepter la nouvelle clé en cliquant sur `Yes`. Avant de cliquer, assurez-vous que l'empreinte se trouve dans la liste des [empreintes de clés hôtes ci-dessous](#empreintes-de-clés-hôtes). Si elle ne s'y trouve pas, contactez le [soutien technique](link_to_support).

#### macOS, Linux, GitBash, Cygwin

Si vous utilisez `ssh` en ligne de commande, une de ces commandes fera en sorte que votre système *oublie* l'ancienne clé hôte :

**Graham**

```bash
for h in 2620:123:7002:4::{2..5} 199.241.166.{2..5} {gra-login{1..3},graham,gra-dtn,gra-dtn1,gra-platform,gra-platform1}.{sharcnet,computecanada}.ca; do ssh-keygen -R $h; done
```

**Cedar**

```bash
for h in 206.12.124.{2,6} cedar{1,5}.cedar.computecanada.ca cedar.computecanada.ca; do ssh-keygen -R $h; done
```

**Beluga**

```bash
for h in beluga{,{1..4}}.{computecanada,calculquebec}.ca 132.219.136.{1..4}; do ssh-keygen -R $h; done
```

**Mp2**

```bash
for h in ip{15..20}-mp2.{computecanada,calculquebec}.ca 204.19.23.2{15..20}; do ssh-keygen -R $h; done
```

La prochaine fois que vous vous connecterez par SSH, vous devrez confirmer les nouvelles clés, par exemple

```
$ ssh graham.computecanada.ca
The authenticity of host 'graham.computecanada.ca (142.150.188.70)' can't be established.
ED25519 key fingerprint is SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk.
ED25519 key fingerprint is MD5:bc:93:0c:64:f7:e7:cf:d9:db:81:40:be:4d:cd:12:5c.
Are you sure you want to continue connecting (yes/no)?
```

Avant d'entrer `yes`, assurez-vous que l'empreinte se trouve dans la liste des [empreintes de clés hôtes ci-dessous](#empreintes-de-clés-hôtes). Si elle ne s'y trouve pas, contactez le [soutien technique](link_to_support).

### Dépannage

#### Puis-je tester mon client SSH avant que les modifications soient apportées?

Oui. Nous avons configuré un serveur qui vous permet de tester le fonctionnement de votre logiciel avec les nouveaux protocoles. Si vous pouvez vous connecter à `ssh-test.computecanada.ca` avec les informations d'indentification pour votre compte Calcul Canada, vous pourrez vous connecter après que les modifications auront été apportées, mais vous devrez quand même mettre à jour les clés SSH enregistrées localement par votre client.

Voir la liste des [empreintes de clés hôtes ci-dessous](#empreintes-de-clés-hôtes).

#### Ma clé SSH ne fonctionne plus

Si on vous demande un mot de passe mais que vous utilisiez les clés SSH par le passé, il est probable que ce soit dû à la désactivation des clés DSA et RSA 1024 bits. Il vous faudra générer une nouvelle clé plus forte. La méthode est différente selon que vous êtes sous Windows ou Linux/macOS. Dans ce dernier cas, la référence décrit aussi comment ajouter votre nouvelle clé publique au serveur hôte à distance pour que l'authentification se fasse par la clé plutôt que par mot de passe.

#### Impossibilité de se connecter

Les messages d'erreur suivants

```
Unable to negotiate with 142.150.188.70 port 22: no matching cipher found.
Unable to negotiate with 142.150.188.70 port 22: no matching key exchange method found.
Unable to negotiate with 142.150.188.70 port 22: no matching mac found.
```

indiquent que vous devez effectuer la mise à jour de votre client SSH et utiliser une version compatible parmi celles listées ci-dessous.

#### Clients compatibles

La liste suivante n'est pas complète, mais nous avons testé la configuration avec ces clients; il est possible que les versions antérieures de ces clients ne soient pas compatibles. Nous vous recommandons de mettre à jour votre système d'exploitation et votre client SSH.

* **Linux:** OpenSSH_7.4p1, OpenSSL 1.0.2k-fips (CentOS 7.5, 7.6); OpenSSH_6.6.1p1 Ubuntu-2ubuntu2.13, OpenSSL 1.0.1f (Ubuntu 14)
* **OS X:** Pour connaître la version de votre client SSH, utilisez la commande `ssh -V`. OpenSSH 7.4p1, OpenSSL 1.0.2k (Homebrew); OpenSSH 7.9p1, LibreSSL 2.7.3 (OS X 10.14.5)
* **Windows:** MobaXterm Hoome Edition v11.1; PuTTY 0.72; Windows Services for Linux (WSL) v1 Ubuntu 18.04 (OpenSSH_7.6p1 Ubuntu-4ubuntu0.3, OpenSSL 1.0.2n); openSUSE Leap 15.1 (OpenSSH_7.9p1, OpenSSL 1.1.0i-fips)
* **iOS:** Termius, 4.3.12

## Empreintes de clés hôtes

Les commandes suivantes servent à récupérer les empreintes de clés hôtes à distance :

```bash
ssh-keyscan <hostname> | ssh-keygen -E md5 -l -f -
ssh-keyscan <hostname> | ssh-keygen -E sha256 -l -f -
```

Les empreintes pour les grappes de Calcul Canada sont listées ci-dessous. Si l'empreinte que vous recevez ne correspond à aucune de cette liste, n'acceptez pas la connexion et contactez le [soutien technique](link_to_support).

**Béluga**

* ED25519 SHA256:lwmU2AS/oQ0Z2M1a31yRAxlKPcMlQuBPFP+ji/HorHQ  MD5:2d:d7:cc:d0:85:f9:33:c1:44:80:38:e7:68:ce:38:ce
* RSA SHA256:7ccDqnMTR1W181U/bSR/Xg7dR4MSiilgzDlgvXStv0o  MD5:7f:11:29:bf:61:45:ae:7a:07:fc:01:1f:eb:8c:cc:a4

**Cedar**

* ED25519 SHA256:a4n68wLDqJhxtePn04T698+7anVavd0gdpiECLBylAU  MD5:f8:6a:45:2e:b0:3a:4b:16:0e:64:da:fd:68:74:6a:24
* RSA SHA256:91eMtc/c2vBrAKM0ID7boyFySo3vg2NEcQcC69VvCg8  MD5:01:27:45:a0:fd:34:27:9e:77:66:b0:97:55:10:0e:9b

**Graham**

* ED25519 SHA256:mf1jJ3ndpXhpo0k38xVxjH8Kjtq3o1+ZtTVbeM0xeCk  MD5:bc:93:0c:64:f7:e7:cf:d9:db:81:40:be:4d:cd:12:5c
* RSA SHA256:tB0gbgW4PV+xjNysyll6JtDi4aACmSaX4QBm6CGd3RM  MD5:21:51:ca:99:15:a8:f4:92:3b:8e:37:e5:2f:12:55:d3

**Narval**

* ED25519 SHA256:pTKCWpDC142truNtohGm10+lB8gVyrp3Daz4iR5tT1M  MD5:79:d5:b2:8b:c6:2c:b6:3b:79:d2:75:0e:3b:31:46:17
* RSA SHA256:tC0oPkkY2TeLxqYHgfIVNq376+RfBFFUZaswnUeeOnw  MD5:bc:63:b5:f9:e6:48:a3:b7:0d:4a:23:26:a6:31:19:ef

**Niagara**

* ED25519 SHA256:SauX2nL+Yso9KBo2Ca6GH/V9cSFLFXwxOECGWXZ5pxc  MD5:b4:ae:76:a5:2b:37:8d:57:06:0e:9a:de:62:00:26:be
* RSA SHA256:k6YEhYsI73M+NJIpZ8yF+wqWeuXS9avNs2s5QS/0VhU  MD5:98:e7:7a:07:89:ef:3f:d8:68:3d:47:9c:6e:a6:71:5e

**ssh-test.computecanada.ca**

* ED25519 (256b) SHA256:Tpu6li6aynYkhmB83Q9Sh7x8qdhT8Mbw4QcDxTaHgxY  MD5:33:8f:f8:57:fa:46:f9:7f:aa:73:e2:0b:b1:ce:66:38
* RSA (4096b) SHA256:DMSia4nUKIyUhO5axZ/As4I8uqlaX0jPcJvcK93D2H0  MD5:a7:08:00:7c:eb:81:f2:f7:2f:5a:92:b0:85:e3:e8:5d

**Mp2**

* ED25519 (256b) SHA256:hVAo6KoqKOEbtOaBh6H6GYHAvsStPsDEcg4LXBQUP50  MD5:44:71:28:23:9b:a1:9a:93:aa:4b:9f:af:8d:9b:07:01
* RSA (4096b) SHA256:XhbK4jWsnoNNjoBudO6zthlgTqyKkFDtxiuNY9md/aQ  MD5:88:ef:b0:37:26:75:a2:93:91:f6:15:1c:b6:a7:a9:37

**Siku**

* ED25519 (256b) SHA256:F9GcueU8cbB0PXnCG1hc4URmYYy/8JbnZTGo4xKflWU  MD5:44:2b:1d:40:31:60:1a:83:ae:1d:1a:20:eb:12:79:93
* RSA (2048b) SHA256:cpx0+k52NUJOf8ucEGP3QnycnVkUxYeqJQMp9KOIFrQ  MD5:eb:44:dc:42:70:32:f7:61:c5:db:3a:5c:39:04:0e:91


Remember to replace `link_to_other_page`, `link_to_niagara_updates`, `link_to_man_in_the_middle_attack`, `link_to_dns_spoofing`, and `link_to_support` with the actual links.
