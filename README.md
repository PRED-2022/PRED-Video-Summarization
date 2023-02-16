# PRED - Résumé vidéo et caractéristiques perceptuelles
Auteurs : Josik SALLAUD & Nathan ROCHER

# Contexte

L’émergence des vidéos générées par des utilisateurs (User-Generated Content) ainsi que la quantité de vidéos proposées sur différentes plateformes augmentant exponentiellement ces dernières années rendent néces saire leurs pré-visualisations sous forme de résumé vidéo pour permettre aux utilisateurs de s’y retrouver parmi une large collection de vidéos.

De nombreuses méthodes permettent de générer automatiquement ces résumés de la vidéo, la plupart d’entre elles utilisent l’apprentissage profond et sont décrites dans ce rapport. Néanmoins, aucune d’entre elles ne se sert de l’influence de la composante perceptuelle humaine qui est essentielle pour générer un résumé vidéo attractif de vidéos générées par des utilisateurs.

Nous proposons dans ce travail, une méthode de génération automatique de résumé vidéo pour répondre à cette question en étudiant différentes caractéristiques perceptuelles (la congruence visuelle inter-observateurs, la mémorabilité et l’intensité émotionnelle).

Ce travail est réalisé dans le cadre de notre dernière année d’étude d’école d’ingénieurs au sein de Polytech Nantes.

# Installation

Installer les dépendances via `pip install -r requirements.txt`

Version de python : `3.9.13`

Résumé vidéo : [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM)

Reconnaissance d'émotion : [ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork)

Mémorabilité : [ResMem](https://github.com/Brain-Bridge-Lab/resmem)

IOVC : Modèle et poids disponible dans le dossier ``IOVC``



Ajouté commentaire et header en haut du fichier pour faire une courte description sur a quoi sert le fichier
et pareils sur les fonctions avec notamment les entrées et sorties


# Dossier des données inférées

Les données inférées sont dans ce uncloud : https://uncloud.univ-nantes.fr/index.php/s/Kd2GfoNdSoSAsxR


# Modèles
Les poids des modèles de réseau de neurone sont disponible dans le dossier `models`.

# Liste des fichiers
Faire une decseiption des fichiers

Faire doc avec signature et lien vers le git / uncloud
