# PRED - Résumé vidéo et caractéristiques perceptuelles
Auteurs : Josik SALLAUD & Nathan ROCHER

# Contexte

L’émergence des vidéos générées par des utilisateurs (User-Generated Content) ainsi que la quantité de vidéos proposées sur différentes plateformes augmentant exponentiellement ces dernières années rendent néces saire leurs pré-visualisations sous forme de résumé vidéo pour permettre aux utilisateurs de s’y retrouver parmi une large collection de vidéos.

De nombreuses méthodes permettent de générer automatiquement ces résumés de la vidéo, la plupart d’entre elles utilisent l’apprentissage profond et sont décrites dans ce rapport. Néanmoins, aucune d’entre elles ne se sert de l’influence de la composante perceptuelle humaine qui est essentielle pour générer un résumé vidéo attractif de vidéos générées par des utilisateurs.

Nous proposons dans ce travail, une méthode de génération automatique de résumé vidéo pour répondre à cette question en étudiant différentes caractéristiques perceptuelles (la congruence visuelle inter-observateurs, la mémorabilité et l’intensité émotionnelle).

Ce travail est réalisé dans le cadre de notre dernière année d’étude d’école d’ingénieurs au sein de Polytech Nantes.

# Installation

Version de python : `3.9.13`

Installer les dépendances via `pip install -r requirements.txt`

A cloner dans la racine de ce projet :

Résumé vidéo : [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM)

Reconnaissance d'émotion : [ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork)

Mémorabilité : [ResMem](https://github.com/Brain-Bridge-Lab/resmem)

IOVC : Modèle et poids disponible dans le dossier `IOVC`



Ajouté commentaire et header en haut du fichier pour faire une courte description sur a quoi sert le fichier
et pareils sur les fonctions avec notamment les entrées et sorties


# Dossier des données inférées

Les données inférées sont disponible dans ce uncloud : https://uncloud.univ-nantes.fr/index.php/s/Kd2GfoNdSoSAsxR


# Modèles
Les poids des modèles de réseau de neurone sont disponible dans le dossier `models`.

# Liste des fichiers

Émotions : 
- `./Emotion/face_intensity_video.py` --> Permet d'inférer la reconnaissance d'émotion sur une vidéo

Memorabilité :
- `./Memorability/memorability-models.py` --> 
- `./Memorability/predict_memorability.py` --> 
- `./Memorability/test-resmem.py` --> 

IOVC:
- `./IOVC/infer_summe.py` --> Permet d'inférer le score d'IOVC par image pour le jeu de données SUMME
- `./IOVC/infer_tvsum.py` --> Permet d'inférer le score d'IOVC par image pour le jeu de données TVSum
- `./IOVC/infer_VSUMM_OpenVideo.py` --> Permet d'inférer le score d'IOVC par image pour le jeu de données VSUMM mais que les vidéos OpenVideo
- `./IOVC/infer_VSUMM_Youtube.py` --> Permet d'inférer le score d'IOVC par image pour le jeu de données VSUMM mais que les vidéos Youtube

Modèles : 
- `./test_model_lstm.py` --> Permet d'inférer une vidéo avec un de nos modèles
- `./test_model_nn_sequence.py` --> Permet d'inférer une vidéo avec un de nos modèles
- `./train_model_decision_tree.py` --> Permet d'entrainer un modèle d'arbre de décision
- `./train_model_lstm.py` --> Permet d'entrainer un modèle de réseau de neurone avec couche LSTM avec des séquences de caractéristiques
- `./train_model_nn_features.py` --> Permet d'entrainer un modèle de réseau de neurone avec seulement les caractéristiques d'une image
- `./train_model_nn_sequence.py` --> Permet d'entrainer un modèle de réseau de neurone avec des séquences de caractéristiques
- `./train_model_svr.py` --> Permet d'entrainer un modèle de SVR avec seulement les caractéristiques d'une image


Autre :
- `./correlations.py` -->
- `./graphique_entrainement.py` --> Permet de générer des graphiques de résultats lors de l'entrainement d'un réseau
- `./heatmap_wasserstein.py` --> Permet de générer un graphique d'une matrice de Wasserstein
- `./read_summe.py` -->
- `./read_tvsum.py` -->
- `./read_vsumm.py` -->