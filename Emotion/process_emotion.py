"""
Permet d'extraire des scores de chaque image de la vidéo, les vecteurs qui nous intéresse
"""

import json
from glob import glob

face_files = glob("*-face-intensity.json")

# On boucle sur les différents fichier
for file_name in face_files:

    # On lit le fichier
    with open(file_name) as json_file:
        face_data = json.load(json_file)

        # Pour chaque vidéo
        for video_key in face_data.keys():
            video_data = face_data[video_key]

            # On analyse chaque frame
            new_video_features = []
            for frame_data in video_data:
                if frame_data is not None :
                    nbr_face = len(frame_data)

                    # Récupération des max des proba des visages
                    max_emotion_dict = { "happy": 0, "angry": 0, "disgust": 0, "neutral": 0, "fear": 0, "sad": 0, "surprise": 0}
                    max_proba = 0
                    for face in frame_data:
                        for proba_dict in face["proba_list"]:
                            key, value = list(proba_dict.items())[0]

                            # max proba par categorie
                            if max_emotion_dict[key] < value:
                                max_emotion_dict[key] = value
                            
                            # max proba toute catégorie confondues
                            if max_proba < value:
                                max_proba = value
                    
                    # On récupère les valeurs finales
                    final_features_dict = {"nbr_face": nbr_face, "max_proba": max_proba, **max_emotion_dict}

                    new_video_features.append(final_features_dict)
                else:
                    new_video_features.append(None)

            face_data[video_key] = new_video_features

        # On écrit les valeurs qui nous intéresse
        with open("PROCESSED-" + file_name, "w") as processed_file:
            processed_file.write(json.dumps(face_data))