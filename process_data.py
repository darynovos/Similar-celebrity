import os
import logging
import pandas as pd
import numpy as np
import face_recognition

logging.basicConfig(format = '%(levelname)s:%(message)s', level=logging.INFO)

class GetEmbedings:
    def __init__(self, path, list_names, new_path):
        self.list_names= list_names
        self.path = path
        self.new_path = new_path

    def labels(self):
        dict_actors = dict()
        for id, name in enumerate(self.list_names):
            dict_actors[name] = id  
        return dict_actors
    
    def get_embedings(self):
        embedings = np.empty(128)
        target = []
        dict_actors = self.labels()

        logging.info('Get Embendings')

        #Check if less than 2 images
        for name in dict_actors.keys():
            images = os.listdir(f'{self.path}/{name}')
            if len(images) < 2:
                print(f'delete {name} from list due to absence of enough pictures')
            else:       
                for image in images:
                    face = face_recognition.load_image_file(f'./{self.path}/{name}/{image}')
                    face_box = face_recognition.face_locations(face)
                                    
                    #Check if there is only one face
                    if len(face_box) == 1:
                        face_enc = face_recognition.face_encodings(face)[0]
                        embedings = np.vstack((embedings, face_enc))
                        target.append(dict_actors[name])

                    else:
                        print(f'More than 1 face {name}')
                        continue

        return  embedings[1:], target

    def upload_to(self):
        embedings, target = self.get_embedings()
        dict_actors = self.labels()
        
        logging.info('Writing embedings and target to files')
        
        if not os.path.exists(self.new_path):
            os.makedirs(self.new_path)  

        
        pd.DataFrame(embedings).to_csv(self.new_path +'/embendings.csv', sep = ',')
        pd.Series(target).to_csv(self.new_path + '/target.csv')
        pd.Series(dict_actors).to_csv(self.new_path + '/dict_actors.csv')