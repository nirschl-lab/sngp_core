import cv2
import PIL
import numpy as np

class ArtifactSimulator():
    '''
    imgs_folder: Folder where different artifact images are present 
    csv_path: csv describing image name and arfifact category 
    '''
    
    def __init__(self, imgs_folder, csv_path):
        self.imgs_folder = imgs_folder 
        self.csv_path = csv_path

    def apply_random_artifact(image: np.array | PIL.Image.Image) -> np.array | PIL.Image.Image:
        '''
        takes image and applies random artifact
        '''
        pass

        return 

    def find_unique_artifact_category():
        pass

if __name__ == '__main__':
    pass