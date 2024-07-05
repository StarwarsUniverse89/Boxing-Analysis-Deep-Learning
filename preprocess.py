

import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_images(data_dir, target_size=(224, 224)):
    images = []
    labels = []
    label_map = {'Boxer': 1, 'Not_Boxer': 0}
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    img = preprocess_input(img)
                    images.append(img)
                    labels.append(label_map[label])
    
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    data_dir = "/content/drive/My Drive/BoxingML"
    images, labels = preprocess_images(data_dir)
    np.save('images.npy', images)
    np.save('labels.npy', labels)

