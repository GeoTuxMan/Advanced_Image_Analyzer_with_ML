#dupa ce am rulat training.py, si am salvat modelul .pkl
# putem face predictii, pentru o noua inagine
import joblib
from skimage.io import imread
from skimage.transform import resize
import numpy as np

IMAGE_SIZE = (64, 64)

def predict_image(model_path, img_path):
    clf, classes = joblib.load(model_path)
    img = imread(img_path)
    img = resize(img, IMAGE_SIZE)
    img_flat = img.flatten().reshape(1, -1)
    pred = clf.predict(img_flat)[0]
    return classes[pred]

if __name__ == "__main__":
    label = predict_image("image_classifier.pkl", "cat.jpg")
    print("Predicted class:", label) # output: cats
