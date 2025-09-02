import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

IMAGE_SIZE = (64, 64)  # resize all images

def load_images(folder):
    X, y = [], []
    classes = os.listdir(folder)
    for label, cls in enumerate(classes):
        cls_folder = os.path.join(folder, cls)
        for f in os.listdir(cls_folder):
            img = imread(os.path.join(cls_folder, f))
            img = resize(img, IMAGE_SIZE)
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y), classes

def train_model(folder):
    X, y, classes = load_images(folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=classes))
    joblib.dump((clf, classes), "image_classifier.pkl")
    return clf, classes
