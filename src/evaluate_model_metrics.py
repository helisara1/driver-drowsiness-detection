import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EYE_DATA = os.path.join(BASE_DIR,"data","eye")
YAWN_DATA = os.path.join(BASE_DIR,"data","yawn")

EYE_MODEL = os.path.join(BASE_DIR,"models","eye_model.h5")
YAWN_MODEL = os.path.join(BASE_DIR,"models","yawn_model.h5")

img_size = 224
batch_size = 32

eye_model = load_model(EYE_MODEL)
yawn_model = load_model(YAWN_MODEL)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

eye_val = datagen.flow_from_directory(
    EYE_DATA,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

yawn_val = datagen.flow_from_directory(
    YAWN_DATA,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Evaluating Eye Model")
eye_preds = np.argmax(eye_model.predict(eye_val),axis=1)

print(classification_report(eye_val.classes,eye_preds))
print(confusion_matrix(eye_val.classes,eye_preds))

print("\nEvaluating Yawn Model")
yawn_preds = np.argmax(yawn_model.predict(yawn_val),axis=1)

print(classification_report(yawn_val.classes,yawn_preds))
print(confusion_matrix(yawn_val.classes,yawn_preds))