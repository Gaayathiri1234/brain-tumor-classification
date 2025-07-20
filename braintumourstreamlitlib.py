import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Define the path to your dataset
dataset_path = r'D:\mini project\brain tumor\Training'

# Define the labels
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Initialize lists to store image data and labels
X_train = []
y_train = []

# Define image size
image_size = 150

# Loop through each label (class)
for label in labels:
    folderPath = os.path.join(dataset_path, label)
    
    # Loop through each image in the class folder
    for image_name in tqdm(os.listdir(folderPath)):
        image_path = os.path.join(folderPath, image_name)
        
        # Load the image
        img = cv2.imread(image_path)
        
        # Resize the image
        img = cv2.resize(img, (image_size, image_size))
        
        # Append the image and label to the lists
        X_train.append(img)
        y_train.append(label)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Convert labels to numerical format
label_dict = {label: idx for idx, label in enumerate(labels)}
y_train = np.array([label_dict[label] for label in y_train])

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the model
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32)

# Evaluate the model
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = y_test  # Already in numerical format
st.text(classification_report(y_test_new, pred))

# Display confusion matrix
st.text(confusion_matrix(y_test_new, pred))

# File uploader widget
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((image_size, image_size))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_array = np.array(image)
    img_array = img_array.reshape(1, 150, 150, 3)
    p = model.predict(img_array)
    p = np.argmax(p, axis=1)[0]
    st.write(f'The Model predicts class {p}')
