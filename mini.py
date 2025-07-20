import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display, clear_output
from warnings import filterwarnings

# Define the path to your dataset
dataset_path = r'D:\mini project\brain tumor\Training'

# List the contents of the main directory
print(os.listdir(dataset_path))

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

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Print the shapes of the arrays
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Sample images from each label
k = 0
fig, ax = plt.subplots(1, 4, figsize=(20, 20))
fig.text(s='Sample Image From Each Label', size=18, fontweight='bold',
         fontname='monospace', y=0.62, x=0.4, alpha=0.8)
for i in labels:
    j = 0
    while True:
        if y_train[j] == i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k += 1
            break
        j += 1

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=101)
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Encode labels
y_train_new = [labels.index(i) for i in y_train]
y_train = np.array(y_train_new)
y_train = tf.keras.utils.to_categorical(y_train)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the model
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32,
                    callbacks=[tensorboard, checkpoint, reduce_lr])
filterwarnings('ignore')

# Plot training history
epochs = range(12)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold',
         fontname='monospace', y=1, x=0.28, alpha=0.8)

sns.despine()
ax[0].plot(epochs, train_acc, marker='o', markerfacecolor='green', color='green',
           label='Training Accuracy')
ax[0].plot(epochs, val_acc, marker='o', markerfacecolor='red', color='red',
           label='Validation Accuracy')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs, train_loss, marker='o', markerfacecolor='green', color='green',
           label='Training Loss')
ax[1].plot(epochs, val_loss, marker='o', markerfacecolor='red', color='red',
           label='Validation Loss')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training & Validation Loss')
fig.show()

# Evaluate the model
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)
print(classification_report(y_test_new, pred))

# Confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y_test_new, pred), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
           cmap='Greens', alpha=0.7, linewidths=2, linecolor='black')
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', y=0.92, x=0.28, alpha=0.8)

plt.show()
