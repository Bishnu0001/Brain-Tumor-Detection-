import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf

train_path = "C:\\Spyder\\NLP\\project_img\\archive (9)\\Training"
test_path = "C:\\Spyder\\NLP\\project_img\\archive (9)\\Testing"

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_data.class_names
print("Classes:", class_names)

# ---------------- MODEL ----------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

# 4 classes
model.add(Dense(4, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    epochs=2,
    validation_data=test_data
)

# ---------------- EVALUATE ----------------
model.evaluate(test_data)

# ---------------- PREDICTION ----------------
for images, labels in test_data.take(1):

    img = images[0]
    true_label = labels[0]

    plt.imshow(img.numpy().astype("uint8"))
    plt.axis('off')
    plt.show()

    img_batch = tf.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)

    pred_class = np.argmax(prediction)

    print("Actual Label   :", class_names[int(true_label)])
    print("Predicted Label:", class_names[pred_class])


model.save("C://Spyder//NLP//project_img//brain_tumor_model.h5")
