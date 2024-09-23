import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import mediapipe as mp

DATA_DIR = r'E:\Model\Dataset-Yoga\archive' #Your Own DataSet location
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = 'pose_classifier_advanced.h5'

# Enhanced Training Function
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = train_datagen.flow_from_directory(
        directory=os.path.join(DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_data = train_datagen.flow_from_directory(
        directory=os.path.join(DATA_DIR, 'valid'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freezing base model layers

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for early stopping and learning rate reduction
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# Enhanced Testing with Real Pose Detection
def test_model():
    model = load_model(MODEL_PATH)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    class RealPoseDetector:
        def findPose(self, img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            return img, result.pose_landmarks

        def findPosition(self, landmarks, img):
            lmList = []
            if landmarks:
                for id, lm in enumerate(landmarks.landmark):
                    h, w, _ = img.shape
                    lmList.append([id, int(lm.x * w), int(lm.y * h)])
            return lmList
        
        def predictPose(self, lmList):
            features = np.array(lmList).flatten().reshape(1, -1)
            label = model.predict(features)
            return np.argmax(label, axis=1)[0]

    cap = cv2.VideoCapture('video1.mp4')
    detector = RealPoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, landmarks = detector.findPose(img)
        lmList = detector.findPosition(landmarks, img)

        if len(lmList) != 0:
            label = detector.predictPose(lmList)
            cv2.putText(img, f'Pose: {label}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Testing completed.")

if __name__ == "__main__":
    choice = input("Enter 'train' to train the model or 'test' to test the model: ") #If you're running for first time, You have to train this model first.
    if choice == 'train':
        train_model()
    elif choice == 'test':
        test_model()
    else:
        print("Invalid choice. Please enter 'train' or 'test'.")
