#Download requirements.txt before
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility 
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)

# Load and preprocess the MNIST dataset (Dataset of 70,000 handwritten digits (0-9))
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to add channel dimension (28, 28, 1) 1 because it is grayscale image 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to categorical one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train_categorical.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test_categorical.shape}")

# Display sample images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.show()

# Build the CNN model
def create_cnn_model():
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    
    return model

# Create and compile the model
print("Creating CNN model...")
model = create_cnn_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Train the model
print("\nStarting training...")
history = model.fit(
    x_train, y_train_categorical,
    batch_size=128,
    epochs=30,
    validation_data=(x_test, y_test_categorical),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display some test predictions
plt.figure(figsize=(15, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted_label = predicted_classes[i]
    actual_label = y_test[i]
    confidence = np.max(predictions[i])
    
    color = 'green' if predicted_label == actual_label else 'red'
    plt.title(f'Pred: {predicted_label}, Actual: {actual_label}\nConf: {confidence:.3f}', 
              color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('Test Predictions (Green=Correct, Red=Incorrect)')
plt.tight_layout()
plt.show()

# Save the model
print("\nSaving model...")
model.save('digit_recognition_cnn.h5')
print("Model saved as 'digit_recognition_cnn.h5'")

# Function to predict single image
def predict_digit(image):
    """
    Predict digit for a single image
    image: numpy array of shape (28, 28) or (28, 28, 1)
    """
    if len(image.shape) == 2:
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3:
        image = image.reshape(1, 28, 28, 1)
    
    image = image.astype('float32') / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# Example of using the prediction function
print("\nExample prediction:")
sample_image = x_test[0]
pred_class, confidence = predict_digit(sample_image)
print(f"Predicted: {pred_class}, Actual: {y_test[0]}, Confidence: {confidence:.4f}")

print("\nTraining completed successfully!")
print(f"Final Test Accuracy: {test_accuracy:.4f}")