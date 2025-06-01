import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return X_train, X_test, y_train, y_test, class_names


def preprocess_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, X_test, y_train, y_test

 

def visualize_data_samples(X_train, y_train, class_names):
    plt.figure(figsize=(10, 10))
    
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        labels = np.argmax(y_train, axis=1)
    else:
        labels = y_train.squeeze()
    
    # Display a 5x5 grid of images
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i])
        plt.title(class_names[labels[i]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

 

 

def train_simple_cnn(X_train, y_train, X_val, y_val, input_shape=(32, 32, 3)):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("\nTraining Simple CNN...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

 

def train_deep_cnn(X_train, y_train, X_val, y_val, input_shape=(32, 32, 3)):

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("\nTraining Deep CNN...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
  
    return model, history

def evaluate_model(model, X_test, y_test, class_names, model_name="Model"):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    return accuracy

 

def visualize_training_history(history, model_name="Model"):

 

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

 

def visualize_predictions(model, X_test, y_test, class_names, num_images=25):

    predictions = model.predict(X_test[:num_images])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_images], axis=1)
    
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[i])
        
        color = 'green' if pred_classes[i] == true_classes[i] else 'red'
        title = f"True: {class_names[true_classes[i]]}\nPred: {class_names[pred_classes[i]]}"
        
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_mnist_cnn():
    print("Loading MNIST handwritten digits dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {np.argmax(y_train[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining MNIST CNN...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=5,  # Fewer epochs needed for MNIST
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    training_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nMNIST CNN Test Accuracy: {accuracy:.4f}")
    print(f"Training completed in {training_time:.2f} seconds")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('MNIST Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('MNIST Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    predictions = model.predict(X_test[:25])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:25], axis=1)
    
    plt.figure(figsize=(12, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        color = 'green' if pred_classes[i] == true_classes[i] else 'red'
        plt.title(f"True: {true_classes[i]}\nPred: {pred_classes[i]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('MNIST Confusion Matrix')
    plt.show()
    
    return accuracy

def main():
    start_time = time.time()


    print("======= MNIST Handwritten Digits Classification =======")
    mnist_accuracy = train_mnist_cnn()
    print(f"\nMNIST classification completed with accuracy: {mnist_accuracy:.4f}")
    print("\n\n======= CIFAR-10 Image Classification =======")
    
    print("Loading data...")
    X_train, X_test, y_train, y_test, class_names = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
    
    print("\nVisualizing sample images...")
    visualize_data_samples(X_train, y_train, class_names)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Store results for all models
    models_results = []
    
    model_start_time = time.time()
    simple_model, simple_history = train_simple_cnn(X_train, y_train, X_val, y_val)
    model_training_time = time.time() - model_start_time
    
    simple_accuracy = evaluate_model(simple_model, X_test, y_test, class_names, "Simple CNN")
    visualize_training_history(simple_history, "Simple CNN")
    visualize_predictions(simple_model, X_test, y_test, class_names)
    
    models_results.append({
        'name': 'Simple CNN',
        'model': simple_model,
        'accuracy': simple_accuracy,
        'params': simple_model.count_params(),
        'training_time': model_training_time
    })
    


    model_start_time = time.time()
    deep_model, deep_history = train_deep_cnn(X_train, y_train, X_val, y_val)
    model_training_time = time.time() - model_start_time
    
    deep_accuracy = evaluate_model(deep_model, X_test, y_test, class_names, "Deep CNN")
    visualize_training_history(deep_history, "Deep CNN")
    visualize_predictions(deep_model, X_test, y_test, class_names)
    
    models_results.append({
        'name': 'Deep CNN',
        'model': deep_model,
        'accuracy': deep_accuracy,
        'params': deep_model.count_params(),
        'training_time': model_training_time
    })
   


if __name__ == "__main__":
    main()
