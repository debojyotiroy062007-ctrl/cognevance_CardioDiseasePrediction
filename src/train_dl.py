import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

def train_deep_learning_pipeline(data_path="cardio_train (1).csv"):
    df = pd.read_csv(data_path, sep=';' if ';' in open(data_path).readline() else ',')
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop(columns=['cardio'])
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Multi-Layer Perceptron (Dense MLP)
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    print("Training Deep Learning Model...")
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=15, batch_size=128, verbose=1)
    
    loss, acc, auc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Deep Learning Test Results -> Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    return model

if __name__ == "__main__":
    train_deep_learning_pipeline()
