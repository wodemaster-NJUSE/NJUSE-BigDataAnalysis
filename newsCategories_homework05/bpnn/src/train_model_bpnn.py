import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def train_model_bpnn(X_train_tfidf, y_train, num_classes):
    input_size = X_train_tfidf.shape[1]
    hidden_size = 512
    output_size = num_classes

    model = Sequential()
    model.add(Dense(512, input_shape=(input_size,), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Convert the sparse matrix to a dense NumPy array
    X_train_tfidf_array = np.array(X_train_tfidf.toarray())

    model.fit(X_train_tfidf_array, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=2)

    return model
