import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalMaxPooling1D, Dense 


def preprocess(train_data, val_data, test_data, num_words=500):
    '''Preprocess the datasets: tokenization, padding, label encoding.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
        num_words (int): Number of words to keep in the tokenizer.
        maxlen (int): Maximum length for padding sequences.
        
    Returns:
        tuple: Preprocessed training, validation, and test sequences and labels.
    '''
    # Tokenization
    train_sentences = list(train_data[0])
    train_labels = list(s.replace('"', '') for s in train_data[2])
    train_labels = list(s.replace(' ', '') for s in train_labels)

    maxlen = max(map(len, train_sequences))
    
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_pad_sequences = pad_sequences(train_sequences, padding='post', maxlen=maxlen)

    label_encoder = LabelEncoder()
    train_numerical_labels = label_encoder.fit_transform(train_labels)
    num_classes = len(np.unique(train_numerical_labels))
    train_encoded_labels = to_categorical(train_numerical_labels, num_classes)

    # Validation data
    val_sentences = list(val_data[0])
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_pad_sequences = pad_sequences(val_sequences, padding='post', maxlen=maxlen)

    val_labels = list(val_data[2])
    val_labels = list(s.replace('"', '') for s in val_labels)
    val_labels = list(s.replace(' ', '') for s in val_labels)
    val_encoded_labels = to_categorical(label_encoder.transform(val_labels), num_classes)

    # Test data
    test_sentences = list(test_data[0])
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_pad_sequences = pad_sequences(test_sequences, padding='post', maxlen=maxlen)

    test_labels = list(test_data[2])
    test_labels = list(s.replace('"', '') for s in test_labels)
    test_labels = list(s.replace(' ', '') for s in test_labels)
    test_encoded_labels = to_categorical(label_encoder.transform(test_labels), num_classes)

    return (train_pad_sequences, train_encoded_labels,
            val_pad_sequences, val_encoded_labels,
            test_pad_sequences, test_encoded_labels, num_classes, maxlen,tokenizer)


def provar_embeddings(train_pad_sequences, train_encoded_labels, 
                      val_pad_sequences, val_encoded_labels, vocab_size, num_classes, 
                      batch_size, epochs,
                      maxlen, embedding_dims):
    '''Prueba diferentes dimensiones de embedding, guarda histories y grafica la evolución
    de loss y de la métrica (train vs val) por epoch.
    Devuelve: dict: {embedding_dim: history.history}
    '''
    
    results = {}

    for embedding_dim in embedding_dims:
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
        model.add(GlobalMaxPooling1D(data_format='channels_last'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # crear la métrica (igual que en tu código principal)
        metrics = tf.keras.metrics.F1Score(average='macro')
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[metrics]
        )

        history = model.fit(
            train_pad_sequences,
            train_encoded_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_pad_sequences, val_encoded_labels),
            verbose=0
        )

        # almacenar historial
        results[embedding_dim] = history.history

        print(f'Finished embedding_dim={embedding_dim}  |  last val_loss={history.history["val_loss"][-1]:.4f}')

    # Detección de la clave de la métrica (cualquier métrica distinta de loss)
    sample_hist = next(iter(results.values()))
    metric_keys = [k for k in sample_hist.keys() if k not in ('loss', 'val_loss')]
    metric = metric_keys[0] if metric_keys else None
    val_metric = f'val_{metric}' if metric else None

    # Graficar loss
    plt.figure(figsize=(10, 4))
    for dim, hist in results.items():
        plt.plot(hist['loss'], label=f'train loss d={dim}')
        plt.plot(hist['val_loss'], linestyle='--', label=f'val loss d={dim}')
    plt.title('Training and validation loss por epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Graficar métrica si existe
    if metric:
        plt.figure(figsize=(10, 4))
        for dim, hist in results.items():
            if metric in hist and val_metric in hist:
                plt.plot(hist[metric], label=f'train {metric} d={dim}')
                plt.plot(hist[val_metric], linestyle='--', label=f'val {metric} d={dim}')
        plt.title(f'Training and validation {metric} por epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print('No se encontró métrica distinta de loss en el history. Sólo se graficó loss.')

    return results