import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Counter, Tuple, Dict
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalMaxPooling1D, Dense 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from funciones_H import preprocess_text
################################## Preprocess ################################
def preprocess_intent_recognition(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, num_words: int = None) -> Dict[str, np.ndarray|int]:
    '''
    Preprocess the datasets: tokenization, padding, label encoding and delate unseen cases in validation and test datasets.
    
    Args:
        train_data, val_data, test_data (pd.DataFrame): Training, validation, and test datasets with text and labels 
                                                        in columns 0 and 2 respectively.
        num_words (int): Number of words to keep in the tokenizer. If None, all words are kept.
        
    Returns:
        data (dict): A dictionary containing preprocessed training, validation, and test data, along with metadata such as 
                    number of classes, maximum sequence length, and vocabulary size.
    '''
    ####### Separete sentences and labels ########

    # Training data
    train_sentences = list(train_data[0])
    train_sentences = [preprocess_text(s, use_stemming=True) for s in train_sentences]
    train_labels = list(s.replace('"', '') for s in train_data[2])
    train_labels = list(s.replace(' ', '') for s in train_labels)
    # Validation data
    val_sentences = list(val_data[0])
    val_sentences = [preprocess_text(s, use_stemming=True) for s in val_sentences]
    val_labels = list(val_data[2])
    val_labels = list(s.replace('"', '') for s in val_labels)
    val_labels = list(s.replace(' ', '') for s in val_labels)
    # Test data
    test_sentences = list(test_data[0])
    test_sentences = [preprocess_text(s, use_stemming=True) for s in test_sentences]
    test_labels = list(test_data[2])
    test_labels = list(s.replace('"', '') for s in test_labels)
    test_labels = list(s.replace(' ', '') for s in test_labels)

    ######## Tokenization and Padding of sentences ########    

    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_sentences)
    # Train data
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    maxlen = max(map(len, train_sequences))
    train_pad_sequences = pad_sequences(train_sequences, padding='post', maxlen=maxlen)
    # Validation data
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_pad_sequences = pad_sequences(val_sequences, padding='post', maxlen=maxlen)
    # Test data
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_pad_sequences = pad_sequences(test_sequences, padding='post', maxlen=maxlen)

    ######## Label Encoding and One-Hot Encoding of labels ########

    label_encoder = LabelEncoder()
    # Training data
    train_numerical_labels = label_encoder.fit_transform(train_labels)
    num_classes = len(np.unique(train_numerical_labels))
    train_encoded_labels = to_categorical(train_numerical_labels, num_classes)
    # Aux function to remove unseen labels and corresponding sequences in val and test data
    values_to_remove = ['day_name','airfare+flight','flight+airline','flight_no+airline', 'flight']
    def remove_values_and_indices(input_list, values_to_remove, other_list):
        indices_to_remove = [idx for idx, item in enumerate(input_list) if item in values_to_remove]
        cleaned_list = [item for item in input_list if item not in values_to_remove]
        cleaned_other_list = [item for idx, item in enumerate(other_list) if idx not in indices_to_remove]
        return cleaned_list, np.array(cleaned_other_list)
    # Validation data
    val_labels, val_pad_sequences = remove_values_and_indices(val_labels, values_to_remove, val_pad_sequences)
    val_encoded_labels = to_categorical(label_encoder.transform(val_labels), num_classes)
    # Test data
    test_labels, test_pad_sequences = remove_values_and_indices(test_labels, values_to_remove, test_pad_sequences)
    test_encoded_labels = to_categorical(label_encoder.transform(test_labels), num_classes)

    ##### Return preprocessed data and metadata ########
    data = {
        'train_X': train_pad_sequences,
        'train_y': train_encoded_labels,
        'val_X': val_pad_sequences,
        'val_y': val_encoded_labels,
        'test_X': test_pad_sequences,
        'test_y': test_encoded_labels,
        'num_classes': num_classes,
        'maxlen': maxlen,
        'vocab_size': num_words +1
    }
    return data
################################### Model training ################################
def train_model(model: tf.keras.Model, train_pad_sequences: np.ndarray, train_encoded_labels: np.ndarray, 
                val_pad_sequences: np.ndarray, val_encoded_labels: np.ndarray, class_weights: dict = None, 
                batch_size: int = 32, epochs: int = 30, patience: int = 5)-> tf.keras.callbacks.History:
    '''
    Train the model with EarlyStopping based on validation loss.
        Args:
            model (tf.keras.Model): The Keras model to be trained.
            train_pad_sequences (np.ndarray): Padded training sequences.
            train_encoded_labels (np.ndarray): One-hot encoded training labels.
            val_pad_sequences (np.ndarray): Padded validation sequences.
            val_encoded_labels (np.ndarray): One-hot encoded validation labels.
            class_weights (dict): Optional dictionary mapping class indices to weights for handling class imbalance.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Maximum number of epochs to train the model.
            patience (int): Number of epochs with no improvement after which training will be stopped.
        Returns:
            history (tf.keras.callbacks.History): A record of training loss values and metrics values at successive epochs.
    '''
    # Compile the model with F1 Score metric
    metrics = tf.keras.metrics.F1Score(average='macro')
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[metrics])
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',        # o 'val_f1_score'
        patience= patience,                # detener tras 3 épocas sin mejora
        restore_best_weights=True,
        verbose=0)
    # Train the model
    history = model.fit(
        train_pad_sequences,
        train_encoded_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_pad_sequences, val_encoded_labels),
        callbacks=[early_stop],
        verbose=0,
        class_weight=class_weights)
    return history



def provar_embeddings(model_build: callable, preprocessed_data: Dict[str, np.ndarray|int], embedding_dims: list[int], 
                      batch_size: int = 32, epochs: int = 30, patience=5, runs=5):
    '''
    Prueba diferentes dimensiones de embedding, promedia 'runs' ejecuciones,
    guarda histories promediados y grafica la evolución (train vs val) por epoch.
    Devuelve: dict: {embedding_dim: averaged_history} donde cada value es np.array (epochs,)
    '''
    train_pad_sequences, train_encoded_labels = preprocessed_data['train_X'], preprocessed_data['train_y']
    val_pad_sequences, val_encoded_labels = preprocessed_data['val_X'], preprocessed_data['val_y']
    vocab_size = preprocessed_data['vocab_size']
    maxlen = preprocessed_data['maxlen']
    num_classes = preprocessed_data['num_classes']

    results = {}

    for embedding_dim in embedding_dims:
        print(f'-------Running embedding_dim={embedding_dim}  ({runs} runs)...---------')
        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = Sequential()
            model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
            model = model_build(model, num_classes)

            history = train_model(model, train_pad_sequences, train_encoded_labels,
                                  val_pad_sequences, val_encoded_labels,
                                  batch_size=batch_size, epochs=epochs, patience=patience)

            # Run report
            for k, v in history.history.items():
                accum.setdefault(k, []).append(np.array(v))

            print(f'  run {run+1}/{runs} done | last val_loss={history.history["val_loss"][-1]:.4f}')

        # promediar por key -> numpy arrays (epochs,), respetando EarlyStopping
        averaged = {}
        for k, arrs in accum.items():
            max_len = max(a.shape[0] for a in arrs)
            stacked = np.full((len(arrs), max_len), np.nan, dtype=float)
            for i, a in enumerate(arrs):
                stacked[i, :a.shape[0]] = a
            averaged[k] = np.nanmean(stacked, axis=0)  # media ignorando NaNs

        results[embedding_dim] = averaged

        # calcular último val_loss válido (puede haber NaNs si ninguna run llegó a la última época)
        val_loss_avg = averaged.get("val_loss")
        if val_loss_avg is not None:
            finite_idx = np.where(~np.isnan(val_loss_avg))[0]
            last_val_loss = val_loss_avg[finite_idx[-1]] if finite_idx.size else np.nan
        else:
            last_val_loss = np.nan

        print(f'Finished embedding_dim={embedding_dim}  |  averaged last val_loss={last_val_loss:.4f}')

    plot(results, embedding_dims, runs)

    return results

def plot(results: Dict[int, Dict[str, np.ndarray]], names: list[int], runs: int):
    '''
    Grafica la evolución de loss y otra métrica (si existe) en train y val por epoch,
    promediando sobre 'runs' ejecuciones.
    '''
    # detectar métrica distinta de loss
    sample_hist = next(iter(results.values()))
    metric_keys = [k for k in sample_hist.keys() if k not in ('loss', 'val_loss')]
    metric = metric_keys[0] if metric_keys else None
    val_metric = f'val_{metric}' if metric else None

    cmap = plt.get_cmap('tab10')

    # Graficar loss (train continua, val discontinua), mismo color por dimensión
    plt.figure(figsize=(10, 4))
    for i, dim in enumerate(names):
        hist = results.get(dim)
        if hist is None:
            continue
        color = cmap(i % cmap.N)
        plt.plot(hist['loss'], label=f'train loss d={dim}', color=color, linestyle='-')
        plt.plot(hist['val_loss'], label=f'val loss d={dim}', color=color, linestyle='--')
    plt.title('Training and validation loss (averaged) por epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Graficar métrica promedio (train continua, val discontinua), usando la media sobre runs
    have_metric = metric and any((metric in hist and val_metric in hist) for hist in results.values())
    if have_metric:
        plt.figure(figsize=(10, 4))
        for i, dim in enumerate(names):
            hist = results.get(dim)
            if hist is None:
                continue
            if metric in hist and val_metric in hist:
                color = cmap(i % cmap.N)
                plt.plot(hist[metric], label=f'train {metric} d={dim}', color=color, linestyle='-')
                plt.plot(hist[val_metric], label=f'val {metric} d={dim}', color=color, linestyle='--')
        plt.title(f'Training and validation {metric} (averaged over {runs} runs) por epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(loc='upper left', fontsize='small')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print('No se encontró métrica distinta de loss en los histories. Sólo se graficó loss.')

def calculate_class_weights(y: np.ndarray) -> dict[int, float]:
    """Calculate class weights to handle class imbalance. The formula used is:

    class_weight = total_samples / (num_classes * class_count)

    input: numpy array of one-hot encoded labels
    output: dictionary mapping class indices to weights
    """

    # Convert one-hot encoded labels to class indices
    y_indices = np.argmax(y, axis=1)

    # Get unique classes
    classes = np.unique(y_indices)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_indices)

    # Create a dictionary mapping class indices to weights
    class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
    return class_weight_dict

def probar_class_weights(model_build: callable, preprocessed_data: Dict[str, np.ndarray|int], 
                         class_weights_list: list[dict[int, float]|None], batch_size: int=32, epochs: int=10, patience=5, runs=5):
    '''
    Prueba diferentes dimensiones de embedding, promedia 'runs' ejecuciones,
    guarda histories promediados y grafica la evolución (train vs val) por epoch.
    Devuelve: dict: {embedding_dim: averaged_history} donde cada value es np.array (epochs,)
    '''
    train_pad_sequences, train_encoded_labels = preprocessed_data['train_X'], preprocessed_data['train_y']
    val_pad_sequences, val_encoded_labels = preprocessed_data['val_X'], preprocessed_data['val_y']
    test_pad_sequences, test_encoded_labels = preprocessed_data['test_X'], preprocessed_data['test_y']
    vocab_size = preprocessed_data['vocab_size']
    maxlen = preprocessed_data['maxlen']
    num_classes = preprocessed_data['num_classes']

    results = {}
    test_evaluations = {}
    for class_weights in class_weights_list:
        print(f'-------Running class_weights={class_weights}  ({runs} runs)...---------')
        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        test_loss_mean = 0
        test_f1_mean = 0
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = model_build(num_classes, vocab_size, maxlen)

            history = train_model(model, train_pad_sequences, train_encoded_labels,
                                  val_pad_sequences, val_encoded_labels,
                                  batch_size=batch_size, epochs=epochs, patience=patience,
                                  class_weights=class_weights)
            test_loss, test_f1 = model.evaluate(test_pad_sequences, test_encoded_labels, verbose=0)
            test_loss_mean += test_loss
            test_f1_mean += test_f1

            # Run report
            for k, v in history.history.items():
                accum.setdefault(k, []).append(np.array(v))

            print(f'  run {run+1}/{runs} done | last val_loss={history.history["val_loss"][-1]:.4f}')

        # promediar por key -> numpy arrays (epochs,), respetando EarlyStopping
        averaged = {}
        for k, arrs in accum.items():
            max_len = max(a.shape[0] for a in arrs)
            stacked = np.full((len(arrs), max_len), np.nan, dtype=float)
            for i, a in enumerate(arrs):
                stacked[i, :a.shape[0]] = a
            averaged[k] = np.nanmean(stacked, axis=0)  # media ignorando NaNs
        keys = ['no_class_weights', 'with_class_weights']
        if class_weights is None:
            key = keys[0]
            results[key] = averaged
        else:
            key = keys[1]
            results[key] = averaged

        # calcular último val_loss válido (puede haber NaNs si ninguna run llegó a la última época)
        val_loss_avg = averaged.get("val_loss")
        if val_loss_avg is not None:
            finite_idx = np.where(~np.isnan(val_loss_avg))[0]
            last_val_loss = val_loss_avg[finite_idx[-1]] if finite_idx.size else np.nan
        else:
            last_val_loss = np.nan
        test_evaluations[key] = {
            'test_loss': test_loss_mean / runs,
            'test_f1': test_f1_mean / runs
        }

        print(f'Finished embedding_dim={key}  |  averaged last val_loss={last_val_loss:.4f}')
    for k, v in test_evaluations.items():
        print(f'Test evaluation for {k}: Loss={v["test_loss"]:.4f}, F1 Score={v["test_f1"]:.4f}')

    plot(results, keys, runs)
    return results, test_evaluations