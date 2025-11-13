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
from sklearn.metrics import f1_score
########### Layers ##############
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): #num_heads no es full attention
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False): #Training argument for dropout layers when training
        attn_output = self.att(inputs, inputs) #Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings
    
#################### Preprocess ########################
def preprocess_entity_recognition(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, num_words: int = None) -> Dict[str, np.ndarray|int]:
    ################## Separate sentences and labels ##############
    #Remove quotes from sentences and labels
    train_data = train_data.map(lambda x: x.replace('"', ''))
    val_data = val_data.map(lambda x: x.replace('"', ''))
    test_data = test_data.map(lambda x: x.replace('"', ''))
    # Train data
    train_sentences = train_data[0].tolist()
    train_labels = train_data[1].tolist()
    # Validation data
    val_sentences = val_data[0].tolist()
    val_labels = val_data[1].tolist()
    # Test data
    test_sentences = test_data[0].tolist()
    test_labels = test_data[1].tolist()
    ######## Tokenization and Padding of sentences ########
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    # Train data
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    max_sequence_length = max(len(seq) for seq in train_sequences)
    train_pad_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
    # Validation data
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_pad_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length)
    # Test data
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_pad_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)
    ######## ########
    train_seq_lens = [len(seq) for seq in train_sequences]
    val_seq_lens = [len(seq) for seq in val_sequences]
    test_seq_lens = [len(seq) for seq in test_sequences]
    seq_lens = {
        'train': train_seq_lens,
        'val': val_seq_lens,
        'test': test_seq_lens
    }
    ######## Label Encoding and One-Hot Encoding of labels ########
    # Aux function to count unique entities in training labels
    def count_unique_entities(y: list[str]) -> Tuple[int, Counter]:
        flat_labels = []
        for labels in y:
            flat_labels += labels.split()
        unique_entities = Counter(flat_labels)
        return len(unique_entities), unique_entities
    _, unique_entities = count_unique_entities(train_labels)
    # Train data
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_entities.keys()))
    train_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in train_labels]
    train_pad_labels = pad_sequences(train_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))
    # Aux function to remove sentences with unseen entities in val and test data
    def remove_sentences(y: list[str], X: np.ndarray):
        idx_to_remove = []
        labels_to_remove = set()
        for idx, labels in enumerate(y):
            for label in labels.split():
                if label not in unique_entities:
                    idx_to_remove.append(idx)
                    labels_to_remove.add(label)
        labels = [elem for i, elem in enumerate(y) if i not in idx_to_remove]
        sequences = [elem for i, elem in enumerate(X) if i not in idx_to_remove]
        return labels, np.array(sequences)
    # Test data
    test_labels_cleaned, test_pad_sequences = remove_sentences(test_labels, test_pad_sequences)
    test_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in test_labels_cleaned]
    test_pad_labels = pad_sequences(test_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))
    # Validation data
    val_labels_cleaned, val_pad_sequences = remove_sentences(val_labels, val_pad_sequences)
    val_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in val_labels_cleaned]
    val_pad_labels = pad_sequences(val_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))
    # One-Hot Encoding of labels
    train_labels_one_hot = to_categorical(train_pad_labels, num_classes=len(label_encoder.classes_))
    test_labels_one_hot = to_categorical(test_pad_labels, num_classes=len(label_encoder.classes_))
    val_labels_one_hot = to_categorical(val_pad_labels, num_classes=len(label_encoder.classes_))
    
    data = {
        'train_X': train_pad_sequences,
        'train_y': train_labels_one_hot,
        'val_X': val_pad_sequences,
        'val_y': val_labels_one_hot,
        'test_X': test_pad_sequences,
        'test_y': test_labels_one_hot,
        'num_classes': len(label_encoder.classes_),
        'maxlen': max_sequence_length,
        'vocab_size': len(tokenizer.word_index)+1,
        'seq_lens': seq_lens
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
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy')
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',        
        patience= patience,               
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
def predict_model (model, seq_lens, train_X, train_y, val_X, val_y ):
    train_y_pred = model.predict(train_X)
    val_y_pred = model.predict(val_X)
    def preds_to_index(preds, seq_lens):
        '''
        Turn predictions to numerical indexes, flatten the sentences and discard padding.
        '''
        idx_preds = []
        for pred, seq_len in zip(preds,seq_lens):
            for l in range(seq_len):
                idx_preds.append(np.argmax(pred[l]))
        return idx_preds
    
    train_y_idx = preds_to_index (train_y, seq_lens["train"])
    val_y_idx = preds_to_index (val_y, seq_lens["val"])
    train_y_pred_idx = preds_to_index(train_y_pred, seq_lens["train"])
    val_y_pred_idx = preds_to_index(val_y_pred, seq_lens["val"])
    train_f1 = f1_score(train_y_idx, train_y_pred_idx, average='macro', zero_division=1.0 )
    val_f1 = f1_score(val_y_idx, val_y_pred_idx, average='macro', zero_division=1.0 )
    return train_f1, val_f1

def provar_num_words(model_build: callable, preprocess: callable, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                     test_data: pd.DataFrame, num_words_list: list[int], batch_size: int = 32, epochs: int = 30, 
                     patience: int = 5, runs: int = 5) -> Dict[int, Dict[str, np.ndarray]]:
    '''
    Prueba diferentes cantidades de palabras (num_words), promedia 'runs' ejecuciones,
    guarda histories promediados y grafica la evolución (train vs val) por epoch.
    Devuelve: dict: {num_words: averaged_history} donde cada value es np.array (epochs,)
    '''
    results = {}
    f1_average = {}
    train_f1_list = [] #list with final f1 of each run
    val_f1_list = [] 
    for num_words in num_words_list:
        print(f'-------Running num_words={num_words}  ({runs} runs)...---------')
        data = preprocess(train_data, val_data, test_data, num_words=num_words)
        train_pad_sequences, train_encoded_labels = data['train_X'], data['train_y']
        val_pad_sequences, val_encoded_labels = data['val_X'], data['val_y']
        vocab_size = data['vocab_size']
        maxlen = data['maxlen']
        num_classes = data['num_classes']
        seq_lens = data['seq_lens']

        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        train_f1_mean = 0
        val_f1_mean = 0
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = model_build(num_classes, vocab_size, maxlen)
            history = train_model(model, train_pad_sequences, train_encoded_labels,
                                  val_pad_sequences, val_encoded_labels,
                                  batch_size=batch_size, epochs=epochs, patience=patience)
            # Run report
            for k, v in history.history.items():
                accum.setdefault(k, []).append(np.array(v))

            print(f'  run {run+1}/{runs} done | last val_loss={history.history["val_loss"][-1]:.4f}')

            train_f1, val_f1 = predict_model(model, seq_lens, train_pad_sequences, train_encoded_labels, val_pad_sequences, val_encoded_labels)
            train_f1_mean += train_f1
            val_f1_mean += val_f1
        train_f1_list.append(train_f1_mean / runs)
        val_f1_list.append(val_f1_mean / runs)
        # promediar por key -> numpy arrays (epochs,), respetando EarlyStopping
        averaged = {}
        for k, arrs in accum.items():
            max_len = max(a.shape[0] for a in arrs)
            stacked = np.full((len(arrs), max_len), np.nan, dtype=float)
            for i, a in enumerate(arrs):
                stacked[i, :a.shape[0]] = a
            averaged[k] = np.nanmean(stacked, axis=0)  # media ignorando NaNs

        results[num_words] = averaged

        # calcular último val_loss válido (puede haber NaNs si ninguna run llegó a la última época)
        val_loss_avg = averaged.get("val_loss")
        if val_loss_avg is not None:
            finite_idx = np.where(~np.isnan(val_loss_avg))[0]
            last_val_loss = val_loss_avg[finite_idx[-1]] if finite_idx.size else np.nan
        else:
            last_val_loss = np.nan

        print(f'Finished num_words={num_words}  |  averaged last val_loss={last_val_loss:.4f}')
        f1_average[num_words] = {"Train": np.array(train_f1_list), "Val": np.array(val_f1_list)}
    print(f1_average)
    plot_loss(results, num_words_list)
    plot_f1(f1_average, num_words_list)
    return results


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
    f1_average = {}
    train_f1_list = [] #list with final f1 of each embedding_dim
    val_f1_list = [] 
    for embedding_dim in embedding_dims:
        print(f'-------Running embedding_dim={embedding_dim}  ({runs} runs)...---------')
        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        train_f1_mean = 0
        val_f1_mean = 0
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = model_build(model, num_classes, vocab_size, maxlen, embedding_dim)

            history = train_model(model, train_pad_sequences, train_encoded_labels,
                                  val_pad_sequences, val_encoded_labels,
                                  batch_size=batch_size, epochs=epochs, patience=patience)

            # Run report
            for k, v in history.history.items():
                accum.setdefault(k, []).append(np.array(v))

            print(f'  run {run+1}/{runs} done | last val_loss={history.history["val_loss"][-1]:.4f}')
            train_f1, val_f1 = predict_model(model, maxlen, train_pad_sequences, train_encoded_labels, val_pad_sequences, val_encoded_labels)
            train_f1_mean += train_f1
            val_f1_mean += val_f1
        train_f1_list.append(train_f1_mean / runs)
        val_f1_list.append(val_f1_mean / runs)
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
        f1_average[embedding_dim] = {"Train": np.array(train_f1_list), "Val": np.array(val_f1_list)}

    plot_loss(results, embedding_dims)
    plot_f1(f1_average, embedding_dims)
    return results

def plot_loss(results: Dict[int, Dict[str, np.ndarray]], names: list[int]):
    '''
    Grafica la evolución de loss y otra métrica (si existe) en train y val por epoch,
    promediando sobre 'runs' ejecuciones.
    '''
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

# ...existing code...
def plot_f1(f1_averaged: Dict[int, Dict[str, float]], names: list[int]):
    """Graficar los f1-scores para cada name. Aquí 'dim' será el eje Y."""
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(6, 8))

    dims = []
    train_vals = []
    val_vals = []

    for i, dim in enumerate(names):
        hist = f1_averaged.get(dim)
        if hist is None:
            continue
        # extraer un valor escalar por configuración: si es array tomar el último (f1 final)
        t = np.asarray(hist['Train'])
        v = np.asarray(hist['Val'])
        train_val = float(t[-1]) if t.size > 1 else float(t)
        val_val = float(v[-1]) if v.size > 1 else float(v)

        dims.append(dim)
        train_vals.append(train_val)
        val_vals.append(val_val)

    if not dims:
        print("No hay datos en f1_averaged para las names dadas.")
        return

    # dibujar líneas horizontales / puntos (eje X = F1, eje Y = dim)
    dims = np.array(dims)
    train_vals = np.array(train_vals)
    val_vals = np.array(val_vals)

    plt.figure(figsize=(10, 4))
    plt.plot(dims, train_vals, marker='o', linestyle='-', color='C0', label='Train')
    plt.plot(dims, val_vals, marker='x', linestyle='--', color='C1', label='Val')
    plt.title('Train vs Val F1 por dimensión')
    plt.xlabel('Dim')
    plt.ylabel('F1 Score')
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

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
    vocab_size = preprocessed_data['vocab_size']
    maxlen = preprocessed_data['maxlen']
    num_classes = preprocessed_data['num_classes']

    results = {}

    for class_weights in class_weights_list:
        print(f'-------Running class_weights={class_weights}  ({runs} runs)...---------')
        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = model_build(num_classes, vocab_size, maxlen)

            history = train_model(model, train_pad_sequences, train_encoded_labels,
                                  val_pad_sequences, val_encoded_labels,
                                  batch_size=batch_size, epochs=epochs, patience=patience,
                                  class_weights=class_weights)

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

        print(f'Finished embedding_dim={key}  |  averaged last val_loss={last_val_loss:.4f}')

    plot(results, keys, runs)

    return results