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
from tensorflow.keras.callbacks import EarlyStopping

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
    
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    maxlen = max(map(len, train_sequences))
    train_pad_sequences = pad_sequences(train_sequences, padding='post', maxlen=maxlen)

    label_encoder = LabelEncoder()
    train_numerical_labels = label_encoder.fit_transform(train_labels)
    num_classes = len(np.unique(train_numerical_labels))
    train_encoded_labels = to_categorical(train_numerical_labels, num_classes)
    
    values_to_remove = ['day_name','airfare+flight','flight+airline','flight_no+airline', 'flight']

    def remove_values_and_indices(input_list, values_to_remove, other_list):
        #print(f'Initial {len(input_list)} samples')
        #print(f'Initial {len(other_list)} samples')
        indices_to_remove = [idx for idx, item in enumerate(input_list) if item in values_to_remove]
        #print(f'Removing {len(indices_to_remove)} samples')
        cleaned_list = [item for item in input_list if item not in values_to_remove]
        #print(f'Remaining {len(cleaned_list)} samples')
        cleaned_other_list = [item for idx, item in enumerate(other_list) if idx not in indices_to_remove]
        #print(f'Remaining {len(cleaned_other_list)} samples')
        return cleaned_list, np.array(cleaned_other_list)

    # Validation data
    val_sentences = list(val_data[0])
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_pad_sequences = pad_sequences(val_sequences, padding='post', maxlen=maxlen)

    val_labels = list(val_data[2])
    val_labels = list(s.replace('"', '') for s in val_labels)
    val_labels = list(s.replace(' ', '') for s in val_labels)
    val_labels, val_pad_sequences = remove_values_and_indices(val_labels, values_to_remove, val_pad_sequences)
    val_encoded_labels = to_categorical(label_encoder.transform(val_labels), num_classes)

    # Test data
    test_sentences = list(test_data[0])
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_pad_sequences = pad_sequences(test_sequences, padding='post', maxlen=maxlen)

    test_labels = list(test_data[2])
    test_labels = list(s.replace('"', '') for s in test_labels)
    test_labels = list(s.replace(' ', '') for s in test_labels)
    test_labels, test_pad_sequences = remove_values_and_indices(test_labels, values_to_remove, test_pad_sequences)
    test_encoded_labels = to_categorical(label_encoder.transform(test_labels), num_classes)

    data = {
        'train': (train_pad_sequences, train_encoded_labels),
        'val': (val_pad_sequences, val_encoded_labels),
        'test': (test_pad_sequences, test_encoded_labels),
        'num_classes': num_classes,
        'maxlen': maxlen,
        'vocab_size': len(tokenizer.word_index)
    }
    return data

def train_model(model, train_pad_sequences, train_encoded_labels, 
                val_pad_sequences, val_encoded_labels, batch_size=32, epochs=30, patience=5):
    metrics = tf.keras.metrics.F1Score(average='macro')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[metrics]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',        # o 'val_f1_score'
        patience= patience,                # detener tras 3 épocas sin mejora
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        train_pad_sequences,
        train_encoded_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_pad_sequences, val_encoded_labels),
        callbacks=[early_stop],
        verbose=0
    )
    return history

def provar_num_words(model_build, train_data, val_data, test_data, 
                     num_words_list, batch_size=32, epochs=30, patience=5, runs=5):
    '''Prueba diferentes valores de num_words para el tokenizer y grafica la loss y la metrica f1.'''
    results = {}
    for num_words in num_words_list:
        print(f'-------Running num_words={num_words}  ({runs} runs)...---------')
        data = preprocess(train_data, val_data, test_data, num_words=num_words)
        train_pad_sequences, train_encoded_labels = data['train']
        val_pad_sequences, val_encoded_labels = data['val']
        vocab_size = data['vocab_size']
        maxlen = data['maxlen']
        num_classes = data['num_classes']

        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
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
    plot(results, num_words_list, runs)
    return results


def provar_embeddings(model_build, train_data, val_data, test_data, num_words,
                      batch_size, epochs, embedding_dims, patience=5, runs=5):
    '''Prueba diferentes dimensiones de embedding, promedia 'runs' ejecuciones,
    guarda histories promediados y grafica la evolución (train vs val) por epoch.
    Devuelve: dict: {embedding_dim: averaged_history} donde cada value es np.array (epochs,)
    '''
    data = preprocess(train_data, val_data, test_data, num_words=num_words)
    train_pad_sequences, train_encoded_labels = data['train']
    val_pad_sequences, val_encoded_labels = data['val']
    vocab_size = data['vocab_size']
    maxlen = data['maxlen']
    num_classes = data['num_classes']

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

def plot(results, names, runs):
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
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print('No se encontró métrica distinta de loss en los histories. Sólo se graficó loss.')

#Para el balanceo de clases
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
def calculate_class_weights(y):
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