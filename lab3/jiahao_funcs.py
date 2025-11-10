import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from collections import Counter

from tensorflow.keras.callbacks import EarlyStopping

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

def preds_to_index(preds, seq_lens):
  '''
  Turn predictions to numerical indexes, flatten the sentences and discard padding.
  '''
  idx_preds = []
  for pred, seq_len in zip(preds,seq_lens):
      for l in range(seq_len):
        idx_preds.append(np.argmax(pred[l]))
  return idx_preds


def iniciar_datoset():
    #Download and extract dataset
    train_data = pd.read_csv('data/train.csv', header=None)
    val_data = train_data.tail(900)
    train_data = pd.read_csv('./data/train.csv', header=None, nrows=4078)
    test_data = pd.read_csv('data/test.csv', header=None)

    train_data = train_data.map(lambda x: x.replace('"', ''))
    val_data = val_data.map(lambda x: x.replace('"', ''))
    test_data = test_data.map(lambda x: x.replace('"', ''))

    train_sentences = train_data[0].tolist()
    train_labels = train_data[1].tolist()

    val_sentences = val_data[0].tolist()
    val_labels = val_data[1].tolist()

    test_sentences = test_data[0].tolist()
    test_labels = test_data[1].tolist()
    #Preprocess datasets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)

    max_sequence_length = max(len(seq) for seq in train_sequences)
    train_pad_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
    val_pad_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length)
    test_pad_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

    def count_unique_entities(list_of_label_sentences):
        flat_labels = []
        for labels in list_of_label_sentences:
            flat_labels += labels.split()
        unique_entities = Counter(flat_labels)
        return len(unique_entities), unique_entities
    num_unique_entities, unique_entities = count_unique_entities(train_labels)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_entities.keys()))
    train_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in train_labels]
    train_pad_labels = pad_sequences(train_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))

    def remove_sentences(list_labels, list_sequences):
        idx_to_remove = []
        labels_to_remove = set()
        for idx, labels in enumerate(list_labels):
            for label in labels.split():
                if label not in unique_entities:
                    idx_to_remove.append(idx)
                    labels_to_remove.add(label)
        labels = [elem for i, elem in enumerate(list_labels) if i not in idx_to_remove]
        sequences = [elem for i, elem in enumerate(list_sequences) if i not in idx_to_remove]
        return labels, np.array(sequences)
    test_labels_cleaned, test_pad_sequences = remove_sentences(test_labels, test_pad_sequences)
    test_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in test_labels_cleaned]
    test_pad_labels = pad_sequences(test_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))

    val_labels_cleaned, val_pad_sequences = remove_sentences(val_labels, val_pad_sequences)
    val_numerical_labels = [label_encoder.transform(labels.split()).tolist() for labels in val_labels_cleaned]
    val_pad_labels = pad_sequences(val_numerical_labels, maxlen=max_sequence_length, padding='post', value=label_encoder.transform(['O']))

    train_labels_one_hot = to_categorical(train_pad_labels, num_classes=len(label_encoder.classes_))
    test_labels_one_hot = to_categorical(test_pad_labels, num_classes=len(label_encoder.classes_))
    val_labels_one_hot = to_categorical(val_pad_labels, num_classes=len(label_encoder.classes_))

    len_test_sequences = [len(seq) for seq in test_sequences]
    
    return (train_pad_sequences, train_labels_one_hot,
            val_pad_sequences, val_labels_one_hot,
            test_pad_sequences, test_labels_one_hot,
            tokenizer, label_encoder, max_sequence_length,len_test_sequences)


def provar_embeddings(train_pad_sequences, train_encoded_labels, 
                      val_pad_sequences, val_encoded_labels, vocab_size, num_classes, 
                      batch_size, epochs,
                      maxlen, embedding_dims, patience=5, runs=5):
    '''Prueba diferentes dimensiones de embedding, promedia 'runs' ejecuciones,
    guarda histories promediados y grafica la evolución (train vs val) por epoch.
    Devuelve: dict: {embedding_dim: averaged_history} donde cada value es np.array (epochs,)
    '''
    results = {}

    for embedding_dim in embedding_dims:
        print(f'-------Running embedding_dim={embedding_dim}  ({runs} runs)...---------')
        accum = {}  # recolecta listas de arrays (runs, epochs) por clave
        for run in range(runs):
            tf.random.set_seed(run)
            np.random.seed(run)

            model = Sequential()
            model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
            model.add(GlobalMaxPooling1D(data_format='channels_last'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

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

            # acumular arrays numpy por clave
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

    # detectar métrica distinta de loss
    sample_hist = next(iter(results.values()))
    metric_keys = [k for k in sample_hist.keys() if k not in ('loss', 'val_loss')]
    metric = metric_keys[0] if metric_keys else None
    val_metric = f'val_{metric}' if metric else None

    cmap = plt.get_cmap('tab10')

    # Graficar loss (train continua, val discontinua), mismo color por dimensión
    plt.figure(figsize=(10, 4))
    for i, dim in enumerate(embedding_dims):
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
        for i, dim in enumerate(embedding_dims):
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

    return results