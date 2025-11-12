"""
Funciones auxiliares para preprocesamiento de texto y gestión de secuencias.
Este módulo contiene funciones utilizadas en P2_HUI.ipynb para experimentos de preprocesamiento.
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def preprocess_text(text, lowercase=True, use_stemming=False, use_lemmatization=False, remove_stopwords=False):
    """
    Preprocesa un texto aplicando diferentes técnicas de normalización.
    
    Args:
        text (str): Texto a preprocesar
        lowercase (bool): Si True, convierte a minúsculas
        use_stemming (bool): Si True, aplica stemming (PorterStemmer)
        use_lemmatization (bool): Si True, aplica lematización (WordNetLemmatizer)
        remove_stopwords (bool): Si True, elimina stopwords en inglés
    
    Returns:
        str: Texto preprocesado
    """
    if lowercase:
        text = text.lower()
    
    tokens = text.split()
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)


def remove_values_and_indices(array, values_to_remove, sentences):
    """
    Elimina elementos de un array y las frases correspondientes basándose en valores específicos.
    
    Args:
        array: Array de numpy con las etiquetas
        values_to_remove: Lista de valores a eliminar
        sentences: Array de frases correspondientes
    
    Returns:
        tuple: (array_filtrado, frases_filtradas) - Arrays sin los elementos especificados
    """
    mask = ~np.isin(array, values_to_remove)
    filtered_array = array[mask]
    filtered_sentences = sentences[mask]
    return filtered_array, filtered_sentences


def apply_preprocessing(sentences, config):
    """
    Aplica la función de preprocesamiento a una lista de frases según la configuración.
    
    Args:
        sentences (list): Lista de frases a preprocesar
        config (dict): Diccionario con los parámetros de preprocesamiento
            - lowercase (bool): Convertir a minúsculas
            - use_stemming (bool): Aplicar stemming
            - use_lemmatization (bool): Aplicar lematización
            - remove_stopwords (bool): Eliminar stopwords
    
    Returns:
        list: Lista de frases preprocesadas
    """
    return [
        preprocess_text(
            sentence,
            lowercase=config['lowercase'],
            use_stemming=config['use_stemming'],
            use_lemmatization=config['use_lemmatization'],
            remove_stopwords=config['remove_stopwords']
        )
        for sentence in sentences
    ]


def swap_global_sequences(train_seq, val_seq, test_seq):
    """
    Sustituye temporalmente las secuencias globales para reutilizar funciones de entrenamiento.
    
    Args:
        train_seq: Secuencias de entrenamiento
        val_seq: Secuencias de validación
        test_seq: Secuencias de test
    
    Returns:
        dict: Diccionario con las secuencias originales guardadas
    """
    original_data = {
        'train_pad_sequences': globals().get('train_pad_sequences'),
        'val_pad_sequences': globals().get('val_pad_sequences'),
        'test_pad_sequences': globals().get('test_pad_sequences')
    }
    globals()['train_pad_sequences'] = train_seq
    globals()['val_pad_sequences'] = val_seq
    globals()['test_pad_sequences'] = test_seq
    return original_data


def restore_global_sequences(original_data):
    """
    Restaura las secuencias globales a sus valores originales.
    
    Args:
        original_data (dict): Diccionario con las secuencias originales
    """
    for key, value in original_data.items():
        globals()[key] = value
