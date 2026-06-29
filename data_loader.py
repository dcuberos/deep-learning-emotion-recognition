"""
Carregamento e geração de dados do EMOTIC (multi-etiqueta).

Pontos importantes:
  - As imagens são arrays .npy (RGB, 224x224x3, uint8) e são lidas com
    np.load (cv2.imread NÃO lê .npy — devolvia imagens a preto).
  - O pré-processamento é o esperado pela backbone com pesos ImageNet
    (resnet50.preprocess_input / efficientnet.preprocess_input), e NÃO um
    simples /255, que degrada o transfer learning.
  - As labels são vetores multi-hot (26), usados diretamente (sem to_categorical).
"""

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50, efficientnet

from config import CONFIG, EMOTIONS


def get_preprocess_fn(backbone):
    """Devolve a função de pré-processamento adequada à backbone."""
    if backbone == 'resnet50':
        return resnet50.preprocess_input
    if backbone == 'efficientnet':
        return efficientnet.preprocess_input
    # CNN customizada: normalização simples [0, 1]
    return lambda x: x / 255.0


def load_annotations(annotations_file):
    """
    Lê annotations.json e agrupa por split oficial do EMOTIC.

    Returns:
        dict: {'train': (paths, labels), 'val': (...), 'test': (...)}
        onde paths é np.ndarray[str] e labels é np.ndarray[float32] (N, 26).
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    splits = {}
    for split in ('train', 'val', 'test'):
        items = [a for a in data if a.get('split') == split]
        paths = np.array([a['image_path'] for a in items])
        labels = np.array([a['labels'] for a in items], dtype=np.float32)
        splits[split] = (paths, labels)
        print(f" {split}: {len(paths)} amostras")
    return splits


def compute_pos_weights(labels):
    """
    Peso por classe = nº negativos / nº positivos (para mitigar desbalanceamento).
    Usado por uma loss ponderada ou apenas para inspeção.
    """
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos = np.maximum(pos, 1.0)  # evitar divisão por zero
    return (neg / pos).astype(np.float32)


class EmotionDataGenerator(keras.utils.Sequence):
    """Gerador que carrega arrays .npy sob demanda (multi-etiqueta)."""

    def __init__(self, image_paths, labels, batch_size, img_size,
                 backbone='resnet50', augment=False, shuffle=True):
        super().__init__()
        self.image_paths = np.asarray(image_paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.img_size = tuple(img_size)
        self.augment = augment
        self.shuffle = shuffle
        self.preprocess_fn = get_preprocess_fn(backbone)
        self.indices = np.arange(len(self.image_paths))

        if augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest',
            )

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def _load_image(self, path):
        """Carrega um .npy RGB e devolve float32 (224,224,3) sem pré-processar."""
        arr = np.load(path).astype(np.float32)
        if arr.shape[:2] != self.img_size:
            arr = tf.image.resize(arr, self.img_size).numpy()
        return arr

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_indices]

        X = np.zeros((len(batch_paths), *self.img_size, 3), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            try:
                img = self._load_image(path)
                if self.augment:
                    img = self.augmentor.random_transform(img)
                X[i] = self.preprocess_fn(img)
            except Exception as e:
                print(f"Aviso: falha ao carregar {path}: {e}")

        y = self.labels[batch_indices]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTE DO MÓDULO DE CARREGAMENTO DE DADOS")
    print("=" * 70)
    ann = "data/emotic/annotations.json"
    if os.path.exists(ann):
        splits = load_annotations(ann)
        paths, labels = splits['train']
        if len(paths):
            gen = EmotionDataGenerator(
                paths[:8], labels[:8], batch_size=4,
                img_size=(CONFIG['img_height'], CONFIG['img_width']),
                backbone=CONFIG['backbone'], augment=True,
            )
            X, y = gen[0]
            print(f" Batch X: {X.shape} ({X.dtype}), y: {y.shape}")
            print(f" Faixa de X após preprocess: [{X.min():.2f}, {X.max():.2f}]")
            print(" TESTE PASSOU.")
    else:
        print(f" {ann} não existe. Execute primeiro: python load_emotic_direct.py")
