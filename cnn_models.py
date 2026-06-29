"""
Arquiteturas CNN para classificação MULTI-ETIQUETA de emoções (EMOTIC).

Decisões chave:
  - Saída: Dense(26, activation='sigmoid')  -> cada emoção é independente.
  - Loss:  BinaryFocalCrossentropy(apply_class_balancing=True) -> mitiga o
           desbalanceamento severo (ratio ~30:1) sem ter de inventar class_weights.
  - Métricas: AUC (multi_label), Precision e Recall -> "accuracy" é enganadora
              em problemas multi-etiqueta desbalanceados.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import CONFIG


def _find_backbone(model):
    """Localiza a backbone (sub-modelo Functional) dentro do modelo de transfer learning."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


class EmotionCNNModels:
    """Fábrica de arquiteturas CNN."""

    @staticmethod
    def create_custom_cnn(input_shape, num_classes):
        return models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid'),
        ])

    @staticmethod
    def _transfer_head(base_model, input_shape, num_classes):
        """Cabeça de classificação comum às backbones de transfer learning."""
        base_model.trainable = False  # fase 1: base congelada

        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='sigmoid')(x)
        return keras.Model(inputs, outputs)

    @staticmethod
    def create_resnet_transfer(input_shape, num_classes):
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        return EmotionCNNModels._transfer_head(base, input_shape, num_classes)

    @staticmethod
    def create_efficientnet_transfer(input_shape, num_classes):
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        return EmotionCNNModels._transfer_head(base, input_shape, num_classes)

    @staticmethod
    def build(backbone, input_shape, num_classes):
        """Constrói a arquitetura indicada em CONFIG['backbone']."""
        if backbone == 'resnet50':
            return EmotionCNNModels.create_resnet_transfer(input_shape, num_classes)
        if backbone == 'efficientnet':
            return EmotionCNNModels.create_efficientnet_transfer(input_shape, num_classes)
        if backbone == 'custom':
            return EmotionCNNModels.create_custom_cnn(input_shape, num_classes)
        raise ValueError(f"Backbone desconhecida: {backbone}")

    @staticmethod
    def _metrics():
        return [
            keras.metrics.AUC(name='auc', multi_label=True),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]

    @staticmethod
    def compile_model(model, learning_rate=1e-3):
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryFocalCrossentropy(
                gamma=2.0, apply_class_balancing=True
            ),
            metrics=EmotionCNNModels._metrics(),
        )
        return model

    @staticmethod
    def enable_finetuning(model, num_layers, learning_rate=1e-5):
        """
        Fase 2: descongela as últimas `num_layers` camadas da backbone e
        recompila com um learning rate baixo.
        """
        base = _find_backbone(model)
        if base is None:
            print(" Modelo sem backbone congelável (provavelmente CNN custom).")
            return model

        base.trainable = True
        for layer in base.layers[:-num_layers]:
            layer.trainable = False
        # BatchNorm congelada durante fine-tuning (boa prática de transfer learning)
        for layer in base.layers[-num_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        return EmotionCNNModels.compile_model(model, learning_rate)


def create_callbacks(model_name):
    """Callbacks de treino. Monitoriza val_auc (mais informativo que accuracy)."""
    os.makedirs('models', exist_ok=True)

    checkpoint = ModelCheckpoint(
        f'models/{model_name}_best.keras',
        monitor='val_auc', mode='max',
        save_best_only=True, verbose=1,
    )
    early_stop = EarlyStopping(
        monitor='val_auc', mode='max',
        patience=10, restore_best_weights=True, verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, min_lr=1e-7, verbose=1,
    )
    return [checkpoint, early_stop, reduce_lr]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTE DO MÓDULO DE MODELOS CNN")
    print("=" * 70)
    input_shape = (CONFIG['img_height'], CONFIG['img_width'], 3)
    n = CONFIG['num_classes']
    for bb in ('custom', 'resnet50', 'efficientnet'):
        try:
            m = EmotionCNNModels.build(bb, input_shape, n)
            m = EmotionCNNModels.compile_model(m, CONFIG['learning_rate'])
            print(f" {bb}: OK ({m.count_params():,} parâmetros)")
            del m
        except Exception as e:
            print(f" {bb}: ERRO -> {e}")
