"""
Script de treino do projeto — Reconhecimento de Emoções através de Posturas Corporais.

Pipeline:
  1. Carrega annotations.json (multi-etiqueta) usando os splits oficiais do EMOTIC.
  2. Constrói a backbone escolhida em CONFIG['backbone'].
  3. Treino em 2 fases: cabeça (base congelada) + fine-tuning (topo da base).
  4. Avalia no conjunto de teste com métricas multi-etiqueta (mAP, F1, AP/classe).
  5. Guarda modelo (.keras) e gráficos de treino.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import tensorflow as tf

from config import CONFIG, EMOTIONS, SEED
from data_loader import EmotionDataGenerator, load_annotations, compute_pos_weights
from cnn_models import EmotionCNNModels, create_callbacks
from evaluation import ModelEvaluator

tf.random.set_seed(SEED)
ANNOTATIONS_FILE = "data/emotic/annotations.json"


def _make_gen(paths, labels, augment):
    return EmotionDataGenerator(
        paths, labels,
        batch_size=CONFIG['batch_size'],
        img_size=(CONFIG['img_height'], CONFIG['img_width']),
        backbone=CONFIG['backbone'],
        augment=augment,
        shuffle=augment,
    )


def _plot_history(histories, save_path):
    """Concatena o histórico das 2 fases e plota loss/auc/precision/recall."""
    keys = ['loss', 'auc', 'precision', 'recall']
    merged = {k: [] for k in keys}
    merged_val = {k: [] for k in keys}
    for h in histories:
        for k in keys:
            merged[k] += h.history.get(k, [])
            merged_val[k] += h.history.get(f'val_{k}', [])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, k in zip(axes.ravel(), keys):
        ax.plot(merged[k], label='Treino')
        ax.plot(merged_val[k], label='Validação')
        ax.set_title(k.upper())
        ax.set_xlabel('Época')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Gráficos de treino salvos em {save_path}")


def train_model():
    print("=" * 70)
    print(" RECONHECIMENTO DE EMOÇÕES ATRAVÉS DE POSTURAS CORPORAIS")
    print(f" Backbone: {CONFIG['backbone']}  |  Classes: {CONFIG['num_classes']}")
    print("=" * 70)

    if not os.path.exists(ANNOTATIONS_FILE):
        print(f" ERRO: {ANNOTATIONS_FILE} não encontrado.")
        print(" Execute primeiro: python load_emotic_direct.py")
        sys.exit(1)

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ---- Dados (splits oficiais) ----
    print("\nPASSO 1: Carregar dados...")
    splits = load_annotations(ANNOTATIONS_FILE)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    pos_weights = compute_pos_weights(y_train)
    print(f" Desbalanceamento (neg/pos) — máx: {pos_weights.max():.1f}, "
          f"mín: {pos_weights.min():.1f}")

    train_gen = _make_gen(X_train, y_train, augment=True)
    val_gen = _make_gen(X_val, y_val, augment=False)

    # ---- Modelo ----
    print("\nPASSO 2: Construir modelo...")
    input_shape = (CONFIG['img_height'], CONFIG['img_width'], 3)
    model = EmotionCNNModels.build(CONFIG['backbone'], input_shape, CONFIG['num_classes'])
    model = EmotionCNNModels.compile_model(model, CONFIG['learning_rate'])
    print(f" Modelo criado ({model.count_params():,} parâmetros)")

    callbacks = create_callbacks(f"emotion_{CONFIG['backbone']}")
    histories = []

    # ---- Fase 1: cabeça ----
    print("\nPASSO 3: Treino fase 1 (cabeça, base congelada)...")
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CONFIG['epochs_head'], callbacks=callbacks, verbose=1,
    )
    histories.append(h1)

    # ---- Fase 2: fine-tuning ----
    if CONFIG['backbone'] in ('resnet50', 'efficientnet') and CONFIG['epochs_finetune'] > 0:
        print("\nPASSO 4: Treino fase 2 (fine-tuning do topo da base)...")
        model = EmotionCNNModels.enable_finetuning(
            model, CONFIG['finetune_layers'], CONFIG['learning_rate_finetune'],
        )
        h2 = model.fit(
            train_gen, validation_data=val_gen,
            epochs=CONFIG['epochs_finetune'], callbacks=callbacks, verbose=1,
        )
        histories.append(h2)

    _plot_history(histories, 'results/training_history.png')

    # ---- Avaliação ----
    print("\nPASSO 5: Avaliar no conjunto de teste...")
    evaluator = ModelEvaluator(model, class_names=EMOTIONS)
    evaluator.evaluate(X_test, y_test, batch_size=CONFIG['batch_size'])

    # ---- Guardar ----
    model_path = f"models/emotion_{CONFIG['backbone']}_final.keras"
    model.save(model_path)
    with open('models/emotions.json', 'w', encoding='utf-8') as f:
        json.dump(EMOTIONS, f, ensure_ascii=False, indent=2)
    print(f"\n Modelo salvo: {model_path}")
    print(" Ordem das emoções salva: models/emotions.json")
    print("\n TREINO CONCLUÍDO.")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f" GPU detectada: {len(gpus)} dispositivo(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print(" Sem GPU — treino em CPU (mais lento).")

    try:
        train_model()
    except KeyboardInterrupt:
        print("\n Treino interrompido pelo utilizador.")
        sys.exit(0)
    except Exception as e:
        print(f"\n ERRO durante treino: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
