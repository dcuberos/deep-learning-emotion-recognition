"""
Configurações globais do projeto.

Reconhecimento de Emoções através de Posturas Corporais (dataset EMOTIC).
O EMOTIC é um problema MULTI-ETIQUETA: cada pessoa pode exibir várias das
26 emoções em simultâneo. Por isso usamos ativação sigmoid + binary
crossentropy, e não softmax + categorical crossentropy.
"""

import numpy as np

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Nomes das 26 colunas de emoção, exatamente como aparecem nos CSV do EMOTIC.
# A ordem desta lista define a ordem dos vetores multi-hot de labels.
EMOTIONS = [
    'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
    'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
    'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue', 'Embarrassment',
    'Yearning', 'Disapproval', 'Aversion', 'Annoyance', 'Anger',
    'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering',
]

# Parâmetros do projeto
CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,

    # Treino em duas fases: cabeça (base congelada) + fine-tuning (topo da base)
    'epochs_head': 15,
    'epochs_finetune': 25,
    'learning_rate': 1e-3,          # fase 1 (cabeça)
    'learning_rate_finetune': 1e-5,  # fase 2 (fine-tuning)
    'finetune_layers': 30,           # nº de camadas do topo da base a descongelar

    'num_classes': len(EMOTIONS),    # 26
    'backbone': 'resnet50',          # 'resnet50' | 'efficientnet' | 'custom'
    'classification_threshold': 0.5,  # limiar para converter probabilidades em labels
    'seed': SEED,
}
