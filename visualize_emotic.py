"""
Notebook para visualizar amostras do dataset EMOTIC
Mostra exemplos de diferentes emoções com suas imagens
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

os.makedirs('results', exist_ok=True)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Carregar anotações
ANNOTATIONS_FILE = "data/emotic/annotations.json"

print("Carregando dataset EMOTIC...")
with open(ANNOTATIONS_FILE, 'r') as f:
    annotations = json.load(f)

print(f" Total de imagens: {len(annotations)}")

# ============================================================================
# ESTATÍSTICAS DO DATASET
# ============================================================================

# Contar emoções (multi-etiqueta: cada anotação pode ter várias)
emotions = [emo for ann in annotations for emo in ann['emotions']]
emotion_counts = Counter(emotions)

print(f"\n Distribuição de Emoções:")
print(f"   Total de categorias: {len(emotion_counts)}")
print(f"\n   Top 10 emoções:")
for emotion, count in emotion_counts.most_common(10):
    percentage = (count / len(annotations)) * 100
    print(f"   {emotion:20s}: {count:5d} ({percentage:5.2f}%)")

# ============================================================================
# VISUALIZAÇÃO: TOP 10 EMOÇÕES
# ============================================================================

# Gráfico de barras
top_10 = emotion_counts.most_common(10)
emotions_names = [e[0] for e in top_10]
emotions_values = [e[1] for e in top_10]

plt.figure(figsize=(12, 6))
bars = plt.bar(emotions_names, emotions_values, color='steelblue', alpha=0.8)
plt.xlabel('Emoção', fontsize=12)
plt.ylabel('Número de Imagens', fontsize=12)
plt.title('Top 10 Emoções no Dataset EMOTIC', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, value in zip(bars, emotions_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(value)}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/emotion_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Gráfico salvo: results/emotion_distribution.png")

# ============================================================================
# VISUALIZAÇÃO: AMOSTRAS DE CADA EMOÇÃO
# ============================================================================

def load_and_display_samples(emotion, num_samples=6):
    """
    Carrega e exibe amostras de uma emoção específica
    
    Args:
        emotion: nome da emoção
        num_samples: número de amostras a mostrar
    """
    # Filtrar anotações desta emoção
    emotion_annotations = [ann for ann in annotations if emotion in ann['emotions']]

    if len(emotion_annotations) == 0:
        print(f" Nenhuma imagem encontrada para {emotion}")
        return
    
    # Selecionar amostras aleatórias
    num_samples = min(num_samples, len(emotion_annotations))
    samples = random.sample(emotion_annotations, num_samples)
    
    # Criar grid de visualização
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, ann in enumerate(samples):
        try:
            # Carregar imagem .npy
            img = np.load(ann['image_path'], allow_pickle=True)
            
            # Normalizar se necessário
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Exibir
            axes[idx].imshow(img)
            axes[idx].set_title(f'{emotion} (Imagem {idx+1})', fontsize=12)
            axes[idx].axis('off')
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Erro ao carregar', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.suptitle(f'Exemplos de Emoção: {emotion}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/samples_{emotion.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MOSTRAR AMOSTRAS DAS TOP 5 EMOÇÕES
# ============================================================================

print("\n" + "="*70)
print(" VISUALIZANDO AMOSTRAS DAS TOP 5 EMOÇÕES")
print("="*70 + "\n")

top_5_emotions = [e[0] for e in emotion_counts.most_common(5)]

for emotion in top_5_emotions:
    print(f"\nMostrando exemplos de: {emotion}")
    load_and_display_samples(emotion, num_samples=6)
    print(f" Salvo: results/samples_{emotion.lower()}.png")

# ============================================================================
# GRID COMPARATIVO: 1 EXEMPLO DE CADA EMOÇÃO TOP 10
# ============================================================================

print("\n" + "="*70)
print(" CRIANDO GRID COMPARATIVO (1 EXEMPLO DE CADA TOP 10)")
print("="*70 + "\n")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for idx, emotion in enumerate(emotions_names):
    try:
        # Pegar uma amostra aleatória desta emoção
        emotion_annotations = [ann for ann in annotations if emotion in ann['emotions']]
        sample = random.choice(emotion_annotations)
        
        # Carregar imagem
        img = np.load(sample['image_path'], allow_pickle=True)
        
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Exibir
        axes[idx].imshow(img)
        axes[idx].set_title(f'{emotion}\n({emotion_counts[emotion]} imagens)', 
                           fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'{emotion}\n(erro)', 
                      ha='center', va='center', fontsize=10)
        axes[idx].axis('off')

plt.suptitle('Dataset EMOTIC - Top 10 Emoções (1 Exemplo de Cada)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/emotion_comparison_grid.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Salvo: results/emotion_comparison_grid.png")

# ============================================================================
# ESTATÍSTICAS FINAIS
# ============================================================================

print("\n" + "="*70)
print(" RESUMO DO DATASET")
print("="*70)
print(f"\n Estatísticas:")
print(f"   Total de imagens: {len(annotations)}")
print(f"   Categorias de emoções: {len(emotion_counts)}")
print(f"   Emoção mais comum: {emotion_counts.most_common(1)[0][0]} ({emotion_counts.most_common(1)[0][1]} imagens)")
print(f"   Emoção mais rara: {emotion_counts.most_common()[-1][0]} ({emotion_counts.most_common()[-1][1]} imagens)")

# Calcular balanceamento
max_count = emotion_counts.most_common(1)[0][1]
min_count = emotion_counts.most_common()[-1][1]
imbalance_ratio = max_count / min_count

print(f"\n  Desbalanceamento:")
print(f"   Ratio (max/min): {imbalance_ratio:.1f}x")
if imbalance_ratio > 10:
    print(f"   Status:  MUITO DESBALANCEADO")
elif imbalance_ratio > 5:
    print(f"   Status:  Desbalanceado")
else:
    print(f"   Status:  Razoavelmente balanceado")

print("\n" + "="*70)
print(" VISUALIZAÇÃO CONCLUÍDA")
print("="*70)
print("\n Ficheiros gerados:")
print("   - results/emotion_distribution.png")
print("   - results/emotion_comparison_grid.png")
print("   - results/samples_[emotion].png (para cada emoção)")
print("\n Pode usar estas imagens na apresentação!")
