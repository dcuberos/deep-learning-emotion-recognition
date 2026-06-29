"""
Carrega o dataset EMOTIC diretamente dos ficheiros .npy (sem conversão para JPG).

Gera data/emotic/annotations.json com, para cada crop:
  - image_path: caminho completo do .npy (array RGB 224x224x3, uint8)
  - labels:     vetor multi-hot de dimensão 26 (ordem de config.EMOTIONS)
  - emotions:   lista legível das emoções presentes
  - split:      'train' | 'val' | 'test'  (splits oficiais do EMOTIC)

NOTA: o EMOTIC é multi-etiqueta. Mantemos TODAS as emoções de cada pessoa
(não apenas a primeira) e descartamos crops sem qualquer emoção anotada
(em vez de inventar uma etiqueta aleatória).
"""

import os
import json
import numpy as np
import pandas as pd

from config import EMOTIONS

print("=" * 70)
print(" CARREGAMENTO DIRETO DO EMOTIC (MULTI-ETIQUETA, SEM CONVERSÃO)")
print("=" * 70)

current_dir = os.path.dirname(os.path.abspath(__file__))
EMOTIC_ROOT = os.path.join(current_dir, "data", "emotic")
IMG_ARRS_DIR = os.path.join(EMOTIC_ROOT, "img_arrs")

print(f" IMG_ARRS_DIR: {IMG_ARRS_DIR}")
print(f" IMG_ARRS existe? {os.path.exists(IMG_ARRS_DIR)}")


def process_split(csv_path, split_name):
    """Processa um split (train/val/test) e devolve a lista de anotações."""
    if not os.path.exists(csv_path):
        print(f" Não encontrado: {csv_path}")
        return []

    print(f"\n Processando {split_name}...")
    df = pd.read_csv(csv_path)
    print(f"  Total no CSV: {len(df)} amostras")

    # Colunas de emoção realmente presentes no CSV
    present_cols = [c for c in EMOTIONS if c in df.columns]
    missing_cols = [c for c in EMOTIONS if c not in df.columns]
    if missing_cols:
        print(f"   Colunas de emoção em falta no CSV: {missing_cols}")

    annotations = []
    found = missing = no_label = 0

    for _, row in df.iterrows():
        crop_name = str(row.get('Crop_name', ''))
        npy_path = os.path.join(IMG_ARRS_DIR, crop_name)

        if not os.path.exists(npy_path):
            missing += 1
            continue

        # Vetor multi-hot na ordem de EMOTIONS
        labels = [int(float(row[c])) if c in present_cols and not pd.isna(row[c]) else 0
                  for c in EMOTIONS]

        if sum(labels) == 0:
            # Sem qualquer emoção anotada -> descartar (não inventar)
            no_label += 1
            continue

        emotions_present = [emo for emo, v in zip(EMOTIONS, labels) if v == 1]

        annotations.append({
            'image_path': npy_path,
            'labels': labels,
            'emotions': emotions_present,
            'split': split_name,
        })
        found += 1

    print(f"   Válidos: {found}  |  Ficheiro .npy em falta: {missing}  |  Sem emoção: {no_label}")
    return annotations


csv_files = [
    (os.path.join(EMOTIC_ROOT, "annot_arrs_train.csv"), "train"),
    (os.path.join(EMOTIC_ROOT, "annot_arrs_val.csv"), "val"),
    (os.path.join(EMOTIC_ROOT, "annot_arrs_test.csv"), "test"),
]

all_annotations = []
for csv_path, split_name in csv_files:
    all_annotations.extend(process_split(csv_path, split_name))

if all_annotations:
    output_json = os.path.join(EMOTIC_ROOT, "annotations.json")
    with open(output_json, 'w') as f:
        json.dump(all_annotations, f)

    print(f"\n{'=' * 70}")
    print(" CARREGAMENTO CONCLUÍDO")
    print(f"{'=' * 70}")
    print(f" Total de anotações: {len(all_annotations)}")
    print(f" Ficheiro criado: {output_json}")

    # Estatística de frequência por emoção (contagem multi-hot)
    counts = np.sum([a['labels'] for a in all_annotations], axis=0)
    order = np.argsort(counts)[::-1]
    print("\n Frequência por emoção (top 10):")
    for i in order[:10]:
        print(f"  {EMOTIONS[i]}: {int(counts[i])}")

    # Distribuição por split
    for s in ('train', 'val', 'test'):
        n = sum(1 for a in all_annotations if a['split'] == s)
        print(f"  Split {s}: {n}")

    print("\n Agora pode executar: python train.py")
else:
    print("\n Nenhuma anotação processada!")
