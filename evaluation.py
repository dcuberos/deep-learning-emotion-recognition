"""
Avaliação multi-etiqueta de modelos EMOTIC.

Métricas honestas para o problema:
  - mAP (mean Average Precision) macro e micro -> métrica de referência no EMOTIC.
  - AP por classe -> mostra quais emoções o modelo realmente aprende.
  - F1 / Precision / Recall (macro e micro) a um limiar configurável.

NOTA: "accuracy" e matriz de confusão multiclasse não se aplicam a um
problema multi-etiqueta e por isso não são usadas aqui.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    classification_report,
)

from config import CONFIG, EMOTIONS
from data_loader import EmotionDataGenerator


class ModelEvaluator:
    """Avalia um modelo multi-etiqueta treinado."""

    def __init__(self, model, class_names=None, threshold=None):
        self.model = model
        self.class_names = list(class_names) if class_names is not None else list(EMOTIONS)
        self.threshold = threshold if threshold is not None else CONFIG['classification_threshold']

    def predict(self, X_paths, y_true, batch_size=32):
        gen = EmotionDataGenerator(
            X_paths, y_true, batch_size=batch_size,
            img_size=(CONFIG['img_height'], CONFIG['img_width']),
            backbone=CONFIG['backbone'], augment=False, shuffle=False,
        )
        y_probs = self.model.predict(gen, verbose=1)
        return y_probs

    def evaluate(self, X_paths, y_true, batch_size=32, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        y_true = np.asarray(y_true, dtype=np.float32)
        y_probs = self.predict(X_paths, y_true, batch_size)
        y_pred = (y_probs >= self.threshold).astype(int)

        # mAP (não depende de limiar)
        map_macro = average_precision_score(y_true, y_probs, average='macro')
        map_micro = average_precision_score(y_true, y_probs, average='micro')

        # Métricas a limiar fixo
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

        print("\n" + "=" * 60)
        print(" RESULTADOS (multi-etiqueta)")
        print("=" * 60)
        print(f"  mAP (macro):   {map_macro:.4f}")
        print(f"  mAP (micro):   {map_micro:.4f}")
        print(f"  F1  (macro):   {f1_macro:.4f}")
        print(f"  F1  (micro):   {f1_micro:.4f}")
        print(f"  Precision (macro): {prec_macro:.4f}")
        print(f"  Recall    (macro): {rec_macro:.4f}")

        # AP por classe
        ap_per_class = average_precision_score(y_true, y_probs, average=None)
        self._save_per_class_report(y_true, y_pred, ap_per_class, save_dir)
        self._plot_ap_per_class(ap_per_class, save_dir)

        return {
            'map_macro': map_macro, 'map_micro': map_micro,
            'f1_macro': f1_macro, 'f1_micro': f1_micro,
            'precision_macro': prec_macro, 'recall_macro': rec_macro,
            'ap_per_class': ap_per_class,
            'y_true': y_true, 'y_pred': y_pred, 'y_probs': y_probs,
        }

    def _save_per_class_report(self, y_true, y_pred, ap_per_class, save_dir):
        report = classification_report(
            y_true, y_pred, target_names=self.class_names,
            digits=4, zero_division=0,
        )
        ap_df = pd.DataFrame({
            'emotion': self.class_names,
            'support': y_true.sum(axis=0).astype(int),
            'average_precision': ap_per_class,
        }).sort_values('average_precision', ascending=False)

        path = os.path.join(save_dir, 'classification_report.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE CLASSIFICAÇÃO MULTI-ETIQUETA\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Limiar: {self.threshold}\n\n")
            f.write(report + "\n\n")
            f.write("Average Precision por emoção (ordenado):\n")
            f.write(ap_df.to_string(index=False))
        print(f" Relatório salvo em {path}")
        ap_df.to_csv(os.path.join(save_dir, 'ap_per_class.csv'), index=False)

    def _plot_ap_per_class(self, ap_per_class, save_dir):
        order = np.argsort(ap_per_class)
        names = [self.class_names[i] for i in order]
        plt.figure(figsize=(10, 9))
        plt.barh(names, ap_per_class[order], color='steelblue')
        plt.xlabel('Average Precision')
        plt.title('AP por emoção')
        plt.tight_layout()
        path = os.path.join(save_dir, 'ap_per_class.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f" Gráfico AP por classe salvo em {path}")


if __name__ == "__main__":
    print(" Módulo de avaliação multi-etiqueta carregado.")
