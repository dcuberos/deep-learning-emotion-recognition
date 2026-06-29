# Reconhecimento de Emocoes atraves de Posturas Corporais

Projeto de Aprendizagem Profunda - Universidade Catolica Portuguesa

**Autor:** Daniel Rodrigues  
**Data:** Janeiro 2026

## Descricao

Sistema de classificacao de emocoes baseado exclusivamente em posturas corporais, removendo deliberadamente informacao facial para focar em sinais nao-verbais do corpo. Utiliza o dataset EMOTIC com 26 categorias emocionais e arquitetura ResNet50 com transfer learning.

## Dataset

- **EMOTIC** (EMOTions In Context)
- 23,788 anotacoes em 18,316 imagens
- 26 categorias emocionais
- Desbalanceamento significativo (Engagement: 43.9%, Excitement: 1.5%)

## Metodologia

### Pre-processamento
- Remocao automatica de rostos (MediaPipe Face Detection + Gaussian Blur)
- Extracao de bounding boxes individuais
- Normalizacao para 224x224 pixels

### Formulacao do problema
O EMOTIC e **multi-etiqueta**: cada pessoa pode exibir varias das 26 emocoes
em simultaneo. Por isso o modelo usa ativacao **sigmoid** (26 saidas
independentes) e **Binary Focal Crossentropy** (com `apply_class_balancing`),
e nao softmax + categorical crossentropy.

### Modelo
- **Arquitetura:** ResNet50 com Transfer Learning (configuravel em `config.py`:
  `resnet50` | `efficientnet` | `custom`)
- **Pre-processamento:** `resnet50.preprocess_input` (o esperado pelos pesos
  ImageNet), nao um simples `/255`
- **Treino em 2 fases:**
  - Fase 1 — cabeca (base congelada), Adam lr=1e-3, 15 epocas
  - Fase 2 — fine-tuning do topo da base, Adam lr=1e-5, 25 epocas
- **Loss:** Binary Focal Crossentropy (mitiga o desbalanceamento severo)
- **Splits:** os splits oficiais train/val/test do EMOTIC sao respeitados
- **Batch size:** 32
- **Metricas:** AUC (multi-label), Precision, Recall durante o treino;
  mAP, F1 (macro/micro) e AP por classe na avaliacao

## Estrutura do Projeto

```
.
├── config.py                  # Configuracoes globais e lista de emocoes (multi-hot)
├── load_emotic_direct.py     # Gera annotations.json (multi-etiqueta, splits oficiais)
├── data_loader.py            # Geradores de dados (.npy + preprocess por backbone)
├── cnn_models.py             # Arquiteturas, focal loss e fine-tuning em 2 fases
├── face_removal.py           # Remocao de rostos (MediaPipe/OpenCV)
├── evaluation.py             # Metricas multi-etiqueta (mAP, F1, AP por classe)
├── train.py                  # Script principal de treino (ponto de entrada)
├── visualize_emotic.py       # Visualizacao do dataset
└── requirements.txt          # Dependencias
```

## Instalacao

```bash
# Instalar dependencias
pip install -r requirements.txt

# Carregar dataset EMOTIC
python load_emotic_direct.py

# Treinar modelo
python train.py
```

## Requisitos

- Python 3.8+
- TensorFlow 2.18.0
- OpenCV 4.8.0+
- MediaPipe 0.10.0+
- NumPy, Pandas, Scikit-learn

## Resultados

> **Nota:** os resultados anteriores (Accuracy 30.42%, F1-macro 0.0179) foram
> obtidos com um pipeline que continha bugs criticos — em particular, as
> imagens `.npy` eram lidas com `cv2.imread`, que nao le esse formato, pelo
> que **o modelo treinava com imagens completamente pretas** e colapsava na
> classe maioritaria (Engagement). Esses valores foram removidos por nao
> serem representativos. Apos as correcoes, reexecutar `python train.py`
> para obter as novas metricas (mAP, F1 macro/micro, AP por classe), guardadas
> em `results/`.

### Correcoes aplicadas (face as causas da baixa performance)
1. **Imagens pretas** — `data_loader` passa a ler `.npy` com `np.load`
2. **Etiquetas aleatorias** — crops sem emocao sao descartados (nao inventados)
3. **Multi-etiqueta** — sigmoid + Binary Focal Crossentropy em vez de softmax
4. **Pre-processamento** — `preprocess_input` da backbone em vez de `/255`
5. **Desbalanceamento** — focal loss com balanceamento de classes
6. **Treino** — 2 fases (cabeca + fine-tuning), mais epocas, splits oficiais
7. **Metricas honestas** — mAP e AP por classe em vez de accuracy global

## Limitacoes e Trabalho Futuro

### Trabalho futuro
1. **Remocao facial no pipeline `.npy`** — `face_removal.py` so e aplicado ao
   caminho antigo de crops JPG; falta aplica-lo (em cache) aos arrays `.npy`
2. Usar **keypoints de pose** (MediaPipe Pose) em vez da imagem crua
3. Incorporar informacao **temporal** (video)
4. Testar noutros datasets (BodyTalk, HAPPEI)
5. Ajustar o **limiar** de decisao por classe (em vez de 0.5 global)

## Implicacoes Eticas

- **Privacidade:** Remocao facial reduz identificacao biometrica
- **Consentimento:** Necessario para qualquer aplicacao real
- **Supervisao humana:** Obrigatoria dado baixa accuracy
- **Uso apropriado:** Nao adequado para decisoes criticas

## Ficheiros Gerados

Apos execucao do treino:
- `models/emotion_<backbone>_final.keras` - Modelo treinado (formato nativo Keras)
- `models/emotions.json` - Ordem das 26 emocoes (alinhada com os vetores multi-hot)
- `results/training_history.png` - Curvas de treino (loss/AUC/precision/recall)
- `results/ap_per_class.png` - Average Precision por emocao
- `results/classification_report.txt` - Relatorio multi-etiqueta detalhado
- `results/ap_per_class.csv` - AP e suporte por emocao

## Referencia

Para mais detalhes, consultar o relatorio completo:
`Relatorio_Final_Emocoes_Posturais.docx`

## Licenca

Projeto academico - Universidade Catolica Portuguesa
