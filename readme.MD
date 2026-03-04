## Visão geral do projeto

Este repositório implementa um **classificador de sentimento em reviews de filmes IMDB** usando um **Transformer** em PyTorch.  
O pipeline cobre **carregamento de dados**, **tokenização simples**, **treino** com early stopping e **avaliação** com métricas e matriz de confusão.

### Organização atual do código

```text
.
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # Dataset IMDB + dataloaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer_sentiment.py  # Modelo TransformerSentiment
│   └── training/
│       ├── __init__.py
│       ├── loop.py                 # EarlyStopping, train_one_epoch, evaluate
│       └── plots.py                # plot_curves (loss/acc)
├── scripts/
│   ├── train.py                    # ponto de entrada de treino (CLI)
│   └── evaluate.py                 # ponto de entrada de avaliação (CLI)
├── main.py                         # utilitário opcional p/ salvar processed_train.pt
├── data/
│   └── aclImdb/                    # dados brutos (fora do git)
├── doc/
│   ├── DATASET_DECISION.md
│   └── DATASET_PROCESSING_STRATEGY.md
├── checkpoints/                    # gerado em tempo de execução
│   ├── best_model.pt
│   └── last_model.pt
├── training_log.csv                # log de treino (CSV)
├── training_curves.png             # curvas de treino/validação
└── README.md
```

- **`src/`**: código de biblioteca reutilizável (dados, modelos, treinamento).
- **`scripts/`**: scripts de linha de comando para treino e avaliação.
- **`main.py`**: script utilitário (não faz parte do fluxo normal; ver abaixo).

## Como instalar e rodar o projeto

### 1. Pré-requisitos

- Python 3.10+ (recomendado).
- `pip` ou `conda`.
- GPU CUDA opcional, mas suportada.

### 2. Clonar o repositório

```bash
git clone <url-do-repositorio.git>
cd transformers_rna
```

### 3. Criar ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Baixar e posicionar o dataset IMDB

O código assume a estrutura padrão do **aclImdb**:

```text
data/aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```

- As pastas `train/pos`, `train/neg`, `test/pos`, `test/neg` contêm arquivos `.txt` com uma review por arquivo.
- O caminho base usado por padrão é `data/aclImdb` (pode ser alterado com `--base-path`).

## Rodando o treino

O script principal de treino é `scripts/train.py`.  
Sempre execute a partir da raiz do projeto, usando o modo módulo:

### Com parâmetros padrão

```bash
python -m scripts.train
```

Isso irá:

- Carregar `train_samples=2000` positivos e 2000 negativos.
- Carregar `test_samples=500` positivos e 500 negativos.
- Construir o vocabulário a partir do treino.
- Treinar um Transformer com `d_model=256`, `num_layers=4`, `nhead=8`, etc.
- Aplicar **early stopping** com `patience=5`.
- Salvar:
  - `checkpoints/best_model.pt`
  - `checkpoints/last_model.pt`
  - `training_log.csv`
  - `training_curves.png`

### Com argumentos customizados

Exemplos:

```bash
# Mais épocas e batch maior
python -m scripts.train --epochs 20 --batch-size 64

# Ajustar tamanho da sequência e learning rate
python -m scripts.train --max-len 256 --lr 3e-4

# Mudar tamanho do dataset
python -m scripts.train --train-samples 5000 --test-samples 1000

# Forçar uso de CPU
python -m scripts.train --device cpu
```

### Retomando de um checkpoint

```bash
python -m scripts.train --checkpoint checkpoints/best_model.pt
```

O script vai:

- Carregar pesos, otimizador e scheduler.
- Continuar a partir da próxima época.

## Rodando a avaliação

Use `scripts/evaluate.py` para avaliar um checkpoint salvo:

```bash
python -m scripts.evaluate
```

Por padrão:

- Usa `checkpoints/best_model.pt`.
- Recria o vocabulário com os mesmos hiperparâmetros usados no treino.
- Calcula:
  - Acurácia.
  - Precisão, recall e F1 _macro_.
  - Matriz de confusão.
- Salva `confusion_matrix.png`.

### Opções úteis

```bash
# Avaliar outro checkpoint
python -m scripts.evaluate --checkpoint checkpoints/last_model.pt

# Usar um split de teste maior
python -m scripts.evaluate --test-samples 1000

# Usar batch maior
python -m scripts.evaluate --batch-size 128

# Salvar predições individuais
python -m scripts.evaluate --save-predictions predictions.csv
```

## `main.py` (opcional)

O arquivo `main.py` é um **utilitário** para gerar um tensor com todos os exemplos de treino:

```bash
python main.py
```

Ele:

- Carrega o `train_loader` via `src.data.dataset.get_dataloaders`.
- Empilha todos os `input_ids` e `labels`.
- Salva em `data/processed_train.pt`.

## Funcionamento das principais classes e fluxo

### `src/data/dataset.py`

- **Funções auxiliares**:
  - `load_reviews(path, label, max_samples)`:
    - Lê arquivos de texto de uma pasta (`pos` ou `neg`), embaralha e limita a `max_samples`.
  - `preprocess(text, max_len)`:
    - Converte para minúsculas, faz `split()` simples em espaço e trunca para `max_len` tokens.
  - `build_vocab(texts, min_freq=2)`:
    - Conta frequências e cria vocabulário a partir dos textos de treino.
    - Reserva:
      - `<pad>` (0) — padding.
      - `<unk>` (1) — token desconhecido.
  - `encode(tokens, vocab, max_len)`:
    - Converte tokens em índices (`vocab[token]` ou `<unk>`).
    - Faz padding até `max_len` com `<pad>`.

- **Classe `IMDBDataset`**:
  - Implementa `__len__` e `__getitem__`.
  - `__getitem__`:
    - Aplica `preprocess` no texto.
    - Codifica usando `encode`.
    - Retorna `(input_ids, label)` como tensores `torch.long`.

- **Função `get_dataloaders(...)`**:
  - Carrega dados balanceados:
    - `train_pos`, `train_neg`, `test_pos`, `test_neg`.
  - Embaralha e concatena.
  - Constrói vocabulário **apenas com o treino**.
  - Cria `IMDBDataset` de treino e teste.
  - Cria `DataLoader`s de treino (com `shuffle=True`) e teste.
  - Retorna `(train_loader, test_loader, vocab_size)`.

### `src/models/transformer_sentiment.py`

- **`SinusoidalPositionalEncoding`**:
  - Implementa a codificação posicional clássica (Vaswani et al., 2017):
    - Para cada posição \(pos\) e dimensão \(i\):
      - \(\text{PE}_{pos,2i}   = \sin(pos / 10000^{2i/d\_model})\)
      - \(\text{PE}_{pos,2i+1} = \cos(pos / 10000^{2i/d\_model})\)
  - Pré-calcula uma matriz `[max_len, d_model]` e registra como `buffer` (não treinável).
  - No `forward`, soma essa matriz às embeddings (`[B, L, D]`) e aplica `Dropout`.

- **`TransformerSentiment`**:
  - Componentes:
    - `nn.Embedding(vocab_size, d_model, padding_idx=pad_id)` — embedding de tokens.
    - `SinusoidalPositionalEncoding(d_model, max_len, dropout)` — codificação posicional.
    - `nn.TransformerEncoder` com:
      - `num_layers` camadas.
      - `d_model`, `nhead`, `dim_feedforward`, `dropout`.
      - `batch_first=True`, `activation="gelu"`, `norm_first=True`.
    - `nn.Linear(d_model, num_classes)` — classificador final.
  - Fluxo no `forward(input_ids)`:
    1. Cria `pad_mask = (input_ids == pad_id)` — True onde é padding.
    2. Calcula `x = embedding(input_ids)` → `[B, L, D]`.
    3. Adiciona codificação posicional: `x = positional(x)`.
    4. Passa pelo encoder Transformer:
       - `x = encoder(x, src_key_padding_mask=pad_mask)` → `[B, L, D]`.
    5. Faz **mean pooling mascarado**:
       - Zera posições de padding.
       - Soma ao longo de `L` e divide pela contagem de tokens válidos.
    6. Aplica `classifier` para obter `logits` `[B, num_classes]`.

### `src/training/loop.py` e `scripts/train.py`

- **Parsing de argumentos (`parse_args`)**:
  - Hiperparâmetros de dados:
    - `--base-path`, `--max-len`, `--train-samples`, `--test-samples`.
  - Hiperparâmetros de modelo:
    - `--d-model`, `--nhead`, `--num-layers`, `--dim-feedforward`, `--dropout`, `--num-classes`.
  - Hiperparâmetros de treino:
    - `--epochs`, `--patience`, `--batch-size`, `--lr`, `--weight-decay`, `--grad-clip`.
  - Misc:
    - `--seed`, `--checkpoint`, `--checkpoint-dir`, `--log-file`, `--plot-file`, `--device`.

- **Early stopping (`EarlyStopping`)**:
  - Monitora `val_acc`.
  - Se não houver melhora maior que `min_delta` por `patience` épocas consecutivas, marca `triggered=True` e o loop principal interrompe.

- **Função `train_one_epoch`**:
  - Modo `model.train()`.
  - Loop sobre batches de `(input_ids, labels)`:
    - Envia para `device`.
    - `logits = model(input_ids)`.
    - `loss = CrossEntropyLoss(logits, labels)`.
    - `backward`, `clip_grad_norm_` (opcional), `optimizer.step`.
  - Calcula acurácia em tempo real.

- **Função `evaluate`**:
  - Modo `model.eval()` com `@torch.no_grad()`.
  - Mesmo padrão de loop, mas sem `backward`.

- **Funções de checkpoint**:
  - `save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args)`:
    - Serializa pesos do modelo, estados do otimizador/scheduler e alguns metadados.
  - `load_checkpoint(path, model, optimizer, scheduler)`:
    - Restaura esses estados e retorna `(epoch, best_val_acc)`.

- **Plot de curvas (`plot_curves`)**:
  - Recebe uma lista de dicionários `history` com métricas por época.
  - Gera um gráfico 2x1 com loss e acurácia de treino/validação.
  - Marca a época de early stopping, se houver.

- **Fluxo do `main()`**:
  1. Lê argumentos.
  2. Define semente e dispositivo (`cpu`/`cuda`/`mps`).
  3. Cria dataloaders via `get_dataloaders`.
  4. Instancia `TransformerSentiment`.
  5. Cria `CrossEntropyLoss`, `AdamW`, `CosineAnnealingLR`.
  6. (Opcional) retoma de checkpoint.
  7. Abre `training_log.csv` e prepara `DictWriter`.
  8. Loop de épocas:
     - `train_one_epoch` + `evaluate`.
     - Atualiza scheduler.
     - Salva `best_model.pt` e `last_model.pt`.
     - Registra métricas no CSV e em memória.
     - Checa early stopping.
  9. Finaliza, imprime resumo e gera `training_curves.png`.

### `scripts/evaluate.py`

- **Fluxo principal (`main`)**:
  1. Lê argumentos (`--checkpoint`, `--base-path`, `--test-samples`, `--batch-size`, `--device`, `--save-predictions`, `--confusion-plot`).
  2. Carrega o checkpoint (`torch.load`) e recupera hiperparâmetros usados no treino.
  3. Reconstrói o vocabulário chamando `get_dataloaders` com os mesmos parâmetros de treino (mesmo `seed` e `train_samples`).
  4. Reconstrói o modelo `TransformerSentiment` com os hiperparâmetros recuperados.
  5. Carrega os pesos do modelo.
  6. Executa `run_inference` no conjunto de teste:
     - Retorna listas de rótulos verdadeiros, predições e confidências (máx softmax).
  7. Calcula métricas com `compute_metrics` (sem depender de `sklearn`).
  8. Plota matriz de confusão com `plot_confusion_matrix`.
  9. (Opcional) salva um CSV `predictions.csv` com `true_label`, `pred_label`, `confidence`, `correct`, etc.

## Fluxo de dados e workflow (resumo)

1. **Dados brutos**:
   - Arquivos `.txt` em `data/aclImdb/train/{pos,neg}` e `data/aclImdb/test/{pos,neg}`.
2. **Pré-processamento**:
   - `get_dataloaders`:
     - Lê os arquivos, aplica `preprocess` (lowercase + split + truncamento).
     - Constrói o vocabulário a partir do treino.
     - Codifica tokens em índices com padding até `max_len`.
     - Cria `IMDBDataset` e `DataLoader`s.
3. **Treino**:
   - `scripts/train.py`:
     - Instancia `TransformerSentiment`.
     - Para cada época:
       - Loop de batches (forward → loss → backward → step).
       - Avaliação no teste.
       - Early stopping, checkpoints, logging e plots.
4. **Avaliação**:
   - `scripts/evaluate.py`:
     - Restaura hiperparâmetros e pesos do melhor modelo.
     - Recria dataloaders consistentes com o treino.
     - Calcula métricas, matriz de confusão e (opcionalmente) salva predições.

Com essa visão, fica mais fácil **evoluir** o projeto (por exemplo, adicionar novos modelos em `src/models/` ou novos datasets em `src/data/`) sem quebrar o fluxo atual de treino e avaliação.

