# 📈 LSTM Stock Price Prediction API

> **Tech Challenge Fase 4 — Pos Tech Machine Learning Engineering**  
> Modelo preditivo de série temporal com LSTM + API RESTful + Docker + Monitoramento

---

## 🗂 Estrutura do Projeto

```
lstm-stock-api/
├── app/
│   └── main.py              # FastAPI — endpoints de predição e monitoramento
├── model/
│   ├── train.py             # Pipeline de treinamento LSTM completa
│   └── artifacts/           # Gerado após treino: modelo, scaler, métricas
│       ├── lstm_best.keras
│       ├── scaler.pkl
│       ├── meta.json
│       ├── metrics.json
│       └── results.png
├── monitoring/
│   ├── monitor.py           # Dashboard de monitoramento em tempo real
│   └── api.log              # Log persistido (gerado em runtime)
├── tests/
│   └── test_api.py          # Testes automáticos com pytest
├── Dockerfile               # Multi-stage build otimizado
├── docker-compose.yml       # Orquestração API + treino
├── render.yaml              # Deploy gratuito no Render.com
├── requirements.txt
└── README.md
```

---

## 🧠 Arquitetura do Modelo

| Componente | Detalhe |
|---|---|
| Ação | **PETR4.SA** (Petrobras) — customizável |
| Features | Open, High, Low, Close, Volume (5 features) |
| Janela temporal | **60 dias** de histórico para cada predição |
| Arquitetura | LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1) |
| Loss | Huber (robusto a outliers) |
| Otimizador | Adam (lr=0.001) com ReduceLROnPlateau |
| Callbacks | EarlyStopping (patience=15) + ModelCheckpoint |
| Split | 80% treino / 10% validação / 10% teste |

---

## 🚀 Como Rodar

### Pré-requisitos
- Python 3.11+
- Docker & Docker Compose (opcional)

---

### 1. Instalar dependências

```bash
git clone https://github.com/seu-usuario/lstm-stock-api.git
cd lstm-stock-api
pip install -r requirements.txt
```

---

### 2. Treinar o modelo

```bash
python model/train.py
```

O script vai:
1. Baixar dados históricos da **PETR4.SA** via `yfinance`
2. Normalizar com `MinMaxScaler` e criar janelas de 60 dias
3. Treinar a rede LSTM com early stopping
4. Salvar os artefatos em `model/artifacts/`
5. Imprimir as métricas (MAE, RMSE, MAPE, R²)
6. Gerar gráfico `model/artifacts/results.png`

> Para trocar a ação, edite `CONFIG["symbol"]` em `model/train.py`.

---

### 3. Rodar a API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse:
- **Swagger UI**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Métricas**: http://localhost:8000/metrics

---

### 4. Rodar com Docker

```bash
# Build e start da API
docker compose up --build

# Treinar dentro do Docker (antes de subir a API)
docker compose --profile train up trainer
```

---

## 📡 Endpoints da API

### `GET /health`
Verifica se a API e o modelo estão carregados.

```json
{
  "status": "ok",
  "model_loaded": true,
  "symbol": "PETR4.SA",
  "uptime_requests": 42
}
```

---

### `POST /predict`
Gera previsões de preço de fechamento.

**Request body:**
```json
{
  "symbol": "PETR4.SA",
  "days_ahead": 5,
  "use_latest": true
}
```

**Response:**
```json
{
  "symbol": "PETR4.SA",
  "predictions": [
    { "date": "2025-01-20", "predicted_close": 37.82 },
    { "date": "2025-01-21", "predicted_close": 38.10 },
    ...
  ],
  "model_metrics": {
    "MAE": 0.4231,
    "RMSE": 0.6102,
    "MAPE": 1.38,
    "R2": 0.9741
  },
  "generated_at": "2025-01-19T14:32:00"
}
```

**Com preços manuais (sem internet):**
```json
{
  "use_latest": false,
  "days_ahead": 3,
  "prices": [35.1, 35.4, 35.8, ...]
}
```

---

### `GET /metrics`
Monitoramento da API em tempo real.

```json
{
  "total_requests": 150,
  "successful_predictions": 143,
  "failed_requests": 7,
  "response_time_ms": {
    "avg": 234.5,
    "p95": 450.2,
    "p99": 890.1,
    "min": 45.3,
    "max": 1200.0
  },
  "model_metrics": { "MAE": 0.42, "RMSE": 0.61, "MAPE": 1.38, "R2": 0.97 }
}
```

---

### `GET /model/info`
Metadados do modelo carregado.

### `POST /model/reload`
Recarrega os artefatos do modelo sem reiniciar a API (útil após retreino).

---

## 📊 Monitoramento em Tempo Real

```bash
python monitoring/monitor.py --url http://localhost:8000 --interval 5
```

Exibe um dashboard no terminal atualizado a cada 5 segundos com:
- Total de requisições / predições / erros
- Latência média, P95, P99
- Métricas de qualidade do modelo
- Última predição realizada

Os logs também são persistidos em `monitoring/api.log`.

---

## 🧪 Testes

```bash
pytest tests/ -v
```

Testes cobrem:
- Endpoints de saúde e métricas
- Validação de schema do request (`days_ahead` fora do range, etc.)
- Comportamento sem modelo carregado
- Presença do header de timing

---

## ☁️ Deploy em Nuvem (Gratuito)

### Render.com

1. Crie uma conta em [render.com](https://render.com)
2. Conecte seu repositório GitHub
3. Clique em **New → Blueprint** e selecione o repositório
4. O `render.yaml` configura tudo automaticamente

> ⚠️ Treine o modelo localmente e faça commit dos artefatos em `model/artifacts/` antes do deploy, ou configure o treino como parte do build.

### Railway / Hugging Face Spaces

Alternativas gratuitas — use o `Dockerfile` incluso. Basta conectar o repositório.

---

## 🔧 Customização

| Parâmetro | Onde alterar | Default |
|---|---|---|
| Ação analisada | `CONFIG["symbol"]` em `train.py` | `PETR4.SA` |
| Janela temporal | `CONFIG["sequence_length"]` | `60` |
| Neurônios LSTM | `CONFIG["lstm_units"]` | `[128, 64]` |
| Dropout | `CONFIG["dropout_rate"]` | `0.2` |
| Épocas máximas | `CONFIG["epochs"]` | `100` |
| Split treino | `CONFIG["train_ratio"]` | `0.8` |

---

## 📦 Tecnologias

| Área | Tecnologia |
|---|---|
| Deep Learning | TensorFlow / Keras 2.16 |
| Dados | yfinance, pandas, numpy |
| API | FastAPI + Uvicorn |
| Containerização | Docker + Docker Compose |
| Testes | pytest + httpx |
| Deploy | Render.com / Railway |

---

## 📄 Licença

Projeto acadêmico — Tech Challenge Fase 4, Pos Tech FIAP.
