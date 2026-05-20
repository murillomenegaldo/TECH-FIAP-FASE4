# Tech Challenge Fase 4 — LSTM Stock Price Prediction

Projeto desenvolvido para o Tech Challenge da Fase 4 do curso de Machine Learning Engineering da Pos Tech FIAP.

O objetivo foi construir um modelo de redes neurais LSTM capaz de prever o preço de fechamento de uma ação, e entregar esse modelo através de uma API REST que pode ser rodada localmente ou em nuvem via Docker.

A ação escolhida foi a **Apple (AAPL)**, usando dados históricos de 2018 a 2024 coletados via Yahoo Finance.

---

## Resultado do treinamento

O modelo foi treinado com uma janela de 60 dias de histórico (Open, High, Low, Close, Volume) para prever o fechamento do dia seguinte. Os resultados no conjunto de teste foram:

- MAE: 12.98
- RMSE: 14.51
- MAPE: 5.81%
- R²: 0.40

O R² relativamente baixo é esperado em séries financeiras — o modelo captura a tendência geral, mas não consegue prever movimentos bruscos causados por eventos externos (resultados trimestrais, notícias, etc). Para uso real seria necessário adicionar features externas e retreinar periodicamente.

---

## Arquitetura do modelo

Rede LSTM empilhada com duas camadas, dropout para regularização e função de loss Huber (mais robusta a outliers do que MSE):

```
Input (60 dias × 5 features)
  → LSTM(128, return_sequences=True)
  → Dropout(0.2)
  → LSTM(64)
  → Dropout(0.2)
  → Dense(32, relu)
  → Dense(1)
```

Treinado com Adam (lr=0.001), EarlyStopping com patience=15 e ReduceLROnPlateau. Split de 80/10/10 para treino, validação e teste.

---

## Estrutura do projeto

```
lstm-stock-api/
├── app/
│   └── main.py          # API FastAPI com endpoints de predição e monitoramento
├── model/
│   ├── train.py         # Script de coleta, pré-processamento e treinamento
│   └── artifacts/       # Modelo treinado, scaler e metadados (gerados pelo train.py)
├── monitoring/
│   └── monitor.py       # Script de monitoramento via terminal
├── tests/
│   └── test_api.py      # Testes dos endpoints com pytest
├── Dockerfile
├── docker-compose.yml
├── render.yaml          # Configuração para deploy no Render.com
└── requirements.txt
```

---

## Como rodar

Requisito: Python 3.11. No Mac com Apple Silicon usar `/opt/homebrew/bin/python3.11`.

**1. Instalar dependências**
```bash
pip install -r requirements.txt
```

**2. Treinar o modelo**
```bash
python model/train.py
```
Isso baixa os dados da AAPL via yfinance, treina o modelo e salva os artefatos em `model/artifacts/`. Demora entre 5 e 15 minutos dependendo da máquina.

Para usar outra ação, muda o `CONFIG["symbol"]` no início do `train.py`.

**3. Subir a API**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse `http://localhost:8000/docs` para testar pelo Swagger.

**4. Com Docker**
```bash
docker compose up --build
```

---

## Endpoints

**POST /predict**

Recebe o ticker e quantos dias prever, busca os dados mais recentes via yfinance e retorna as previsões:

```json
{
  "symbol": "AAPL",
  "days_ahead": 5,
  "use_latest": true
}
```

Resposta:
```json
{
  "symbol": "AAPL",
  "predictions": [
    { "date": "2026-05-21", "predicted_close": 258.59 },
    { "date": "2026-05-22", "predicted_close": 259.06 }
  ],
  "model_metrics": { "MAE": 12.98, "RMSE": 14.51, "MAPE": 5.81, "R2": 0.40 },
  "generated_at": "2026-05-20T23:32:13"
}
```

Também aceita preços manuais via campo `prices` (útil quando sem internet), passando pelo menos 60 valores de fechamento.

**GET /health** — verifica se o modelo está carregado e retorna total de requisições

**GET /metrics** — latência média, P95, P99, total de predições e últimas requisições

**GET /model/info** — metadados do modelo carregado (symbol, features, sequence_length)

**POST /model/reload** — recarrega os artefatos sem reiniciar a API

---

## Monitoramento

```bash
python monitoring/monitor.py --url http://localhost:8000 --interval 5
```

Exibe no terminal as métricas de latência e qualidade do modelo, atualizadas a cada 5 segundos. Os logs ficam salvos em `monitoring/api.log`.

---

## Testes

```bash
pytest tests/ -v
```

Cobre validação de schema, comportamento sem modelo carregado, presença do header de timing e respostas dos endpoints de saúde e métricas.

---

## Deploy

O arquivo `render.yaml` configura o deploy no Render.com. Basta conectar o repositório e criar um novo Blueprint. Os artefatos do modelo já estão commitados, então a API sobe direto sem precisar retreinar.

---

Pos Tech FIAP — Machine Learning Engineering — Fase 4
