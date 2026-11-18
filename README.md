# De ETL a IA: Pipeline Reproducible de Datos para Machine Learning

Proyecto práctico de acompañamiento al artículo técnico de Kranio: **"De ETL a IA: Cómo Transformar Datos para Machine Learning"**.

Demuestra cómo llevar datos transaccionales crudos a través de una arquitectura modular (**Bronze $\rightarrow$ Silver $\rightarrow$ Gold ML-Ready**) para predecir el abandono de clientes (*Customer Churn*), aplicando **Contratos de Datos**, **Feature Engineering RFM** sin *Data Leakage* y un **Pipeline unificado de `scikit-learn`**.

---

## Arquitectura del Proyecto

```text
etl-to-ml-pipeline/
├── run_pipeline.py              # Orquestador End-to-End (Bronze -> Silver -> Gold -> ML)
├── requirements.txt             # Dependencias Python
├── src/
│   ├── data_generator.py        # [Bronze] Generador sintético con duplicados, nulos y eventos futuros
│   ├── etl_cleaning.py          # [Silver] Contratos de datos, deduplicación y normalización UTC
│   ├── feature_engineering.py   # [Gold] Cálculo de variables RFM respetando cutoff_date (cero Leakage)
│   └── train_model.py           # [ML] Train/Test split previo + ColumnTransformer + RandomForest
└── tests/
    └── test_pipeline.py         # Pruebas unitarias (Data Contracts, Leakage y Pipeline)
```

### Principios Técnicos Implementados
1. **Contratos de Datos (*Fail-Fast*):** Validación estricta de esquema antes de procesar (`src/etl_cleaning.py`).
2. **Prevención de *Data Leakage* Temporal:** Las métricas RFM (*Recency, Frequency, Monetary*) se calculan exclusivamente sobre eventos `<= cutoff_date` (`src/feature_engineering.py`).
3. **Prevención de *Training-Serving Skew*:** El imputador, el escalador (`StandardScaler`) y el codificador (`OneHotEncoder`) viven dentro del mismo `Pipeline` serializable del modelo (`src/train_model.py`).

---

## Ejecución Rápida (Local)

```bash
# 1. Clonar el repositorio
git clone https://github.com/richardKranio/etl-to-ml-pipeline.git
cd etl-to-ml-pipeline

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

# 3. Ejecutar el pipeline completo (Bronze -> Silver -> Gold -> Entrenamiento)
python run_pipeline.py

# 4. Ejecutar pruebas unitarias
pytest -v
```

---

## Salidas Generadas
Al ejecutar `python run_pipeline.py`, el sistema genera:
* `data/bronze/*.csv` — Datos crudos simulados (clientes y transacciones con duplicados e inactividad).
* `data/silver/*.parquet` — Tablas limpias, deduplicadas y normalizadas en UTC.
* `data/gold/churn_features_gold.parquet` — Tabla analítica *ML-Ready* con features RFM.
* `artifacts/churn_pipeline.pkl` — Artefacto serializado con preprocesamiento + modelo entrenado.
