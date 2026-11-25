# BLEVE Overpressure Prediction

Machine learning pipeline that predicts the **peak blast overpressure (bar)** recorded by a sensor after a BLEVE (boiling liquid expanding vapour explosion), from tank, liquid, obstacle and sensor-position data.

A BLEVE happens when a pressurised tank holding superheated liquid fails. Estimating the pressure wave at different positions and behind obstacles is useful for safety distances and barrier design, and it's expensive to simulate, which makes it a good fit for a learned surrogate model.

## Results
| Metric (validation) | Value |
|---|---|
| MAPE | 0.117 |
| R² | 0.950 |
| RMSE | 0.123 bar |

Held-out leaderboard score (MAPE-based): **0.208**.

## Approach
1. **Cleaning** – fix missing values and inconsistent categorical spellings.
2. **Imputation** – KNN imputation (k = 5) for numeric gaps, fitted on training data only to avoid leakage.
3. **Feature engineering** – derived geometric and thermodynamic features.
4. **Models** – ElasticNet, NuSVR, XGBoost, CatBoost and LightGBM, each in a scikit-learn pipeline with hyperparameter tuning.
5. **Ensemble** – weighted blend of XGBoost (0.80), CatBoost (0.15) and LightGBM (0.05).
6. **Diagnostics** – prediction-distribution checks against the training target.

## Data (`data/`)
| File | Rows | Description |
|---|---|---|
| `train.csv` | labelled | features plus `Target Pressure (bar)` |
| `test.csv` | unlabelled | features only |
| `sample_prediction.csv` | – | submission format (`ID`, `Target Pressure (bar)`) |

Features cover tank failure pressure, liquid ratio, tank/vapour/BLEVE dimensions, liquid and vapour temperatures, liquid critical properties, obstacle geometry (distance, width, height, thickness, angle), sensor ID and sensor position.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb
```
Run all cells to train the models and write predictions. A GPU is detected and used automatically when available (it speeds up the boosted models); otherwise everything runs on CPU. `predictions/prediction.csv` contains the output from my final run.
