# Brief Requirements Checklist — Task 3: Fraud Detection (MLOps)

Quick reference for the oral defence: every requirement from the brief,
mapped to where it's satisfied in the repo. Quotes are from the official
task sheet (`DLBDSMTP01`, Task 3).

## Pipeline requirements

| # | Requirement (quoted) | Status | Evidence | Notes |
|---|---|---|---|---|
| R1 | "Design the conceptual architecture of your system... Draft a visual overview." | ✅ Met | [README.md](README.md) — mermaid flowchart | |
| R2 | "Choose an open data source... Alternatively, produce your own fictional sample data." | ⚠️ Open | [data/raw/fraud_data.csv](data/raw/fraud_data.csv) | Data is present and used correctly, but its source isn't cited anywhere in the repo. See `ACTION_PLAN.md` Step 5. |
| R3 | "Build a simple fraud detection model... check basic statistical measures." | ✅ Met | [src/train_model.py](src/train_model.py) — 3 candidate models, cross-validated, ROC-AUC/PR-AUC/precision/recall/F1 | |
| R4 | "Package your model... take data over a standardized RESTful API and respond with a probability for fraud." | ✅ Met | [src/api.py](src/api.py) `/predict` endpoint | |
| R5 | "Performance... can be easily monitored... only those eyes... meant to see the data... have access." | ✅ Met | [src/monitoring.py](src/monitoring.py) (drift monitoring), [src/auth.py](src/auth.py) (API key access control) | |
| R6 | "Re-trains the model after a given time (one month) **or** after the incoming data has changed... simulating one year of new data and monthly re-training... still reachable over RESTful API." | ✅ Met | [.github/workflows/mlops.yml](.github/workflows/mlops.yml) — monthly scheduled trigger, 12-month drift simulation, retrain-on-drift, live API reachability check post-retrain | |
| R7 | Cloud deployment | Not attempted | — | Explicitly a bonus, not required for top grade |

## Report / oral defence questions

The brief also asks you to address these in the report itself — not code to
point at, but things to have a clear answer ready for. Where relevant, here's
where the supporting evidence lives so you're not answering from memory alone.

| # | Question (quoted) | Where the evidence lives |
|---|---|---|
| Q1 | "What were the challenges of integrating a predictive model into an application or service?" | Your own experience — e.g. reconciling training-time vs. serving-time feature engineering (solved via one shared `Pipeline`, see `src/train_model.py` + `src/api.py`) |
| Q2 | "What are the constraints of implementing a predictive model as a service?" | Auth (`src/auth.py`), input validation (`src/api.py`), synchronous single-transaction scoring |
| Q3 | "Which requirements for data acquisition, storage, and processing had to be met, and how did you achieve this?" | `src/data_loader.py`, `src/preprocess.py`, README's "Project flow" section |
| Q4 | "What are monitoring components required for reliable execution of the predictive model?" | `/health` endpoint, `src/monitoring.py`'s statistical drift tests |
| Q5 | "What is the design of your system? Present a visual draft." | README mermaid diagram |
| Q6 | "Provide a link for your audience to follow and reproduce your code." | Have the repo URL ready on the day |

## Summary

6 of 7 pipeline requirements fully met. The one open item (data source
citation) is a 10-minute fix — see `ACTION_PLAN.md` Step 5. R7 (cloud) is a
bonus you can skip without any grade impact.
