# Sentiment Classifier Bias Audit Across Demographic Subgroups

A Text Analytics term project that demonstrates subgroup auditing for a binary sentiment classifier.

## Deliverables
- `sentiment_bias_audit.ipynb` — complete Jupyter Notebook.
- `Term_Project_Report.pdf` — project report.
- `Output_Screenshots.pdf` — report-style output snapshots.
- `Demo_Video.mp4` — short walkthrough video.
- `bias_audit.py` — script version of the core workflow.
- `outputs/` — example output tables and plots (generated when notebook/script runs).

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook sentiment_bias_audit.ipynb
```
Run all notebook cells from top to bottom. The notebook generates a synthetic demonstration dataset and writes audit artifacts into `outputs/`.

## Method
TF–IDF features + Logistic Regression; stratified train/test split; overall and subgroup accuracy, precision, recall, F1, false-positive rate, false-negative rate, and positive prediction rate; configurable disparity flags.

## Important limitations
The data are synthetic and intentionally illustrative. Do not interpret the results as evidence about real demographic groups. For real-world use, replace the synthetic data only with appropriately licensed, consented, representative data; assess subgroup sample sizes and uncertainty; document label quality; and have domain experts review conclusions.

## GitHub submission
Create a new repository on GitHub, upload this project folder (or push it using Git), and share the repository URL with your instructor. A repository has not been published automatically from this environment.
