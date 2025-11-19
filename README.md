# Causal Forest Pipeline — Lalonde Dataset

This folder contains a 3-step pipeline to clean data, fit a causal forest using `econml`, and validate the model with summary metrics and figures.

## Structure
- `step1_clean_data.py` — Reads `../data/lalonde_data.csv`, prints basic diagnostics, saves processed data to `../data/df_processed.pkl`.
- `step2_fit_model.py` — Loads processed data, splits into train/test, performs a simple grid search over `CausalForestDML` hyperparameters using the **std. dev. of CATE** on the test set as the selection proxy. Saves the best model and test sets to:
  - `../data/best_causal_model.pkl`
  - `../data/X_test.pkl`
  - `../data/Y_test.pkl`
- `step3_validate_model.py` — Loads the best model and test features to estimate ATE/CATE and produces figures:
  - Prints ATE and 95% CI
  - Saves figures to `../figures/`:
    - `cate_distribution.pdf`
    - `cate_scatter.pdf`
    - `ate.pdf`

## Prerequisites
- Python 3.9+
- Packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `econml`
- Data file at `../data/lalonde_data.csv` with columns including:
  - `re78` (outcome), `treat` (treatment indicator), and covariates (remaining columns).

## Quickstart
1. Ensure directories exist:
   ```bash
   mkdir -p ../data ../figures
   ```
2. Place `lalonde_data.csv` in `../data/`.
3. Run the steps:
   ```bash
   python step1_clean_data.py
   python step2_fit_model.py
   python step3_validate_model.py
   ```

## Outputs
- Models & data: `../data/df_processed.pkl`, `../data/best_causal_model.pkl`, `../data/X_test.pkl`, `../data/Y_test.pkl`
- Figures: PDFs in `../figures/` as listed above.

## Notes & Tips
- Reproducibility: `random_state=1234` is used for the train/test split and the causal forest.
- Hyperparameter selection is illustrative (lower CATE variance on the test split). Consider cross-fitting or out-of-bag diagnostics for robust tuning.
- **Case sensitivity**: `step3_validate_model.py` tries to read `y_test.pkl`, but `step2_fit_model.py` saves `Y_test.pkl`. Rename the file or adjust the script:
  - Rename: `mv ../data/Y_test.pkl ../data/y_test.pkl`
  - **or** edit `step3_validate_model.py` to load `Y_test.pkl`.

## Citation
These scripts reference the Lalonde dataset setup and use `econml`'s `CausalForestDML` for heterogeneous treatment effect estimation.
