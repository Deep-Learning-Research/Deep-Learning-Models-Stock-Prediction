# Revised experiments (manuscript V3)

Re-run of the repository notebooks with unchanged model code and settings
(1 layer, 128 units, window 5, Adam lr 0.01, 40 epochs, keep prob 0.8, 75-day recursive forecast, EMA 0.3).

- `code/run_<model>.py`  : basic_rnn / lstm / bidirectional_lstm / gru, ported from the *_TWTR notebooks
  Usage: `TF_USE_LEGACY_KERAS=1 python run_lstm.py dataset/AAPL.csv 50 results/AAPL_lstm`
- `code/runtr_<model>.py`: same, but min-max scaler fitted on training data only (Section 4.5)
- `code/analyse.py`      : metrics, naive and Monte Carlo benchmarks, Mann-Whitney tests, all figures
- `results/*_forecasts.npy`, `*_times.npy`: 50 forecasts and run times per model
- `results/results.json` : every number reported in the paper
- `figs/`                : Figures 1 to 4

Environment: Python 3.13, TensorFlow 2.21 (tf-keras legacy mode), 2-core CPU.
Model code adapted from https://github.com/huseinzol05/Stock-Prediction-Models (Apache 2.0).
