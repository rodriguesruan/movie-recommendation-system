# Temporal Model Evaluation

## Protocol

- Dataset: official MovieLens 1M ratings (`ratings.dat`), 1,000,209 rows.
- Split: chronological holdout; the latest 200,041 ratings are validation and the first 800,168 are training. Equal timestamps are kept in the same partition; the training maximum timestamp is strictly earlier than the validation minimum.
- Serving model: Random Forest trained on a deterministic sample of up to 30,000 training rows, using user/item rating history available before each sampled event.
- Baselines: training-set global rating mean and smoothed user/item bias model.
- Metrics: rating-regression RMSE, MAE, and R-squared. These are not top-K ranking metrics.

## Results

| Model | RMSE | MAE | R-squared |
| --- | ---: | ---: | ---: |
| Global mean | 1.1043 | 0.9195 | -0.0016 |
| Smoothed user/item bias | **0.9590** | **0.7585** | **0.2447** |
| Serving Random Forest | 0.9766 | 0.7844 | 0.2165 |

Lower RMSE and MAE are better; higher R-squared is better. On this holdout, the current Random Forest does not outperform the simpler user/item-bias baseline. This is an honest baseline comparison, not evidence that the ensemble or recommendation ranking is superior.

## Reproduce

From the repository root, with the requirements installed:

```bash
python -m src.validation.evaluate
```
