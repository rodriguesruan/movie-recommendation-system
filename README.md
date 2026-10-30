# Movie Recommendation System

This project is a portfolio-ready movie recommendation system built on the MovieLens 1M dataset. The initial phase focuses on environment setup, raw data validation and an exploratory data analysis (EDA) to establish a solid, production-oriented foundation before model development.

## Objective

- Load and validate the official MovieLens 1M dataset.
- Understand user behavior, review counts, rating distributions and sparsity.
- Confirm data quality before any modeling work.
- Prepare a reproducible project structure for later phases of feature engineering, temporal validation and recommendation models.

## Dataset

The project uses the official MovieLens 1M dataset from GroupLens.

- Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Files used: ratings.dat, movies.dat, users.dat
- Raw data is stored locally in `data/raw/ml-1m` and is not committed to the repository.

## Project structure

```text
movie-recommendation-system/
├── .venv/
├── data/
│   └── raw/
│       └── ml-1m/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   └── validation/
├── tests/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── .venv/
```

## Local setup

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate  # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Downloading the dataset

```bash
python - <<'PY'
from pathlib import Path
import urllib.request

url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
out = Path('data/raw/ml-1m.zip')
out.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, out)
print('Downloaded to', out)
PY
```

Then unzip the archive into `data/raw`.

## Notebook execution

```bash
jupyter notebook notebooks/01_data_understanding.ipynb
```

## Running tests

```bash
python -m pytest -q
```

## Current status

This repository is in the first development phase only. The project validates the raw dataset and produces an initial exploratory analysis without training any recommendation model.

## Important notes

- No Docker is used.
- No paid APIs are required.
- Raw data is kept local to avoid committing datasets to GitHub.
- No metrics or model results are invented in this phase.
