# Movie Recommendation System

This project is a movie recommendation system built on MovieLens 1M. It includes data validation and EDA, leakage-safe temporal features, a trained Random Forest ranking model, a FastAPI endpoint, and a Streamlit dashboard.

## Objective

- Load and validate the official MovieLens 1M dataset.
- Understand user behavior, review counts, rating distributions and sparsity.
- Train and evaluate recommendation models using chronologically ordered data.
- Rank unseen movies with Random Forest predictions and provide explanations based on observed rating history.
- Serve recommendations through FastAPI and Streamlit.

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

## Dataset setup

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

Then unzip the archive into `data/raw`. The API and Streamlit app also download the official archive automatically when the required files are missing. The download stays local or in the host's temporary filesystem and is never committed.

## Notebook execution

```bash
jupyter notebook notebooks/01_data_understanding.ipynb
```

## Running tests

```bash
python -m pytest -q
```

## Run the API

Install the requirements, then start the HTTP API from the project root:

```bash
uvicorn src.api:app --reload
```

- Health check: `http://127.0.0.1:8000/health`
- Recommendations: `http://127.0.0.1:8000/recommendations/1?top_n=10`
- Interactive API docs: `http://127.0.0.1:8000/docs`

The API reads MovieLens files from `data/raw/ml-1m`, downloading the official dataset if needed. It returns HTTP 404 for an unknown user, and `top_n` must be between 1 and 50. The serving model is trained once per API process using prior-only rating-history features and a deterministic sample of up to 100,000 rows.

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard downloads MovieLens if needed, trains the same serving model once, and caches the data/model for the session.

## Deploy the dashboard

The repository includes a Render Blueprint in `render.yaml`.

1. Sign in to Render and connect the GitHub repository `rodriguesruan/movie-recommendation-system`.
2. Create a new Blueprint from the repository and approve the `movie-recommendation-system` web service.
3. Wait for the build and open the generated `onrender.com` URL.

The free service may sleep when idle and cold-start by downloading MovieLens and training the model again. The FastAPI service remains available for local use with the command above; a separate hosted API service is not configured.

## Current status

The recommendation dashboard, FastAPI service, and GitHub Actions tests are implemented. The dashboard is prepared for Render deployment, but it is not publicly hosted until the Blueprint is authorized in a Render account.

## Important notes

- No Docker is used.
- No paid APIs are required.
- Raw data is kept local to avoid committing datasets to GitHub.
- Model quality should be judged from the temporal evaluation reports, not from the illustrative UI alone.
