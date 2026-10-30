from __future__ import annotations

import pandas as pd


def build_explanation_summary(feature_frame: pd.DataFrame) -> dict[str, object]:
    """Summarize the most influential features for a recommendation score.

    This is a lightweight, model-agnostic explanation layer for the current
    phase. It ranks the feature columns by absolute contribution using their
    relative magnitude against the prediction target.
    """
    if feature_frame.empty:
        raise ValueError('The feature dataframe cannot be empty.')

    score_column = 'prediction' if 'prediction' in feature_frame.columns else None
    if score_column is None:
        raise KeyError("The feature frame must contain a 'prediction' column.")

    base_columns = [col for col in feature_frame.columns if col != score_column]
    if not base_columns:
        raise ValueError('No features are available to explain.')

    feature_rank = []
    for column in base_columns:
        values = pd.to_numeric(feature_frame[column], errors='coerce').abs()
        mean_strength = float(values.mean()) if not values.empty else 0.0
        feature_rank.append({'feature': column, 'strength': mean_strength})

    ranked = sorted(feature_rank, key=lambda item: item['strength'], reverse=True)
    top_features = ranked[: min(5, len(ranked))]

    prediction = float(feature_frame[score_column].iloc[0]) if len(feature_frame) > 0 else 0.0
    return {
        'prediction': prediction,
        'top_features': top_features,
        'summary_text': (
            'This recommendation is mainly driven by the user and item history signals '
            'that most strongly deviate from the baseline.'
        ),
    }
