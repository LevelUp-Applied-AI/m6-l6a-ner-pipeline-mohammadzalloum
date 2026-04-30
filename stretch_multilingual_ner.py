"""
Module 6 Week A — Stretch: Multilingual NER Comparison

Task 1:
- Load climate_articles.csv
- Validate required columns
- Summarize language/category distribution
- Build a balanced multilingual sample:
  20 English texts + 20 Arabic texts
  using 5 texts per category per language.

Run:
    python stretch_multilingual_ner.py
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/climate_articles.csv")
OUTPUT_DIR = Path("outputs")
SAMPLE_OUTPUT_PATH = OUTPUT_DIR / "stretch_multilingual_sample.csv"

REQUIRED_COLUMNS = {"id", "text", "source", "language", "category"}

LANGUAGES = ["en", "ar"]
CATEGORIES = ["adaptation", "impact", "policy", "science"]

N_PER_LANGUAGE_CATEGORY = 5
RANDOM_STATE = 42


def load_data(data_path=DATA_PATH):
    """Load the climate articles dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {data_path}. "
            "Make sure you are running from the repo root."
        )

    df = pd.read_csv(data_path)
    return df


def validate_data(df):
    """Validate that the dataset has the columns needed for the stretch."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    missing_languages = set(LANGUAGES) - set(df["language"].dropna().unique())

    if missing_languages:
        raise ValueError(f"Missing required languages: {sorted(missing_languages)}")

    return True


def summarize_dataset(df):
    """Print useful dataset-level statistics."""
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nLanguages:")
    print(df["language"].value_counts())

    print("\nCategories:")
    print(df["category"].value_counts())

    print("\nLanguage x Category:")
    print(pd.crosstab(df["language"], df["category"]))

    word_counts = df["text"].fillna("").astype(str).str.split().str.len()
    print("\nText length summary:")
    print(word_counts.describe())


def build_balanced_sample(df):
    """
    Build a fair sample for multilingual comparison.

    We take 5 texts from each category for each language:
    2 languages x 4 categories x 5 texts = 40 texts total.
    """
    sample_parts = []

    for language in LANGUAGES:
        for category in CATEGORIES:
            subset = df[
                (df["language"] == language)
                & (df["category"] == category)
            ].copy()

            if len(subset) < N_PER_LANGUAGE_CATEGORY:
                raise ValueError(
                    f"Not enough rows for language={language}, "
                    f"category={category}. "
                    f"Needed {N_PER_LANGUAGE_CATEGORY}, found {len(subset)}."
                )

            sampled = subset.sample(
                n=N_PER_LANGUAGE_CATEGORY,
                random_state=RANDOM_STATE
            )

            sample_parts.append(sampled)

    sample_df = (
        pd.concat(sample_parts, ignore_index=True)
        .sort_values(["language", "category", "id"])
        .reset_index(drop=True)
    )

    return sample_df


def save_sample(sample_df, output_path=SAMPLE_OUTPUT_PATH):
    """Save the multilingual sample with UTF-8 support for Arabic."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved multilingual sample to: {output_path}")


def summarize_sample(sample_df):
    """Print summary of selected sample."""
    print("\n=== Multilingual Sample Summary ===")
    print(f"Sample shape: {sample_df.shape}")

    print("\nSample language counts:")
    print(sample_df["language"].value_counts())

    print("\nSample language x category:")
    print(pd.crosstab(sample_df["language"], sample_df["category"]))

    print("\nSample preview:")
    preview_cols = ["id", "language", "category", "source"]
    print(sample_df[preview_cols].to_string(index=False))


def main():
    df = load_data()
    validate_data(df)

    summarize_dataset(df)

    sample_df = build_balanced_sample(df)
    summarize_sample(sample_df)
    save_sample(sample_df)


if __name__ == "__main__":
    main()