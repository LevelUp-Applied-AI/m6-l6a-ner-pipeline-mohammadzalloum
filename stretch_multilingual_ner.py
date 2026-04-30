"""
Module 6 Week A — Stretch: Multilingual NER Comparison

Task 1:
- Load climate_articles.csv
- Validate required columns
- Summarize language/category distribution
- Build a balanced multilingual sample:
  20 English texts + 20 Arabic texts
  using 5 texts per category per language.

Task 2:
- Load multilingual NER models:
  spaCy: xx_ent_wiki_sm
  Hugging Face: Davlan/xlm-roberta-base-wikiann-ner
- Run a small smoke test on one English text and one Arabic text.
- Save smoke-test entity examples.

Task 3:
- Run both multilingual models on the full balanced sample.
- Save all extracted entities.
- Save document-level entity counts for no-entity-rate analysis.

Run:
    python stretch_multilingual_ner.py
"""

from pathlib import Path

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline


DATA_PATH = Path("data/climate_articles.csv")
OUTPUT_DIR = Path("outputs")

SAMPLE_OUTPUT_PATH = OUTPUT_DIR / "stretch_multilingual_sample.csv"
SMOKE_TEST_OUTPUT_PATH = OUTPUT_DIR / "multilingual_model_smoke_test.csv"
RAW_ENTITIES_OUTPUT_PATH = OUTPUT_DIR / "multilingual_raw_entities.csv"
DOCUMENT_COUNTS_OUTPUT_PATH = OUTPUT_DIR / "multilingual_document_entity_counts.csv"

REQUIRED_COLUMNS = {"id", "text", "source", "language", "category"}

LANGUAGES = ["en", "ar"]
CATEGORIES = ["adaptation", "impact", "policy", "science"]

N_PER_LANGUAGE_CATEGORY = 5
RANDOM_STATE = 42

SPACY_MULTILINGUAL_MODEL = "xx_ent_wiki_sm"
HF_MULTILINGUAL_MODEL = "Davlan/xlm-roberta-base-wikiann-ner"


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


# ============================================================
# Task 2: Load and test multilingual NER models
# ============================================================

def load_spacy_multilingual_model(model_name=SPACY_MULTILINGUAL_MODEL):
    """
    Load spaCy multilingual NER model.

    This model emits a smaller label set than en_core_web_sm:
    usually PER, LOC, ORG, and MISC.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{model_name}'.\n"
            f"Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc

    return nlp


def load_hf_multilingual_model(model_name=HF_MULTILINGUAL_MODEL):
    """
    Load Hugging Face multilingual NER pipeline.

    aggregation_strategy='simple' combines subword pieces into cleaner
    entity spans where possible.
    """
    try:
        ner_pipeline = hf_pipeline(
            task="ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Hugging Face model '{model_name}'.\n"
            "Check your internet connection the first time you download it, "
            "and make sure transformers and torch are installed."
        ) from exc

    return ner_pipeline


def extract_spacy_entities_from_text(text, text_id, language, model_name, nlp):
    """Run spaCy multilingual NER on one text and return entity rows."""
    rows = []

    doc = nlp("" if pd.isna(text) else str(text))

    for ent in doc.ents:
        rows.append({
            "text_id": text_id,
            "language": language,
            "model": model_name,
            "entity_text": ent.text,
            "entity_label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "score": None,
        })

    return rows


def extract_hf_entities_from_text(text, text_id, language, model_name, ner_pipeline):
    """Run Hugging Face multilingual NER on one text and return entity rows.

    The pipeline usually returns start/end character offsets. When offsets
    are available, we use the original text slice to preserve Arabic text
    exactly and avoid tokenizer artifacts.
    """
    rows = []

    text = "" if pd.isna(text) else str(text)

    entities = ner_pipeline(text)

    for ent in entities:
        entity_label = ent.get("entity_group", ent.get("entity", ""))
        start_char = ent.get("start")
        end_char = ent.get("end")
        score = ent.get("score")

        if start_char is not None and end_char is not None:
            start_char = int(start_char)
            end_char = int(end_char)
            entity_text = text[start_char:end_char]
        else:
            raw_word = str(ent.get("word", ""))
            entity_text = raw_word.replace("▁", " ").replace("##", "").strip()
            entity_text = " ".join(entity_text.split())

        rows.append({
            "text_id": text_id,
            "language": language,
            "model": model_name,
            "entity_text": entity_text,
            "entity_label": entity_label,
            "start_char": start_char,
            "end_char": end_char,
            "score": float(score) if score is not None else None,
        })

    return rows


def pick_smoke_test_texts(sample_df):
    """
    Pick one English and one Arabic text from the balanced sample.

    We choose policy if available because policy texts usually contain
    organizations, agreements, countries, dates, and climate events.
    """
    selected_rows = []

    for language in LANGUAGES:
        language_df = sample_df[sample_df["language"] == language].copy()

        policy_df = language_df[language_df["category"] == "policy"].copy()

        if not policy_df.empty:
            selected_rows.append(policy_df.iloc[0])
        else:
            selected_rows.append(language_df.iloc[0])

    return pd.DataFrame(selected_rows)


def run_model_smoke_test(sample_df, spacy_nlp, hf_ner):
    """
    Run both multilingual models on one English and one Arabic sample text.

    This verifies that both models can process both languages before we run
    the full comparison in later tasks.
    """
    test_df = pick_smoke_test_texts(sample_df)

    all_rows = []

    for _, row in test_df.iterrows():
        text_id = row["id"]
        language = row["language"]
        text = row["text"]

        all_rows.extend(
            extract_spacy_entities_from_text(
                text=text,
                text_id=text_id,
                language=language,
                model_name=f"spaCy:{SPACY_MULTILINGUAL_MODEL}",
                nlp=spacy_nlp,
            )
        )

        all_rows.extend(
            extract_hf_entities_from_text(
                text=text,
                text_id=text_id,
                language=language,
                model_name=f"HF:{HF_MULTILINGUAL_MODEL}",
                ner_pipeline=hf_ner,
            )
        )

    columns = [
        "text_id",
        "language",
        "model",
        "entity_text",
        "entity_label",
        "start_char",
        "end_char",
        "score",
    ]

    smoke_df = pd.DataFrame(all_rows, columns=columns)

    return smoke_df, test_df


def save_smoke_test_results(smoke_df, output_path=SMOKE_TEST_OUTPUT_PATH):
    """Save smoke test entities to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved multilingual model smoke test to: {output_path}")


def print_smoke_test_results(smoke_df, test_df):
    """Print readable smoke test summary."""
    print("\n=== Task 2: Multilingual Model Smoke Test ===")

    print("\nTexts used for smoke test:")
    print(test_df[["id", "language", "category", "source"]].to_string(index=False))

    if smoke_df.empty:
        print("\nNo entities found by either model in the smoke test texts.")
        return

    print("\nEntities found:")
    display_cols = [
        "text_id",
        "language",
        "model",
        "entity_text",
        "entity_label",
        "score",
    ]

    print(smoke_df[display_cols].to_string(index=False))

    print("\nEntity counts by language and model:")
    print(
        smoke_df
        .groupby(["language", "model"])
        .size()
        .reset_index(name="entity_count")
        .to_string(index=False)
    )

    print("\nEntity counts by label:")
    print(
        smoke_df
        .groupby(["language", "model", "entity_label"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )


# ============================================================
# Task 3: Run multilingual NER on the full balanced sample
# ============================================================

def count_words(text):
    """Count whitespace-separated words in a text."""
    text = "" if pd.isna(text) else str(text)
    return len(text.split())


def add_document_metadata(entity_rows, row, text_length_words):
    """Attach article-level metadata to each entity row."""
    enriched_rows = []

    for entity_row in entity_rows:
        enriched_row = entity_row.copy()
        enriched_row["category"] = row["category"]
        enriched_row["source"] = row["source"]
        enriched_row["text_length_words"] = text_length_words
        enriched_rows.append(enriched_row)

    return enriched_rows


def extract_entities_for_sample(sample_df, spacy_nlp, hf_ner):
    """
    Run both multilingual NER models over the full balanced sample.

    Returns:
        raw_entities_df:
            One row per extracted entity.

        document_counts_df:
            One row per text per model, including entity_count.
            This keeps track of texts where a model found zero entities.
    """
    raw_entity_rows = []
    document_count_rows = []

    model_extractors = [
        (
            f"spaCy:{SPACY_MULTILINGUAL_MODEL}",
            lambda text, text_id, language: extract_spacy_entities_from_text(
                text=text,
                text_id=text_id,
                language=language,
                model_name=f"spaCy:{SPACY_MULTILINGUAL_MODEL}",
                nlp=spacy_nlp,
            ),
        ),
        (
            f"HF:{HF_MULTILINGUAL_MODEL}",
            lambda text, text_id, language: extract_hf_entities_from_text(
                text=text,
                text_id=text_id,
                language=language,
                model_name=f"HF:{HF_MULTILINGUAL_MODEL}",
                ner_pipeline=hf_ner,
            ),
        ),
    ]

    for _, row in sample_df.iterrows():
        text_id = row["id"]
        language = row["language"]
        text = "" if pd.isna(row["text"]) else str(row["text"])
        text_length_words = count_words(text)

        for model_name, extractor in model_extractors:
            entity_rows = extractor(text, text_id, language)

            enriched_rows = add_document_metadata(
                entity_rows=entity_rows,
                row=row,
                text_length_words=text_length_words,
            )

            raw_entity_rows.extend(enriched_rows)

            document_count_rows.append({
                "text_id": text_id,
                "language": language,
                "category": row["category"],
                "source": row["source"],
                "model": model_name,
                "text_length_words": text_length_words,
                "entity_count": len(entity_rows),
                "has_entities": len(entity_rows) > 0,
            })

    raw_entity_columns = [
        "text_id",
        "language",
        "category",
        "source",
        "model",
        "entity_text",
        "entity_label",
        "start_char",
        "end_char",
        "score",
        "text_length_words",
    ]

    document_count_columns = [
        "text_id",
        "language",
        "category",
        "source",
        "model",
        "text_length_words",
        "entity_count",
        "has_entities",
    ]

    raw_entities_df = pd.DataFrame(raw_entity_rows, columns=raw_entity_columns)
    document_counts_df = pd.DataFrame(document_count_rows, columns=document_count_columns)

    return raw_entities_df, document_counts_df


def save_full_extraction_outputs(
    raw_entities_df,
    document_counts_df,
    raw_output_path=RAW_ENTITIES_OUTPUT_PATH,
    document_counts_output_path=DOCUMENT_COUNTS_OUTPUT_PATH,
):
    """Save full multilingual NER extraction outputs."""
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_entities_df.to_csv(raw_output_path, index=False, encoding="utf-8-sig")
    document_counts_df.to_csv(
        document_counts_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved raw multilingual entities to: {raw_output_path}")
    print(f"Saved document entity counts to: {document_counts_output_path}")


def print_full_extraction_summary(raw_entities_df, document_counts_df):
    """Print a readable summary of the full multilingual extraction."""
    print("\n=== Task 3: Full Multilingual NER Extraction ===")

    print(f"\nTotal extracted entities: {len(raw_entities_df)}")
    print(f"Documents processed: {document_counts_df['text_id'].nunique()}")
    print(f"Model runs: {len(document_counts_df)}")

    print("\nEntity counts by language and model:")
    print(
        document_counts_df
        .groupby(["language", "model"])["entity_count"]
        .sum()
        .reset_index(name="total_entities")
        .to_string(index=False)
    )

    print("\nAverage entities per document by language and model:")
    print(
        document_counts_df
        .groupby(["language", "model"])["entity_count"]
        .mean()
        .reset_index(name="avg_entities_per_doc")
        .to_string(index=False)
    )

    print("\nDocuments with no entities by language and model:")
    no_entity_summary = (
        document_counts_df
        .assign(no_entities=lambda df: df["entity_count"] == 0)
        .groupby(["language", "model"])["no_entities"]
        .agg(["sum", "count"])
        .reset_index()
    )

    no_entity_summary["no_entity_rate"] = (
        no_entity_summary["sum"] / no_entity_summary["count"]
    )

    print(no_entity_summary.to_string(index=False))

    if raw_entities_df.empty:
        print("\nNo raw entities were extracted.")
        return

    print("\nEntity label counts by language and model:")
    print(
        raw_entities_df
        .groupby(["language", "model", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["language", "model", "count"], ascending=[True, True, False])
        .to_string(index=False)
    )

    print("\nSample extracted entities:")
    print(
        raw_entities_df[
            [
                "text_id",
                "language",
                "category",
                "model",
                "entity_text",
                "entity_label",
                "score",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


def main():
    df = load_data()
    validate_data(df)

    summarize_dataset(df)

    sample_df = build_balanced_sample(df)
    summarize_sample(sample_df)
    save_sample(sample_df)

    print("\n=== Loading multilingual models ===")
    spacy_nlp = load_spacy_multilingual_model()
    print(f"Loaded spaCy model: {SPACY_MULTILINGUAL_MODEL}")

    hf_ner = load_hf_multilingual_model()
    print(f"Loaded Hugging Face model: {HF_MULTILINGUAL_MODEL}")

    smoke_df, test_df = run_model_smoke_test(sample_df, spacy_nlp, hf_ner)
    print_smoke_test_results(smoke_df, test_df)
    save_smoke_test_results(smoke_df)

    raw_entities_df, document_counts_df = extract_entities_for_sample(
        sample_df=sample_df,
        spacy_nlp=spacy_nlp,
        hf_ner=hf_ner,
    )

    print_full_extraction_summary(raw_entities_df, document_counts_df)

    save_full_extraction_outputs(
        raw_entities_df=raw_entities_df,
        document_counts_df=document_counts_df,
    )


if __name__ == "__main__":
    main()