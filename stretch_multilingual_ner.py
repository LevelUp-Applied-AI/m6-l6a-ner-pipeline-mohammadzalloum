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
import re
import unicodedata

import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline


DATA_PATH = Path("data/climate_articles.csv")
OUTPUT_DIR = Path("outputs")

SAMPLE_OUTPUT_PATH = OUTPUT_DIR / "stretch_multilingual_sample.csv"
SMOKE_TEST_OUTPUT_PATH = OUTPUT_DIR / "multilingual_model_smoke_test.csv"
RAW_ENTITIES_OUTPUT_PATH = OUTPUT_DIR / "multilingual_raw_entities.csv"
DOCUMENT_COUNTS_OUTPUT_PATH = OUTPUT_DIR / "multilingual_document_entity_counts.csv"

COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "multilingual_ner_comparison.csv"
LABEL_COUNTS_OUTPUT_PATH = OUTPUT_DIR / "multilingual_label_counts.csv"
EXAMPLE_ENTITIES_OUTPUT_PATH = OUTPUT_DIR / "multilingual_example_entities.csv"

QUALITATIVE_REVIEW_OUTPUT_PATH = OUTPUT_DIR / "multilingual_qualitative_review.csv"
FAILURE_CANDIDATES_OUTPUT_PATH = OUTPUT_DIR / "multilingual_failure_candidates.csv"
QUALITATIVE_NOTES_OUTPUT_PATH = OUTPUT_DIR / "multilingual_qualitative_notes.md"


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


# ============================================================
# Task 4: Build multilingual NER comparison table
# ============================================================

def build_label_counts(raw_entities_df):
    """
    Count entity labels for each language/model pair.

    Example output:
        ar + HF model: LOC=38, ORG=31, PER=1
    """
    columns = ["language", "model", "entity_label", "count"]

    if raw_entities_df.empty:
        return pd.DataFrame(columns=columns)

    label_counts_df = (
        raw_entities_df
        .groupby(["language", "model", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["language", "model", "count", "entity_label"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )

    return label_counts_df


def build_label_count_strings(label_counts_df):
    """
    Convert label counts into compact strings for the comparison table.

    Example:
        LOC=38; ORG=31; PER=1
    """
    columns = ["language", "model", "entity_type_counts"]

    if label_counts_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for (language, model), group in label_counts_df.groupby(["language", "model"]):
        group = group.sort_values(
            ["count", "entity_label"],
            ascending=[False, True],
        )

        label_string = "; ".join(
            f"{row.entity_label}={int(row.count)}"
            for row in group.itertuples(index=False)
        )

        rows.append({
            "language": language,
            "model": model,
            "entity_type_counts": label_string,
        })

    return pd.DataFrame(rows, columns=columns)


def build_example_entities(raw_entities_df, n_examples=3):
    """
    Select up to 3 example entities for each language/model pair.

    The selection tries to include different labels first, then fills
    remaining slots with high-confidence or early examples.
    """
    columns = [
        "language",
        "model",
        "example_rank",
        "text_id",
        "category",
        "source",
        "entity_text",
        "entity_label",
        "score",
    ]

    if raw_entities_df.empty:
        return pd.DataFrame(columns=columns)

    work = raw_entities_df.copy()

    work["entity_text"] = (
        work["entity_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    work = work[work["entity_text"] != ""].copy()

    if work.empty:
        return pd.DataFrame(columns=columns)

    work["score_for_sort"] = pd.to_numeric(
        work["score"],
        errors="coerce",
    ).fillna(-1.0)

    work = work.drop_duplicates(
        subset=["language", "model", "entity_text", "entity_label"]
    )

    selected_groups = []

    for (language, model), group in work.groupby(["language", "model"]):
        group = group.copy()

        label_order = (
            group["entity_label"]
            .value_counts()
            .index
            .tolist()
        )

        selected_indexes = []

        # First pass: try to pick one strong example from each label.
        for label in label_order:
            candidates = (
                group[group["entity_label"] == label]
                .sort_values(
                    ["score_for_sort", "text_id"],
                    ascending=[False, True],
                )
            )

            if not candidates.empty:
                selected_indexes.append(candidates.index[0])

            if len(selected_indexes) >= n_examples:
                break

        # Second pass: fill remaining slots with best remaining examples.
        if len(selected_indexes) < n_examples:
            remaining = (
                group[~group.index.isin(selected_indexes)]
                .sort_values(
                    ["score_for_sort", "text_id"],
                    ascending=[False, True],
                )
            )

            needed = n_examples - len(selected_indexes)
            selected_indexes.extend(list(remaining.index[:needed]))

        selected = group.loc[selected_indexes].copy()
        selected = selected.head(n_examples)
        selected["example_rank"] = range(1, len(selected) + 1)

        selected_groups.append(selected[columns])

    if not selected_groups:
        return pd.DataFrame(columns=columns)

    return pd.concat(selected_groups, ignore_index=True)


def build_example_strings(example_entities_df):
    """
    Convert selected examples into compact strings for the comparison table.

    Example:
        Jordan (LOC) | Green Climate Fund (ORG) | South Korea (LOC)
    """
    columns = ["language", "model", "example_entities"]

    if example_entities_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for (language, model), group in example_entities_df.groupby(["language", "model"]):
        group = group.sort_values("example_rank")

        examples = [
            f"{row.entity_text} ({row.entity_label})"
            for row in group.itertuples(index=False)
        ]

        rows.append({
            "language": language,
            "model": model,
            "example_entities": " | ".join(examples),
        })

    return pd.DataFrame(rows, columns=columns)


def build_comparison_table(raw_entities_df, document_counts_df, n_examples=3):
    """
    Build the main multilingual NER comparison table.

    The table uses native labels from each multilingual model.
    This matches the stretch assignment because we are not computing
    Arabic precision/recall/F1 and only need cross-lingual comparison.
    """
    summary_df = (
        document_counts_df
        .groupby(["language", "model"])
        .agg(
            texts_processed=("text_id", "nunique"),
            total_words=("text_length_words", "sum"),
            total_entities=("entity_count", "sum"),
            avg_entities_per_doc=("entity_count", "mean"),
            no_entity_texts=("entity_count", lambda values: int((values == 0).sum())),
        )
        .reset_index()
    )

    summary_df["entities_per_100_words"] = summary_df.apply(
        lambda row: (
            (row["total_entities"] / row["total_words"]) * 100
            if row["total_words"] > 0
            else 0.0
        ),
        axis=1,
    )

    summary_df["no_entity_rate"] = summary_df.apply(
        lambda row: (
            row["no_entity_texts"] / row["texts_processed"]
            if row["texts_processed"] > 0
            else 0.0
        ),
        axis=1,
    )

    summary_df["label_schema"] = "native_labels"

    label_counts_df = build_label_counts(raw_entities_df)
    label_count_strings_df = build_label_count_strings(label_counts_df)

    example_entities_df = build_example_entities(
        raw_entities_df,
        n_examples=n_examples,
    )
    example_strings_df = build_example_strings(example_entities_df)

    comparison_df = summary_df.merge(
        label_count_strings_df,
        on=["language", "model"],
        how="left",
    )

    comparison_df = comparison_df.merge(
        example_strings_df,
        on=["language", "model"],
        how="left",
    )

    comparison_df["entity_type_counts"] = comparison_df[
        "entity_type_counts"
    ].fillna("None")

    comparison_df["example_entities"] = comparison_df[
        "example_entities"
    ].fillna("None")

    comparison_df["avg_entities_per_doc"] = comparison_df[
        "avg_entities_per_doc"
    ].round(2)

    comparison_df["entities_per_100_words"] = comparison_df[
        "entities_per_100_words"
    ].round(2)

    comparison_df["no_entity_rate"] = comparison_df[
        "no_entity_rate"
    ].round(3)

    comparison_df = comparison_df[
        [
            "language",
            "model",
            "label_schema",
            "texts_processed",
            "total_words",
            "total_entities",
            "avg_entities_per_doc",
            "entities_per_100_words",
            "no_entity_texts",
            "no_entity_rate",
            "entity_type_counts",
            "example_entities",
        ]
    ].sort_values(["language", "model"]).reset_index(drop=True)

    return comparison_df, label_counts_df, example_entities_df


def save_comparison_outputs(
    comparison_df,
    label_counts_df,
    example_entities_df,
    comparison_output_path=COMPARISON_OUTPUT_PATH,
    label_counts_output_path=LABEL_COUNTS_OUTPUT_PATH,
    examples_output_path=EXAMPLE_ENTITIES_OUTPUT_PATH,
):
    """Save Task 4 comparison outputs."""
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(
        comparison_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    label_counts_df.to_csv(
        label_counts_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    example_entities_df.to_csv(
        examples_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved comparison table to: {comparison_output_path}")
    print(f"Saved label counts to: {label_counts_output_path}")
    print(f"Saved example entities to: {examples_output_path}")


def print_comparison_table(comparison_df):
    """Print the final comparison table in the terminal."""
    print("\n=== Task 4: Multilingual NER Comparison Table ===")

    if comparison_df.empty:
        print("No comparison rows were generated.")
        return

    print(comparison_df.to_string(index=False))



# ============================================================
# Task 5: Qualitative cross-lingual NER analysis
# ============================================================

def normalize_arabic_text(text):
    """
    Normalize Arabic text lightly for comparison.

    This is not used to change the saved entity text. It is only used
    to compare whether two models found approximately the same entity.
    """
    text = "" if pd.isna(text) else str(text)

    # Remove diacritics and tatweel.
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    text = text.replace("ـ", "")

    # Normalize common Arabic letter variants.
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    return text


def normalize_entity_for_match(entity_text, language):
    """
    Create a normalized key for approximate model-to-model comparison.

    The original entity text remains unchanged in output files.
    """
    text = "" if pd.isna(entity_text) else str(entity_text)
    text = text.strip().casefold()

    if language == "ar":
        text = normalize_arabic_text(text)

    # Remove punctuation but keep Arabic/English letters, numbers, and spaces.
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def format_entities_for_review(entity_df):
    """
    Convert entity rows into a compact readable string.

    Example:
        Jordan (LOC) | Green Climate Fund (ORG)
    """
    if entity_df.empty:
        return "None"

    work = entity_df.copy()
    work["entity_text"] = work["entity_text"].fillna("").astype(str).str.strip()
    work["entity_label"] = work["entity_label"].fillna("").astype(str).str.strip()

    work = work[work["entity_text"] != ""].copy()

    if work.empty:
        return "None"

    work = work.drop_duplicates(subset=["entity_text", "entity_label"])

    parts = [
        f"{row.entity_text} ({row.entity_label})"
        for row in work.itertuples(index=False)
    ]

    return " | ".join(parts)


def entity_key_to_display_map(entity_df, language):
    """
    Build a map from normalized entity key to readable entity string.

    If several entities normalize to the same key, keep the first display form.
    """
    display_map = {}

    if entity_df.empty:
        return display_map

    for row in entity_df.itertuples(index=False):
        key = normalize_entity_for_match(row.entity_text, language)

        if not key:
            continue

        display = f"{row.entity_text} ({row.entity_label})"
        display_map.setdefault(key, display)

    return display_map


def build_qualitative_review(sample_df, raw_entities_df):
    """
    Build side-by-side qualitative comparison for each text.

    For each text, compare what spaCy found vs what Hugging Face found.
    This is especially useful for Arabic because there is no Arabic gold
    standard in the assignment.
    """
    rows = []

    spacy_model = f"spaCy:{SPACY_MULTILINGUAL_MODEL}"
    hf_model = f"HF:{HF_MULTILINGUAL_MODEL}"

    for article in sample_df.itertuples(index=False):
        text_id = article.id
        language = article.language

        article_entities = raw_entities_df[
            raw_entities_df["text_id"] == text_id
        ].copy()

        spacy_entities = article_entities[
            article_entities["model"] == spacy_model
        ].copy()

        hf_entities = article_entities[
            article_entities["model"] == hf_model
        ].copy()

        spacy_map = entity_key_to_display_map(spacy_entities, language)
        hf_map = entity_key_to_display_map(hf_entities, language)

        spacy_keys = set(spacy_map)
        hf_keys = set(hf_map)

        shared_keys = sorted(spacy_keys & hf_keys)
        spacy_only_keys = sorted(spacy_keys - hf_keys)
        hf_only_keys = sorted(hf_keys - spacy_keys)

        shared_entities = " | ".join(
            hf_map.get(key, spacy_map.get(key, key))
            for key in shared_keys
        ) or "None"

        spacy_only_entities = " | ".join(
            spacy_map[key]
            for key in spacy_only_keys
        ) or "None"

        hf_only_entities = " | ".join(
            hf_map[key]
            for key in hf_only_keys
        ) or "None"

        spacy_count = len(spacy_entities)
        hf_count = len(hf_entities)

        if language == "ar" and spacy_count == 0 and hf_count > 0:
            qualitative_note = (
                "Arabic coverage gap: HF found entities while spaCy found none."
            )
        elif language == "ar" and hf_count >= max(2, spacy_count * 2):
            qualitative_note = (
                "Arabic model difference: HF found substantially more entities than spaCy."
            )
        elif language == "en" and abs(hf_count - spacy_count) <= 2:
            qualitative_note = (
                "English outputs are similar in entity count; inspect label differences."
            )
        else:
            qualitative_note = (
                "Model outputs differ; inspect examples for boundary/type differences."
            )

        text = "" if pd.isna(article.text) else str(article.text)
        text_snippet = text[:260].replace("\n", " ")

        rows.append({
            "text_id": text_id,
            "language": language,
            "category": article.category,
            "source": article.source,
            "text_snippet": text_snippet,
            "spacy_entity_count": spacy_count,
            "hf_entity_count": hf_count,
            "shared_entities": shared_entities,
            "spacy_only_entities": spacy_only_entities,
            "hf_only_entities": hf_only_entities,
            "spacy_entities": format_entities_for_review(spacy_entities),
            "hf_entities": format_entities_for_review(hf_entities),
            "qualitative_note": qualitative_note,
        })

    return pd.DataFrame(rows)


def build_failure_candidates(raw_entities_df):
    """
    Identify candidate errors or outputs worth manual review.

    These are not automatically labeled as wrong. They are review candidates
    that help write the qualitative analysis.
    """
    rows = []

    if raw_entities_df.empty:
        return pd.DataFrame(columns=[
            "text_id",
            "language",
            "category",
            "source",
            "model",
            "entity_text",
            "entity_label",
            "score",
            "review_reason",
        ])

    for row in raw_entities_df.itertuples(index=False):
        entity_text = "" if pd.isna(row.entity_text) else str(row.entity_text).strip()
        entity_label = "" if pd.isna(row.entity_label) else str(row.entity_label).strip()
        score = row.score

        reasons = []

        is_spacy = str(row.model).startswith("spaCy:")
        is_hf = str(row.model).startswith("HF:")

        if row.language == "ar" and is_spacy:
            if entity_label in {"PER", "MISC", "LOC", "ORG"}:
                # Arabic spaCy produced several odd short verbal phrases in Task 3.
                if entity_text.startswith("و") or len(entity_text.split()) <= 2:
                    reasons.append(
                        "Arabic spaCy candidate boundary/type issue"
                    )

        if is_hf and pd.notna(score) and float(score) < 0.75:
            reasons.append("Low-confidence HF entity")

        if len(entity_text) <= 2:
            reasons.append("Very short entity span")

        if reasons:
            rows.append({
                "text_id": row.text_id,
                "language": row.language,
                "category": row.category,
                "source": row.source,
                "model": row.model,
                "entity_text": entity_text,
                "entity_label": entity_label,
                "score": score,
                "review_reason": "; ".join(reasons),
            })

    failure_df = pd.DataFrame(rows)

    if failure_df.empty:
        return failure_df

    return failure_df.sort_values(
        ["language", "model", "text_id", "entity_text"]
    ).reset_index(drop=True)


def select_examples_for_notes(raw_entities_df, failure_candidates_df):
    """
    Select useful examples for the qualitative notes markdown.
    """
    examples = {}

    if raw_entities_df.empty:
        return examples

    work = raw_entities_df.copy()
    work["score_numeric"] = pd.to_numeric(work["score"], errors="coerce")

    hf_ar_good = work[
        (work["language"] == "ar")
        & (work["model"] == f"HF:{HF_MULTILINGUAL_MODEL}")
        & (work["entity_label"].isin(["LOC", "ORG"]))
        & (work["score_numeric"] >= 0.95)
    ].copy()

    hf_ar_good = hf_ar_good.drop_duplicates(
        subset=["entity_text", "entity_label"]
    ).head(6)

    examples["hf_ar_good"] = [
        f"{row.entity_text} ({row.entity_label})"
        for row in hf_ar_good.itertuples(index=False)
    ]

    if not failure_candidates_df.empty:
        spacy_ar_candidates = failure_candidates_df[
            (failure_candidates_df["language"] == "ar")
            & (failure_candidates_df["model"] == f"spaCy:{SPACY_MULTILINGUAL_MODEL}")
        ].copy()

        spacy_ar_candidates = spacy_ar_candidates.drop_duplicates(
            subset=["entity_text", "entity_label"]
        ).head(6)

        examples["spacy_ar_review"] = [
            f"{row.entity_text} ({row.entity_label})"
            for row in spacy_ar_candidates.itertuples(index=False)
        ]

        hf_review_candidates = failure_candidates_df[
            failure_candidates_df["model"] == f"HF:{HF_MULTILINGUAL_MODEL}"
        ].copy()

        hf_review_candidates = hf_review_candidates.drop_duplicates(
            subset=["entity_text", "entity_label"]
        ).head(6)

        examples["hf_review"] = [
            f"{row.entity_text} ({row.entity_label})"
            for row in hf_review_candidates.itertuples(index=False)
        ]

    en_examples = work[
        (work["language"] == "en")
        & (work["entity_label"].isin(["LOC", "ORG", "MISC"]))
    ].copy()

    en_examples = en_examples.drop_duplicates(
        subset=["entity_text", "entity_label"]
    ).head(6)

    examples["en_examples"] = [
        f"{row.entity_text} ({row.entity_label})"
        for row in en_examples.itertuples(index=False)
    ]

    return examples


def build_qualitative_notes(comparison_df, qualitative_review_df, failure_candidates_df, raw_entities_df):
    """
    Build a markdown note file summarizing qualitative findings.

    This file is a working draft for the final stretch_analysis.md.
    """
    examples = select_examples_for_notes(raw_entities_df, failure_candidates_df)

    def comparison_value(language, model_prefix, column):
        row = comparison_df[
            (comparison_df["language"] == language)
            & (comparison_df["model"].str.startswith(model_prefix))
        ]

        if row.empty:
            return "N/A"

        return row.iloc[0][column]

    ar_hf_total = comparison_value("ar", "HF:", "total_entities")
    ar_spacy_total = comparison_value("ar", "spaCy:", "total_entities")
    en_hf_total = comparison_value("en", "HF:", "total_entities")
    en_spacy_total = comparison_value("en", "spaCy:", "total_entities")

    ar_hf_density = comparison_value("ar", "HF:", "entities_per_100_words")
    ar_spacy_density = comparison_value("ar", "spaCy:", "entities_per_100_words")
    en_hf_density = comparison_value("en", "HF:", "entities_per_100_words")
    en_spacy_density = comparison_value("en", "spaCy:", "entities_per_100_words")

    ar_spacy_no_entity = comparison_value("ar", "spaCy:", "no_entity_rate")
    ar_hf_no_entity = comparison_value("ar", "HF:", "no_entity_rate")

    ar_coverage_gaps = qualitative_review_df[
        (qualitative_review_df["language"] == "ar")
        & (qualitative_review_df["spacy_entity_count"] == 0)
        & (qualitative_review_df["hf_entity_count"] > 0)
    ]

    coverage_gap_count = len(ar_coverage_gaps)

    notes = f"""# Multilingual NER Qualitative Notes

## Label schema

This analysis keeps the multilingual models' native labels: PER, LOC, ORG, and MISC. The Arabic side is qualitative because there is no Arabic gold-standard annotation file.

## Quantitative context

- Arabic + HF total entities: {ar_hf_total}
- Arabic + spaCy total entities: {ar_spacy_total}
- English + HF total entities: {en_hf_total}
- English + spaCy total entities: {en_spacy_total}
- Arabic + HF entity density per 100 words: {ar_hf_density}
- Arabic + spaCy entity density per 100 words: {ar_spacy_density}
- English + HF entity density per 100 words: {en_hf_density}
- English + spaCy entity density per 100 words: {en_spacy_density}
- Arabic + HF no-entity rate: {ar_hf_no_entity}
- Arabic + spaCy no-entity rate: {ar_spacy_no_entity}
- Arabic texts where HF found entities but spaCy found none: {coverage_gap_count}

## Strong Arabic HF examples

{chr(10).join(f"- {example}" for example in examples.get("hf_ar_good", ["No examples found"]))}

## Arabic spaCy review examples

These are candidates for false positives or boundary/type errors:

{chr(10).join(f"- {example}" for example in examples.get("spacy_ar_review", ["No examples found"]))}

## HF review examples

These are not automatically wrong, but they should be manually inspected because they are low-confidence or very short spans:

{chr(10).join(f"- {example}" for example in examples.get("hf_review", ["No examples found"]))}

## English examples

{chr(10).join(f"- {example}" for example in examples.get("en_examples", ["No examples found"]))}

## Draft interpretation

The English results are relatively stable across both multilingual models. spaCy extracted slightly more English entities, while HF produced a similar number and focused mostly on LOC and ORG labels.

The Arabic results show a much larger model gap. HF extracted substantially more Arabic entities and found at least one entity in every Arabic text. spaCy extracted far fewer Arabic entities and left several Arabic texts with no entities. Qualitatively, HF captured useful Arabic locations and organizations, while spaCy produced several suspicious Arabic spans that look like verbs or ordinary phrases rather than named entities. This suggests that Arabic entity boundary detection and organization recognition are harder, and that HF is more suitable for this bilingual climate dataset.
"""

    return notes


def save_qualitative_outputs(
    qualitative_review_df,
    failure_candidates_df,
    qualitative_notes,
    qualitative_review_output_path=QUALITATIVE_REVIEW_OUTPUT_PATH,
    failure_candidates_output_path=FAILURE_CANDIDATES_OUTPUT_PATH,
    qualitative_notes_output_path=QUALITATIVE_NOTES_OUTPUT_PATH,
):
    """Save Task 5 qualitative analysis outputs."""
    qualitative_review_output_path.parent.mkdir(parents=True, exist_ok=True)

    qualitative_review_df.to_csv(
        qualitative_review_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    failure_candidates_df.to_csv(
        failure_candidates_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    qualitative_notes_output_path.write_text(
        qualitative_notes,
        encoding="utf-8",
    )

    print(f"\nSaved qualitative review to: {qualitative_review_output_path}")
    print(f"Saved failure candidates to: {failure_candidates_output_path}")
    print(f"Saved qualitative notes to: {qualitative_notes_output_path}")


def print_qualitative_summary(qualitative_review_df, failure_candidates_df):
    """Print a compact qualitative summary."""
    print("\n=== Task 5: Qualitative Review ===")

    print("\nReview rows:")
    print(len(qualitative_review_df))

    print("\nQualitative note counts:")
    print(
        qualitative_review_df["qualitative_note"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "qualitative_note", "count": "count"})
        .to_string(index=False)
    )

    ar_gap_rows = qualitative_review_df[
        (qualitative_review_df["language"] == "ar")
        & (qualitative_review_df["spacy_entity_count"] == 0)
        & (qualitative_review_df["hf_entity_count"] > 0)
    ]

    print("\nArabic texts where HF found entities but spaCy found none:")
    if ar_gap_rows.empty:
        print("None")
    else:
        print(
            ar_gap_rows[
                [
                    "text_id",
                    "category",
                    "source",
                    "hf_entity_count",
                    "hf_entities",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("\nFailure/review candidates:")
    if failure_candidates_df.empty:
        print("None")
    else:
        print(
            failure_candidates_df[
                [
                    "text_id",
                    "language",
                    "model",
                    "entity_text",
                    "entity_label",
                    "score",
                    "review_reason",
                ]
            ]
            .head(20)
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

    comparison_df, label_counts_df, example_entities_df = build_comparison_table(
        raw_entities_df=raw_entities_df,
        document_counts_df=document_counts_df,
        n_examples=3,
    )

    print_comparison_table(comparison_df)

    save_comparison_outputs(
        comparison_df=comparison_df,
        label_counts_df=label_counts_df,
        example_entities_df=example_entities_df,
    )



    qualitative_review_df = build_qualitative_review(
        sample_df=sample_df,
        raw_entities_df=raw_entities_df,
    )

    failure_candidates_df = build_failure_candidates(raw_entities_df)

    qualitative_notes = build_qualitative_notes(
        comparison_df=comparison_df,
        qualitative_review_df=qualitative_review_df,
        failure_candidates_df=failure_candidates_df,
        raw_entities_df=raw_entities_df,
    )

    print_qualitative_summary(
        qualitative_review_df=qualitative_review_df,
        failure_candidates_df=failure_candidates_df,
    )

    save_qualitative_outputs(
        qualitative_review_df=qualitative_review_df,
        failure_candidates_df=failure_candidates_df,
        qualitative_notes=qualitative_notes,
    )

if __name__ == "__main__":
    main()