"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df["text"].fillna("").astype(str).str.split().str.len()

    summary = {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": float(word_counts.mean()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }

    return summary



def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    if pd.isna(text):
        return []

    normalized_text = unicodedata.normalize("NFC", str(text))

    doc = nlp(normalized_text)

    cleaned_tokens = []

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        lemma = token.lemma_.lower().strip()

        if lemma:
            cleaned_tokens.append(lemma)

    return cleaned_tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    columns = ["text_id", "entity_text", "entity_label", "start_char", "end_char"]

    english_df = df[df["language"] == "en"].copy()

    rows = []

    texts = english_df["text"].fillna("").astype(str).tolist()
    text_ids = english_df["id"].tolist()

    for text_id, doc in zip(text_ids, nlp.pipe(texts)):
        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows, columns=columns)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    columns = ["text_id", "entity_text", "entity_label", "start_char", "end_char"]

    english_df = df[df["language"] == "en"].copy()
    rows = []

    def strip_iob_prefix(label):
        """Convert B-ORG / I-ORG to ORG."""
        if isinstance(label, str) and "-" in label:
            return label.split("-", 1)[1]
        return label

    def get_iob_prefix(label):
        """Return B or I from labels like B-ORG / I-ORG."""
        if isinstance(label, str) and "-" in label:
            return label.split("-", 1)[0]
        return ""

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = "" if pd.isna(row["text"]) else str(row["text"])

        raw_entities = ner_pipeline(text)

        current_entity = None

        for ent in raw_entities:
            raw_label = ent.get("entity", ent.get("entity_group", ""))
            label = strip_iob_prefix(raw_label)
            prefix = get_iob_prefix(raw_label)

            word = ent.get("word", "")
            start = ent.get("start")
            end = ent.get("end")

            if start is None or end is None:
                continue

            start = int(start)
            end = int(end)

            is_subword = word.startswith("##")

            should_merge = (
                current_entity is not None
                and current_entity["entity_label"] == label
                and (prefix == "I" or is_subword)
            )

            if should_merge:
                current_entity["end_char"] = end
                current_entity["entity_text"] = text[
                    current_entity["start_char"]:current_entity["end_char"]
                ]
            else:
                if current_entity is not None:
                    rows.append(current_entity)

                current_entity = {
                    "text_id": text_id,
                    "entity_text": text[start:end],
                    "entity_label": label,
                    "start_char": start,
                    "end_char": end,
                }

        if current_entity is not None:
            rows.append(current_entity)

    return pd.DataFrame(rows, columns=columns)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    total_spacy = len(spacy_df)
    total_hf = len(hf_df)

    spacy_pairs = set(
        zip(
            spacy_df["text_id"],
            spacy_df["entity_text"].astype(str).str.strip()
        )
    )

    hf_pairs = set(
        zip(
            hf_df["text_id"],
            hf_df["entity_text"].astype(str).str.strip()
        )
    )

    both = spacy_pairs & hf_pairs
    spacy_only = spacy_pairs - hf_pairs
    hf_only = hf_pairs - spacy_pairs

    comparison = {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": total_spacy,
        "total_hf": total_hf,
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only,
    }

    print("\nNER Output Comparison")
    print("---------------------")
    print(f"Total spaCy entities: {total_spacy}")
    print(f"Total HF entities: {total_hf}")

    print("\nspaCy counts by label:")
    print(spacy_counts)

    print("\nHF counts by label:")
    print(hf_counts)

    print("\nOverlap summary:")
    print(f"Both systems: {len(both)}")
    print(f"spaCy only: {len(spacy_only)}")
    print(f"HF only: {len(hf_only)}")

    return comparison


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    match_cols = ["text_id", "entity_text", "entity_label"]

    gold_eval = gold_df[match_cols].copy()

    gold_text_ids = set(gold_eval["text_id"].unique())

    pred_eval = predicted_df[
        predicted_df["text_id"].isin(gold_text_ids)
    ][match_cols].copy()

    for col in ["entity_text", "entity_label"]:
        gold_eval[col] = gold_eval[col].astype(str).str.strip()
        pred_eval[col] = pred_eval[col].astype(str).str.strip()

    gold_counter = Counter(
        tuple(row) for row in gold_eval[match_cols].itertuples(index=False, name=None)
    )

    pred_counter = Counter(
        tuple(row) for row in pred_eval[match_cols].itertuples(index=False, name=None)
    )

    true_positives = sum((pred_counter & gold_counter).values())
    predicted_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())

    precision = true_positives / predicted_total if predicted_total > 0 else 0.0
    recall = true_positives / gold_total if gold_total > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }




# ============================================================
# Challenge Extensions
# ============================================================

def entity_counts_by_category(entities_df, articles_df):
    """Tier 1: Count entity labels per article category.

    Args:
        entities_df: DataFrame with text_id, entity_text, entity_label, ...
        articles_df: Original articles DataFrame with id and category.

    Returns:
        Pivot table: rows = category, columns = entity_label, values = counts.
    """
    merged = entities_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    counts = (
        merged
        .groupby(["category", "entity_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    return counts


def evaluate_ner_by_category(predicted_df, gold_df, articles_df):
    """Tier 1: Evaluate NER precision/recall/F1 separately per category.

    This uses the existing evaluate_ner() function from Task 6.
    """
    gold_with_category = gold_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    results = {}

    for category, group in gold_with_category.groupby("category"):
        text_ids = set(group["text_id"])

        gold_category = gold_df[gold_df["text_id"].isin(text_ids)].copy()
        pred_category = predicted_df[predicted_df["text_id"].isin(text_ids)].copy()

        metrics = evaluate_ner(pred_category, gold_category)

        metrics["gold_entities"] = int(len(gold_category))
        metrics["predicted_entities"] = int(len(pred_category))
        metrics["gold_texts"] = int(len(text_ids))

        results[category] = metrics

    return pd.DataFrame.from_dict(results, orient="index").sort_index()


def plot_entity_counts_by_category(counts_df, output_path="outputs/entity_counts_by_category.png"):
    """Tier 1: Create a heatmap of entity counts by category.

    Args:
        counts_df: Output of entity_counts_by_category().
        output_path: Where to save the chart.

    Returns:
        Path to saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(counts_df.values)

    ax.set_xticks(range(len(counts_df.columns)))
    ax.set_xticklabels(counts_df.columns, rotation=45, ha="right")

    ax.set_yticks(range(len(counts_df.index)))
    ax.set_yticklabels(counts_df.index)

    ax.set_title("Entity Counts by Category")
    ax.set_xlabel("Entity Label")
    ax.set_ylabel("Category")

    for i in range(len(counts_df.index)):
        for j in range(len(counts_df.columns)):
            value = counts_df.iloc[i, j]
            ax.text(j, i, str(value), ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return str(output_path)


def analyze_category_difficulty(category_metrics_df):
    """Tier 1: Generate a short interpretation of category difficulty."""
    if category_metrics_df.empty:
        return "No category-level metrics were available."

    best_category = category_metrics_df["f1"].idxmax()
    worst_category = category_metrics_df["f1"].idxmin()

    best_f1 = category_metrics_df.loc[best_category, "f1"]
    worst_f1 = category_metrics_df.loc[worst_category, "f1"]

    return (
        f"The easiest category appears to be '{best_category}' with F1={best_f1:.3f}, "
        f"while the hardest appears to be '{worst_category}' with F1={worst_f1:.3f}. "
        "Categories with more clear named entities such as organizations, countries, dates, "
        "and named locations are usually easier for NER. Categories with more descriptive "
        "or scientific language can be harder because important concepts may not be written "
        "as standard named entities."
    )


# ============================================================
# Tier 2: Custom Entity Aggregation Pipeline
# ============================================================

DEFAULT_ENTITY_NORMALIZATION = {
    "un": "United Nations",
    "u.n.": "United Nations",
    "united nations": "United Nations",

    "ipcc": "IPCC",
    "intergovernmental panel on climate change": "IPCC",

    "unep": "UNEP",
    "un environment programme": "UNEP",
    "united nations environment programme": "UNEP",

    "cop28": "COP28",
    "cop 28": "COP28",

    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",

    "us": "United States",
    "u.s.": "United States",
    "usa": "United States",
    "united states": "United States",

    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
}


def _normalization_key(text):
    """Create a simple normalized key for entity matching."""
    text = "" if pd.isna(text) else str(text)
    text = text.strip().lower()
    text = re.sub(r"\.", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_entity_text(entity_text, mapping=None):
    """Normalize entity variants into a canonical name.

    Example:
        U.N. -> United Nations
        UN -> United Nations
        IPCC -> IPCC
    """
    if mapping is None:
        mapping = DEFAULT_ENTITY_NORMALIZATION

    normalized_mapping = {
        _normalization_key(key): value
        for key, value in mapping.items()
    }

    key = _normalization_key(entity_text)
    original = "" if pd.isna(entity_text) else str(entity_text).strip()

    return normalized_mapping.get(key, original)


def apply_entity_normalization(entities_df, mapping=None):
    """Add a canonical_entity column to an entities DataFrame."""
    normalized_df = entities_df.copy()

    normalized_df["canonical_entity"] = normalized_df["entity_text"].apply(
        lambda text: normalize_entity_text(text, mapping=mapping)
    )

    return normalized_df


def compute_entity_cooccurrence(entities_df, entity_col="canonical_entity"):
    """Compute entity co-occurrence pairs.

    Entities co-occur if they appear in the same text_id.
    """
    if entity_col not in entities_df.columns:
        entity_col = "entity_text"

    pair_counts = Counter()

    for text_id, group in entities_df.groupby("text_id"):
        entities = (
            group[entity_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_entities = sorted(set(e for e in entities if e))

        for entity_a, entity_b in combinations(unique_entities, 2):
            pair_counts[(entity_a, entity_b)] += 1

    rows = [
        {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "cooccurrence_count": count,
        }
        for (entity_a, entity_b), count in pair_counts.items()
    ]

    edge_df = pd.DataFrame(rows)

    if edge_df.empty:
        return pd.DataFrame(columns=["entity_a", "entity_b", "cooccurrence_count"])

    return edge_df.sort_values("cooccurrence_count", ascending=False).reset_index(drop=True)


def cooccurrence_adjacency_matrix(edge_df):
    """Convert edge list into an adjacency matrix."""
    if edge_df.empty:
        return pd.DataFrame()

    entities = sorted(set(edge_df["entity_a"]) | set(edge_df["entity_b"]))

    matrix = pd.DataFrame(0, index=entities, columns=entities)

    for _, row in edge_df.iterrows():
        entity_a = row["entity_a"]
        entity_b = row["entity_b"]
        count = row["cooccurrence_count"]

        matrix.loc[entity_a, entity_b] = count
        matrix.loc[entity_b, entity_a] = count

    return matrix


def entity_tfidf_by_category(entities_df, articles_df, entity_col="canonical_entity"):
    """Compute TF-IDF-style entity importance by category.

    TF = entity count inside a category.
    IDF = higher when entity appears in fewer categories.
    Score = TF * IDF.
    """
    if entity_col not in entities_df.columns:
        entity_col = "entity_text"

    merged = entities_df.merge(
        articles_df[["id", "category"]],
        left_on="text_id",
        right_on="id",
        how="left"
    )

    merged = merged.dropna(subset=["category", entity_col]).copy()
    merged[entity_col] = merged[entity_col].astype(str).str.strip()
    merged = merged[merged[entity_col] != ""]

    counts = (
        merged
        .groupby(["category", entity_col])
        .size()
        .reset_index(name="tf")
    )

    n_categories = merged["category"].nunique()

    entity_category_df = (
        counts
        .groupby(entity_col)["category"]
        .nunique()
        .to_dict()
    )

    counts["category_document_frequency"] = counts[entity_col].map(entity_category_df)

    counts["idf"] = counts["category_document_frequency"].apply(
        lambda df_count: math.log((1 + n_categories) / (1 + df_count)) + 1
    )

    counts["tfidf_score"] = counts["tf"] * counts["idf"]

    return counts.sort_values(
        ["category", "tfidf_score"],
        ascending=[True, False]
    ).reset_index(drop=True)


def plot_entity_network(edge_df, output_path="outputs/entity_network.png", top_n=20):
    """Plot a simple network visualization using matplotlib only."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_edges = edge_df.head(top_n).copy()

    if top_edges.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No co-occurrences found", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return str(output_path)

    nodes = sorted(set(top_edges["entity_a"]) | set(top_edges["entity_b"]))

    positions = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / len(nodes)
        positions[node] = (math.cos(angle), math.sin(angle))

    fig, ax = plt.subplots(figsize=(10, 10))

    max_count = top_edges["cooccurrence_count"].max()

    for _, row in top_edges.iterrows():
        entity_a = row["entity_a"]
        entity_b = row["entity_b"]
        count = row["cooccurrence_count"]

        x1, y1 = positions[entity_a]
        x2, y2 = positions[entity_b]

        width = 1 + (count / max_count) * 4

        ax.plot([x1, x2], [y1, y2], linewidth=width, alpha=0.6)

    for node, (x, y) in positions.items():
        ax.scatter(x, y, s=400)
        ax.text(x, y, node, ha="center", va="center", fontsize=8)

    ax.set_title(f"Top {top_n} Entity Co-occurrences")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return str(output_path)


# ============================================================
# Tier 3: Custom NER Evaluator
# ============================================================

def _safe_text(value):
    return "" if pd.isna(value) else str(value).strip()


def _safe_int_or_none(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _records_from_df(df):
    """Convert entity DataFrame rows into normalized dictionaries."""
    records = []

    for _, row in df.iterrows():
        record = {
            "text_id": row["text_id"],
            "entity_text": _safe_text(row["entity_text"]),
            "entity_label": _safe_text(row["entity_label"]),
            "start_char": _safe_int_or_none(row["start_char"]) if "start_char" in df.columns else None,
            "end_char": _safe_int_or_none(row["end_char"]) if "end_char" in df.columns else None,
        }
        records.append(record)

    return records


def _has_valid_span(record):
    return (
        record["start_char"] is not None
        and record["end_char"] is not None
        and record["end_char"] >= record["start_char"]
    )


def _span_overlap_score(pred_record, gold_record):
    """Return IoU-style overlap score between two spans."""
    if not _has_valid_span(pred_record) or not _has_valid_span(gold_record):
        return 0.0

    start = max(pred_record["start_char"], gold_record["start_char"])
    end = min(pred_record["end_char"], gold_record["end_char"])

    overlap = max(0, end - start)

    if overlap == 0:
        return 0.0

    union_start = min(pred_record["start_char"], gold_record["start_char"])
    union_end = max(pred_record["end_char"], gold_record["end_char"])
    union = max(1, union_end - union_start)

    return overlap / union


def _precision_recall_f1(tp_score, predicted_total, gold_total):
    precision = tp_score / predicted_total if predicted_total > 0 else 0.0
    recall = tp_score / gold_total if gold_total > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp_score": float(tp_score),
        "predicted_total": int(predicted_total),
        "gold_total": int(gold_total),
    }


def _score_records(pred_records, gold_records, strategy):
    """Score records using exact, partial, or type_agnostic matching."""
    predicted_total = len(pred_records)
    gold_total = len(gold_records)

    if strategy == "exact":
        pred_counter = Counter(
            (
                r["text_id"],
                r["entity_text"],
                r["entity_label"],
            )
            for r in pred_records
        )

        gold_counter = Counter(
            (
                r["text_id"],
                r["entity_text"],
                r["entity_label"],
            )
            for r in gold_records
        )

        tp_score = sum((pred_counter & gold_counter).values())
        return _precision_recall_f1(tp_score, predicted_total, gold_total)

    if strategy == "type_agnostic":
        def span_or_text_key(record):
            if _has_valid_span(record):
                return (
                    record["text_id"],
                    record["start_char"],
                    record["end_char"],
                )

            return (
                record["text_id"],
                record["entity_text"],
            )

        pred_counter = Counter(span_or_text_key(r) for r in pred_records)
        gold_counter = Counter(span_or_text_key(r) for r in gold_records)

        tp_score = sum((pred_counter & gold_counter).values())
        return _precision_recall_f1(tp_score, predicted_total, gold_total)

    if strategy == "partial":
        candidates = []

        for pred_idx, pred_record in enumerate(pred_records):
            for gold_idx, gold_record in enumerate(gold_records):
                if pred_record["text_id"] != gold_record["text_id"]:
                    continue

                if pred_record["entity_label"] != gold_record["entity_label"]:
                    continue

                score = _span_overlap_score(pred_record, gold_record)

                if score > 0:
                    candidates.append((score, pred_idx, gold_idx))

        candidates.sort(reverse=True)

        matched_predictions = set()
        matched_gold = set()
        tp_score = 0.0

        for score, pred_idx, gold_idx in candidates:
            if pred_idx in matched_predictions or gold_idx in matched_gold:
                continue

            matched_predictions.add(pred_idx)
            matched_gold.add(gold_idx)
            tp_score += score

        return _precision_recall_f1(tp_score, predicted_total, gold_total)

    raise ValueError(f"Unknown matching strategy: {strategy}")


def _macro_score(pred_records, gold_records, strategy):
    """Average metrics per text_id."""
    text_ids = sorted(
        set(r["text_id"] for r in pred_records)
        | set(r["text_id"] for r in gold_records)
    )

    if not text_ids:
        return _precision_recall_f1(0, 0, 0)

    per_text_metrics = []

    for text_id in text_ids:
        pred_text = [r for r in pred_records if r["text_id"] == text_id]
        gold_text = [r for r in gold_records if r["text_id"] == text_id]

        per_text_metrics.append(_score_records(pred_text, gold_text, strategy))

    return {
        "precision": float(np.mean([m["precision"] for m in per_text_metrics])),
        "recall": float(np.mean([m["recall"] for m in per_text_metrics])),
        "f1": float(np.mean([m["f1"] for m in per_text_metrics])),
        "texts_evaluated": int(len(text_ids)),
    }


def ner_error_analysis(predicted_df, gold_df):
    """Categorize errors into type, boundary, missing, and spurious errors."""
    pred_records = _records_from_df(predicted_df)
    gold_records = _records_from_df(gold_df)

    matched_pred = set()
    matched_gold = set()

    errors = []
    error_counts = Counter()

    # 1. Exact matches: not errors
    for pred_idx, pred_record in enumerate(pred_records):
        for gold_idx, gold_record in enumerate(gold_records):
            if gold_idx in matched_gold:
                continue

            same_text = pred_record["text_id"] == gold_record["text_id"]
            same_entity_text = pred_record["entity_text"] == gold_record["entity_text"]
            same_label = pred_record["entity_label"] == gold_record["entity_label"]

            if same_text and same_entity_text and same_label:
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)
                break

    # 2. Type errors: same span/text, wrong label
    for pred_idx, pred_record in enumerate(pred_records):
        if pred_idx in matched_pred:
            continue

        for gold_idx, gold_record in enumerate(gold_records):
            if gold_idx in matched_gold:
                continue

            same_text_id = pred_record["text_id"] == gold_record["text_id"]

            if not same_text_id:
                continue

            same_span = (
                _has_valid_span(pred_record)
                and _has_valid_span(gold_record)
                and pred_record["start_char"] == gold_record["start_char"]
                and pred_record["end_char"] == gold_record["end_char"]
            )

            same_entity_text = pred_record["entity_text"] == gold_record["entity_text"]
            different_label = pred_record["entity_label"] != gold_record["entity_label"]

            if (same_span or same_entity_text) and different_label:
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)

                error_counts["type_error"] += 1
                errors.append({
                    "error_type": "type_error",
                    "text_id": pred_record["text_id"],
                    "predicted": pred_record,
                    "gold": gold_record,
                })
                break

    # 3. Boundary errors: same label, overlapping span, wrong boundary
    for pred_idx, pred_record in enumerate(pred_records):
        if pred_idx in matched_pred:
            continue

        for gold_idx, gold_record in enumerate(gold_records):
            if gold_idx in matched_gold:
                continue

            same_text_id = pred_record["text_id"] == gold_record["text_id"]
            same_label = pred_record["entity_label"] == gold_record["entity_label"]
            overlap_score = _span_overlap_score(pred_record, gold_record)

            same_exact_span = (
                _has_valid_span(pred_record)
                and _has_valid_span(gold_record)
                and pred_record["start_char"] == gold_record["start_char"]
                and pred_record["end_char"] == gold_record["end_char"]
            )

            if same_text_id and same_label and overlap_score > 0 and not same_exact_span:
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)

                error_counts["boundary_error"] += 1
                errors.append({
                    "error_type": "boundary_error",
                    "text_id": pred_record["text_id"],
                    "predicted": pred_record,
                    "gold": gold_record,
                    "overlap_score": overlap_score,
                })
                break

    # 4. Remaining gold entities are missing
    for gold_idx, gold_record in enumerate(gold_records):
        if gold_idx not in matched_gold:
            error_counts["missing_entity"] += 1
            errors.append({
                "error_type": "missing_entity",
                "text_id": gold_record["text_id"],
                "predicted": None,
                "gold": gold_record,
            })

    # 5. Remaining predicted entities are spurious
    for pred_idx, pred_record in enumerate(pred_records):
        if pred_idx not in matched_pred:
            error_counts["spurious_entity"] += 1
            errors.append({
                "error_type": "spurious_entity",
                "text_id": pred_record["text_id"],
                "predicted": pred_record,
                "gold": None,
            })

    return {
        "error_counts": dict(error_counts),
        "errors": errors,
    }


def evaluate_ner_comprehensive(predicted_df, gold_df):
    """Tier 3: Comprehensive NER evaluation.

    Strategies:
        exact: text + label must match.
        partial: overlapping span with same label gets partial credit.
        type_agnostic: span/text matches, label ignored.

    Returns:
        Dictionary with micro metrics, macro metrics, and error report.
    """
    gold_text_ids = set(gold_df["text_id"].unique())
    predicted_eval = predicted_df[predicted_df["text_id"].isin(gold_text_ids)].copy()
    gold_eval = gold_df.copy()

    pred_records = _records_from_df(predicted_eval)
    gold_records = _records_from_df(gold_eval)

    strategies = ["exact", "partial", "type_agnostic"]

    micro = {
        strategy: _score_records(pred_records, gold_records, strategy)
        for strategy in strategies
    }

    macro = {
        strategy: _macro_score(pred_records, gold_records, strategy)
        for strategy in strategies
    }

    errors = ner_error_analysis(predicted_df, gold_df)

    return {
        "micro": micro,
        "macro": macro,
        "errors": errors,
    }




if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            if spacy_metrics is not None:
                print(f"\nspaCy evaluation: {spacy_metrics}")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation: {hf_metrics}")


        # ------------------------------------------------------------
        # Challenge Extensions
        # ------------------------------------------------------------

        print("\n=== Challenge Tier 1: Per-Category NER Analysis ===")

        spacy_category_counts = entity_counts_by_category(spacy_entities, df)
        print("\nspaCy entity counts by category:")
        print(spacy_category_counts)

        chart_path = plot_entity_counts_by_category(
            spacy_category_counts,
            output_path="outputs/spacy_entity_counts_by_category.png"
        )
        print(f"\nSaved category heatmap to: {chart_path}")

        spacy_category_metrics = evaluate_ner_by_category(spacy_entities, gold, df)
        print("\nspaCy category-level metrics:")
        print(spacy_category_metrics)

        print("\nCategory analysis:")
        print(analyze_category_difficulty(spacy_category_metrics))


        print("\n=== Challenge Tier 2: Entity Aggregation Pipeline ===")

        normalized_spacy = apply_entity_normalization(spacy_entities)

        edge_df = compute_entity_cooccurrence(normalized_spacy)
        print("\nTop entity co-occurrences:")
        print(edge_df.head(20))

        adjacency = cooccurrence_adjacency_matrix(edge_df)
        print("\nCo-occurrence adjacency matrix sample:")
        print(adjacency.iloc[:10, :10])

        tfidf_entities = entity_tfidf_by_category(normalized_spacy, df)
        print("\nTop TF-IDF-style entities by category:")
        print(tfidf_entities.groupby("category").head(5))

        network_path = plot_entity_network(
            edge_df,
            output_path="outputs/spacy_entity_network.png",
            top_n=20
        )
        print(f"\nSaved entity network to: {network_path}")


        print("\n=== Challenge Tier 3: Custom NER Evaluator ===")

        comprehensive_spacy = evaluate_ner_comprehensive(spacy_entities, gold)

        print("\nspaCy comprehensive micro metrics:")
        print(comprehensive_spacy["micro"])

        print("\nspaCy comprehensive macro metrics:")
        print(comprehensive_spacy["macro"])

        print("\nspaCy error counts:")
        print(comprehensive_spacy["errors"]["error_counts"])