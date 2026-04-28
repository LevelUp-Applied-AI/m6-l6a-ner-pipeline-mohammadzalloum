"""
Module 6 Week A — Stretch: Custom NER Rules

This script:
1. Loads the climate articles dataset.
2. Defines domain-specific climate EntityRuler patterns.
3. Runs three NER systems:
   - base spaCy
   - EntityRuler before NER
   - EntityRuler after NER
4. Compares entity counts.
5. Evaluates standard-label entities against the gold standard.
6. Saves output CSV files for analysis.

Run:
    python stretch_custom_ner.py | tee outputs/stretch_custom_ner_results.txt
"""

from collections import Counter
from pathlib import Path
import re

import pandas as pd
import spacy


DATA_PATH = "data/climate_articles.csv"
GOLD_PATH = "data/gold_entities.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


STANDARD_LABELS = {
    "ORG",
    "GPE",
    "DATE",
    "LAW",
    "MONEY",
    "PERSON",
    "QUANTITY",
    "LOC",
    "EVENT",
    "WORK_OF_ART",
}


CUSTOM_PATTERNS = [
    # Climate events / conferences
    {"label": "CLIMATE_EVENT", "pattern": "COP28"},
    {"label": "CLIMATE_EVENT", "pattern": "COP27"},
    {"label": "CLIMATE_EVENT", "pattern": "COP26"},
    {"label": "CLIMATE_EVENT", "pattern": "Bonn Climate Change Conference"},
    {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},

    # Agreements / treaties / pledges
    {"label": "AGREEMENT", "pattern": "Paris Agreement"},
    {"label": "AGREEMENT", "pattern": "Kigali Amendment"},
    {"label": "AGREEMENT", "pattern": "Montreal Protocol"},
    {"label": "AGREEMENT", "pattern": "Global Methane Pledge"},

    # Reports / assessments
    {"label": "REPORT", "pattern": "Sixth Assessment Report"},
    {"label": "REPORT", "pattern": "Emissions Gap Report"},
    {"label": "REPORT", "pattern": "Adaptation Gap Report"},
    {"label": "REPORT", "pattern": "Global Methane Assessment"},

    # Climate thresholds / targets
    {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},
    {"label": "THRESHOLD", "pattern": "2 degrees"},
    {"label": "THRESHOLD", "pattern": "2°C target"},

    # Policies / mechanisms / funds
    {"label": "POLICY", "pattern": "Nationally Determined Contribution"},
    {"label": "POLICY", "pattern": "Nationally Determined Contributions"},
    {"label": "POLICY", "pattern": "NDCs"},
    {"label": "POLICY_MECHANISM", "pattern": "Carbon Border Adjustment Mechanism"},
    {"label": "CLIMATE_FUND", "pattern": "Loss and Damage Fund"},
    {"label": "CLIMATE_FUND", "pattern": "Green Climate Fund"},
]


CUSTOM_LABELS = sorted({pattern["label"] for pattern in CUSTOM_PATTERNS})


def load_articles(filepath=DATA_PATH):
    """Load climate articles data."""
    return pd.read_csv(filepath)


def find_pattern_examples(df, patterns, max_examples=3):
    """Find example rows where each custom phrase pattern appears."""
    rows = []
    english_df = df[df["language"] == "en"].copy()

    for pattern in patterns:
        label = pattern["label"]
        text_pattern = pattern["pattern"]

        regex = re.compile(re.escape(text_pattern), flags=re.IGNORECASE)

        matches = english_df[
            english_df["text"].astype(str).str.contains(regex, na=False)
        ]

        rows.append({
            "label": label,
            "pattern": text_pattern,
            "match_count": int(len(matches)),
            "example_text_ids": list(matches["id"].head(max_examples)),
        })

    return pd.DataFrame(rows)


def build_spacy_pipeline(mode):
    """Build one of three spaCy pipelines.

    Args:
        mode:
            "base"   -> spaCy only
            "before" -> EntityRuler before NER
            "after"  -> EntityRuler after NER
    """
    nlp = spacy.load("en_core_web_sm")

    if mode == "base":
        return nlp

    if mode == "before":
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={
                "overwrite_ents": True,
                "phrase_matcher_attr": "LOWER",
            },
        )
        ruler.add_patterns(CUSTOM_PATTERNS)
        return nlp

    if mode == "after":
        ruler = nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={
                "overwrite_ents": False,
                "phrase_matcher_attr": "LOWER",
            },
        )
        ruler.add_patterns(CUSTOM_PATTERNS)
        return nlp

    raise ValueError(f"Unknown pipeline mode: {mode}")


def extract_entities(df, nlp):
    """Extract entities from English texts using the provided spaCy pipeline."""
    columns = [
        "text_id",
        "category",
        "entity_text",
        "entity_label",
        "start_char",
        "end_char",
    ]

    english_df = df[df["language"] == "en"].copy()

    records = list(
        zip(
            english_df["id"].tolist(),
            english_df["category"].tolist(),
            english_df["text"].fillna("").astype(str).tolist(),
        )
    )

    texts = [record[2] for record in records]
    rows = []

    for (text_id, category, _), doc in zip(records, nlp.pipe(texts)):
        for ent in doc.ents:
            rows.append({
                "text_id": text_id,
                "category": category,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

    return pd.DataFrame(rows, columns=columns)


def label_counts(system_entities):
    """Create a label count table for all systems."""
    rows = []

    for system_name, entities_df in system_entities.items():
        counts = entities_df["entity_label"].value_counts().to_dict()

        for label, count in counts.items():
            rows.append({
                "system": system_name,
                "entity_label": label,
                "count": int(count),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["system", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def evaluate_standard_labels(predicted_df, gold_df):
    """Evaluate only standard-label entities against the gold standard.

    Custom labels are excluded because the gold file only contains standard labels.
    """
    match_cols = ["text_id", "entity_text", "entity_label"]

    gold_eval = gold_df[gold_df["entity_label"].isin(STANDARD_LABELS)][match_cols].copy()

    gold_text_ids = set(gold_eval["text_id"].unique())

    pred_eval = predicted_df[
        predicted_df["text_id"].isin(gold_text_ids)
        & predicted_df["entity_label"].isin(STANDARD_LABELS)
    ][match_cols].copy()

    for col in ["entity_text", "entity_label"]:
        gold_eval[col] = gold_eval[col].astype(str).str.strip()
        pred_eval[col] = pred_eval[col].astype(str).str.strip()

    gold_counter = Counter(
        tuple(row) for row in gold_eval.itertuples(index=False, name=None)
    )

    pred_counter = Counter(
        tuple(row) for row in pred_eval.itertuples(index=False, name=None)
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
        "true_positives": int(true_positives),
        "predicted_standard_entities": int(predicted_total),
        "gold_standard_entities": int(gold_total),
    }


def evaluate_all_systems(system_entities, gold_df):
    """Evaluate standard labels for every NER system."""
    rows = []

    base_f1 = None

    for system_name, entities_df in system_entities.items():
        metrics = evaluate_standard_labels(entities_df, gold_df)

        if system_name == "base_spacy":
            base_f1 = metrics["f1"]

        rows.append({
            "system": system_name,
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)

    if base_f1 is not None:
        metrics_df["f1_delta_vs_base"] = metrics_df["f1"] - base_f1
        metrics_df["precision_delta_vs_base"] = metrics_df["precision"] - metrics_df.loc[
            metrics_df["system"] == "base_spacy", "precision"
        ].iloc[0]
        metrics_df["recall_delta_vs_base"] = metrics_df["recall"] - metrics_df.loc[
            metrics_df["system"] == "base_spacy", "recall"
        ].iloc[0]

    return metrics_df


def get_custom_rule_examples(system_entities, df, max_per_system_label=5):
    """Collect examples where custom labels fired."""
    article_lookup = df.set_index("id")[["text", "category"]].to_dict(orient="index")

    rows = []

    for system_name, entities_df in system_entities.items():
        custom_df = entities_df[
            entities_df["entity_label"].isin(CUSTOM_LABELS)
        ].copy()

        if custom_df.empty:
            continue

        grouped = custom_df.groupby("entity_label")

        for label, group in grouped:
            for _, row in group.head(max_per_system_label).iterrows():
                text_id = row["text_id"]
                article = article_lookup.get(text_id, {})
                text = str(article.get("text", ""))

                start = int(row["start_char"])
                end = int(row["end_char"])

                snippet_start = max(0, start - 90)
                snippet_end = min(len(text), end + 130)
                snippet = text[snippet_start:snippet_end].replace("\n", " ")

                rows.append({
                    "system": system_name,
                    "text_id": text_id,
                    "category": article.get("category", row.get("category", "")),
                    "entity_text": row["entity_text"],
                    "entity_label": row["entity_label"],
                    "start_char": start,
                    "end_char": end,
                    "context_snippet": snippet,
                })

    return pd.DataFrame(rows)


def summarize_custom_labels(system_entities):
    """Summarize how many custom entities fired in each system."""
    rows = []

    for system_name, entities_df in system_entities.items():
        custom_df = entities_df[
            entities_df["entity_label"].isin(CUSTOM_LABELS)
        ].copy()

        counts = custom_df["entity_label"].value_counts().to_dict()

        for label in CUSTOM_LABELS:
            rows.append({
                "system": system_name,
                "custom_label": label,
                "count": int(counts.get(label, 0)),
            })

    return pd.DataFrame(rows)


def main():
    df = load_articles()
    gold = pd.read_csv(GOLD_PATH)

    print("Dataset shape:", df.shape)
    print("Languages:", df["language"].value_counts().to_dict())
    print("Categories:", df["category"].value_counts().to_dict())

    print("\n=== Step 1: Pattern Inventory ===")
    pattern_summary = find_pattern_examples(df, CUSTOM_PATTERNS)
    pattern_summary = pattern_summary.sort_values(
        ["match_count", "label", "pattern"],
        ascending=[False, True, True],
    )

    print(pattern_summary.to_string(index=False))

    pattern_summary_path = OUTPUT_DIR / "stretch_pattern_inventory.csv"
    pattern_summary.to_csv(pattern_summary_path, index=False)
    print(f"\nSaved pattern inventory to: {pattern_summary_path}")

    print("\n=== Step 2: Build NER Systems ===")
    pipelines = {
        "base_spacy": build_spacy_pipeline("base"),
        "ruler_before_ner": build_spacy_pipeline("before"),
        "ruler_after_ner": build_spacy_pipeline("after"),
    }

    system_entities = {}

    for system_name, nlp in pipelines.items():
        print(f"\nRunning system: {system_name}")
        entities_df = extract_entities(df, nlp)
        entities_df["system"] = system_name
        system_entities[system_name] = entities_df

        output_path = OUTPUT_DIR / f"stretch_entities_{system_name}.csv"
        entities_df.to_csv(output_path, index=False)

        print(f"Total entities: {len(entities_df)}")
        print(f"Saved entities to: {output_path}")

    print("\n=== Entity Counts by System and Label ===")
    counts_df = label_counts(system_entities)
    print(counts_df.to_string(index=False))

    counts_path = OUTPUT_DIR / "stretch_entity_counts_by_system.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"\nSaved label counts to: {counts_path}")

    print("\n=== Custom Label Counts ===")
    custom_counts_df = summarize_custom_labels(system_entities)
    print(custom_counts_df.to_string(index=False))

    custom_counts_path = OUTPUT_DIR / "stretch_custom_label_counts.csv"
    custom_counts_df.to_csv(custom_counts_path, index=False)
    print(f"\nSaved custom label counts to: {custom_counts_path}")

    print("\n=== Standard-Label Evaluation on Gold Subset ===")
    metrics_df = evaluate_all_systems(system_entities, gold)
    print(metrics_df.to_string(index=False))

    metrics_path = OUTPUT_DIR / "stretch_standard_label_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved standard-label metrics to: {metrics_path}")

    print("\n=== Custom Rule Examples ===")
    examples_df = get_custom_rule_examples(system_entities, df)
    if examples_df.empty:
        print("No custom rule examples found.")
    else:
        print(
            examples_df[
                ["system", "text_id", "category", "entity_text", "entity_label"]
            ].head(40).to_string(index=False)
        )

    examples_path = OUTPUT_DIR / "stretch_custom_rule_examples.csv"
    examples_df.to_csv(examples_path, index=False)
    print(f"\nSaved custom rule examples to: {examples_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()