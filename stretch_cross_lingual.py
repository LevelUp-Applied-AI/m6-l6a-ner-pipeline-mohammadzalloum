import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/climate_articles.csv"
NUM_TEXTS_PER_LANGUAGE = 10

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 256
BATCH_SIZE = 4

HEATMAP_DECIMALS = 2
HEATMAP_ANNOTATION_FONTSIZE = 6
HEATMAP_TICK_FONTSIZE = 9

OUTPUT_DIR = Path("outputs/cross_lingual_embeddings")

SIMILARITY_MATRIX_OUTPUT = OUTPUT_DIR / "stretch_similarity_matrix.csv"
CROSS_LINGUAL_SCORES_OUTPUT = OUTPUT_DIR / "stretch_similarity_scores.csv"
SIMILARITY_SUMMARY_OUTPUT = OUTPUT_DIR / "stretch_similarity_summary.csv"
EXPECTED_PAIRS_OUTPUT = OUTPUT_DIR / "stretch_expected_pair_scores.csv"
HEATMAP_OUTPUT = OUTPUT_DIR / "stretch_cross_lingual_heatmap.png"
LABEL_MAP_OUTPUT = OUTPUT_DIR / "stretch_heatmap_label_map.csv"


def load_climate_data(path):
    """
    Load the bilingual climate dataset from a CSV file.

    This function checks that the file exists before reading it,
    so the error message is clear if the path is wrong.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    df = pd.read_csv(path)

    if "language" not in df.columns:
        raise ValueError("The dataset must contain a 'language' column.")

    return df


def find_text_column(df):
    """
    Find the most likely column that contains the article text.

    Different datasets may name the text column differently,
    so this function checks common names instead of hard-coding one name.
    """
    possible_text_columns = [
        "text",
        "article",
        "content",
        "body",
        "summary",
        "description"
    ]

    for column in possible_text_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a text column. Expected one of: "
        f"{possible_text_columns}"
    )


def select_balanced_language_texts(df, text_column, language_code):
    """
    Select 10 clean texts for one language, balanced across climate categories.

    We select:
    - 3 policy texts
    - 3 science texts
    - 2 adaptation texts
    - 2 impact texts

    This gives better topic coverage for cross-lingual comparison.
    """
    category_targets = {
        "policy": 3,
        "science": 3,
        "adaptation": 2,
        "impact": 2,
    }

    language_df = df[df["language"] == language_code].copy()

    language_df[text_column] = language_df[text_column].fillna("").astype(str)
    language_df[text_column] = language_df[text_column].str.strip()
    language_df = language_df[language_df[text_column] != ""]

    selected_parts = []

    for category, target_count in category_targets.items():
        category_df = language_df[language_df["category"] == category].copy()

        if len(category_df) < target_count:
            raise ValueError(
                f"Not enough {language_code} texts in category '{category}'. "
                f"Found {len(category_df)}, need {target_count}."
            )

        selected_parts.append(category_df.head(target_count))

    selected_df = pd.concat(selected_parts, ignore_index=True)

    if len(selected_df) != 10:
        raise ValueError(
            f"Expected 10 selected texts for language '{language_code}', "
            f"but got {len(selected_df)}."
        )

    return selected_df.reset_index(drop=True)


def print_selected_texts_preview(df, text_column, language_name):
    """
    Print selected texts with category and short preview.
    """
    print(f"\nSelected {language_name} texts:")
    print("-" * 80)

    for idx, row in df.iterrows():
        category = row["category"]
        preview = row[text_column].replace("\n", " ")[:100]
        print(f"{idx + 1}. [{category}] {preview}...")


def get_device():
    """
    Select the best available device.

    If a GPU is available, use CUDA.
    Otherwise, use CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_embedding_model(model_name):
    """
    Load the multilingual BERT tokenizer and model.

    The tokenizer converts raw text into token IDs.
    The model converts token IDs into contextual embeddings.
    """
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = get_device()
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on: {device}")

    return tokenizer, model, device


def mean_pooling(last_hidden_state, attention_mask):
    """
    Convert token embeddings into one embedding per text.

    last_hidden_state shape:
    batch_size x sequence_length x hidden_size

    attention_mask tells us which tokens are real tokens
    and which tokens are padding.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    masked_embeddings = last_hidden_state * mask

    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)

    mean_embeddings = summed_embeddings / token_counts

    return mean_embeddings


def embed_texts(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """
    Extract one multilingual BERT embedding for each text.
    """
    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)

            batch_embeddings = mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            all_embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    return embeddings


def make_similarity_labels(english_df, arabic_df):
    """
    Create clear labels for the 20 selected texts.

    The first 10 labels are English texts.
    The next 10 labels are Arabic texts.
    """
    labels = []

    for idx, row in english_df.iterrows():
        label = f"EN{idx + 1}_{row['category']}_id{row['id']}"
        labels.append(label)

    for idx, row in arabic_df.iterrows():
        label = f"AR{idx + 1}_{row['category']}_id{row['id']}"
        labels.append(label)

    return labels


def compute_similarity_matrix(embeddings):
    """
    Compute the 20x20 cosine similarity matrix.

    Each row and column represents one selected text.
    """
    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix


def save_similarity_matrix(similarity_matrix, labels, output_path):
    """
    Save the full similarity matrix as a CSV file.
    """
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )

    similarity_df.to_csv(output_path)

    return similarity_df


def build_cross_lingual_scores(similarity_matrix, english_df, arabic_df, text_column):
    """
    Build a table of all English-Arabic similarity scores.

    Since English texts are stored first and Arabic texts second:
    - English index i is similarity_matrix[i]
    - Arabic index j is similarity_matrix[english_count + j]
    """
    english_count = len(english_df)
    records = []

    for en_idx, en_row in english_df.iterrows():
        for ar_idx, ar_row in arabic_df.iterrows():
            score = similarity_matrix[en_idx, english_count + ar_idx]

            records.append({
                "english_number": en_idx + 1,
                "arabic_number": ar_idx + 1,
                "english_id": en_row["id"],
                "arabic_id": ar_row["id"],
                "english_category": en_row["category"],
                "arabic_category": ar_row["category"],
                "same_category": en_row["category"] == ar_row["category"],
                "similarity_score": round(float(score), 4),
                "english_preview": str(en_row[text_column]).replace("\n", " ")[:100],
                "arabic_preview": str(ar_row[text_column]).replace("\n", " ")[:100],
            })

    scores_df = pd.DataFrame(records)
    scores_df = scores_df.sort_values("similarity_score", ascending=False)

    return scores_df


def summarize_similarity_groups(similarity_matrix, english_count):
    """
    Compare three similarity groups:
    1. English-English similarity
    2. Arabic-Arabic similarity
    3. English-Arabic similarity

    We remove diagonal self-similarity because every text has similarity 1.0 with itself.
    """
    english_block = similarity_matrix[:english_count, :english_count]
    arabic_block = similarity_matrix[english_count:, english_count:]
    cross_block = similarity_matrix[:english_count, english_count:]

    english_within = english_block[np.triu_indices(english_count, k=1)]
    arabic_within = arabic_block[np.triu_indices(english_count, k=1)]
    cross_lingual = cross_block.flatten()

    summary = pd.DataFrame([
        {
            "group": "English-English",
            "count": len(english_within),
            "mean_similarity": round(float(np.mean(english_within)), 4),
            "min_similarity": round(float(np.min(english_within)), 4),
            "max_similarity": round(float(np.max(english_within)), 4),
        },
        {
            "group": "Arabic-Arabic",
            "count": len(arabic_within),
            "mean_similarity": round(float(np.mean(arabic_within)), 4),
            "min_similarity": round(float(np.min(arabic_within)), 4),
            "max_similarity": round(float(np.max(arabic_within)), 4),
        },
        {
            "group": "English-Arabic",
            "count": len(cross_lingual),
            "mean_similarity": round(float(np.mean(cross_lingual)), 4),
            "min_similarity": round(float(np.min(cross_lingual)), 4),
            "max_similarity": round(float(np.max(cross_lingual)), 4),
        },
    ])

    return summary


def build_expected_pair_scores(similarity_matrix, english_df, arabic_df, text_column):
    """
    Score manually expected same-topic English-Arabic pairs.

    These pairs are based on the selected preview from Task 3.
    Update them only if the selected text order changes.
    """
    english_count = len(english_df)

    expected_pairs = [
        ("IPCC Sixth Assessment Report", 1, 1),
        ("World Bank climate adaptation funding", 3, 2),
        ("COP28 fossil fuel transition", 2, 3),
        ("NASA warmest year 2023", 4, 4),
        ("Greenland Ice Sheet loss", 5, 5),
        ("Mafraq solar power plant", 8, 7),
        ("Dead Sea water loss", 9, 9),
    ]

    records = []

    for topic, english_number, arabic_number in expected_pairs:
        en_idx = english_number - 1
        ar_idx = arabic_number - 1

        score = similarity_matrix[en_idx, english_count + ar_idx]

        en_row = english_df.iloc[en_idx]
        ar_row = arabic_df.iloc[ar_idx]

        records.append({
            "topic": topic,
            "english_number": english_number,
            "arabic_number": arabic_number,
            "english_category": en_row["category"],
            "arabic_category": ar_row["category"],
            "similarity_score": round(float(score), 4),
            "english_preview": str(en_row[text_column]).replace("\n", " ")[:100],
            "arabic_preview": str(ar_row[text_column]).replace("\n", " ")[:100],
        })

    return pd.DataFrame(records).sort_values(
        "similarity_score",
        ascending=False
    )


def build_heatmap_metadata(english_df, arabic_df, text_column):
    """
    Build short heatmap labels and a separate mapping table.

    The heatmap itself uses compact labels such as EN1, EN2, AR1...
    A separate CSV file stores the full preview text for readability.
    """
    compact_labels = []
    records = []

    for idx, row in english_df.iterrows():
        label = f"EN{idx + 1}"
        preview = str(row[text_column]).replace("\n", " ").strip()[:40]

        compact_labels.append(label)

        records.append({
            "heatmap_label": label,
            "language": "en",
            "number": idx + 1,
            "text_id": row["id"],
            "category": row["category"],
            "preview": preview,
        })

    for idx, row in arabic_df.iterrows():
        label = f"AR{idx + 1}"
        preview = str(row[text_column]).replace("\n", " ").strip()[:40]

        compact_labels.append(label)

        records.append({
            "heatmap_label": label,
            "language": "ar",
            "number": idx + 1,
            "text_id": row["id"],
            "category": row["category"],
            "preview": preview,
        })

    label_map_df = pd.DataFrame(records)

    return compact_labels, label_map_df

def plot_similarity_heatmap(
    similarity_matrix,
    labels,
    output_path,
    english_count,
    decimals=2,
    annotate_values=True
):
    """
    Plot a cleaner and more informative similarity heatmap.

    Improvements:
    - short labels (EN1...AR10)
    - better contrast
    - values displayed inside cells
    - adaptive annotation color for readability
    - clear separators between English and Arabic sections
    - light grid lines
    - quadrant labels
    """
    n = len(labels)

    if similarity_matrix.shape != (n, n):
        raise ValueError(
            "The number of labels must match the similarity matrix dimensions. "
            f"Matrix shape: {similarity_matrix.shape}, labels: {n}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ignore the diagonal when computing vmin to improve off-diagonal contrast
    off_diagonal_values = similarity_matrix[~np.eye(n, dtype=bool)]
    vmin = float(off_diagonal_values.min())
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        similarity_matrix,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    # Main ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    ax.set_title(
        "Cross-Lingual Embedding Similarity Heatmap\n"
        "Multilingual BERT: English and Arabic Climate Texts",
        fontsize=16,
        pad=12
    )
    ax.set_xlabel("Selected texts", fontsize=12)
    ax.set_ylabel("Selected texts", fontsize=12)

    # Light grid between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Strong separators between English and Arabic sections
    ax.axhline(english_count - 0.5, color="white", linewidth=2.5)
    ax.axvline(english_count - 0.5, color="white", linewidth=2.5)

    # Quadrant labels
    box_style = dict(facecolor="black", alpha=0.28, boxstyle="round,pad=0.25")

    ax.text(
        2, 1, "EN-EN",
        color="white", fontsize=10, weight="bold",
        bbox=box_style
    )
    ax.text(
        english_count + 2, 1, "EN-AR",
        color="white", fontsize=10, weight="bold",
        bbox=box_style
    )
    ax.text(
        2, english_count + 1, "AR-EN",
        color="white", fontsize=10, weight="bold",
        bbox=box_style
    )
    ax.text(
        english_count + 2, english_count + 1, "AR-AR",
        color="white", fontsize=10, weight="bold",
        bbox=box_style
    )

    # Add numeric annotations inside cells
    if annotate_values:
        threshold = (vmin + vmax) / 2

        for i in range(n):
            for j in range(n):
                value = float(similarity_matrix[i, j])

                # Use white text on dark cells, black text on bright cells
                text_color = "white" if value < threshold else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine Similarity", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def prepare_output_dir(output_dir):
    """
    Create the output directory if it does not already exist.

    This keeps all generated stretch files organized in one folder
    instead of saving them in the project root.
    """
    output_dir.mkdir(parents=True, exist_ok=True)





def main():
    df = load_climate_data(DATA_PATH)

    prepare_output_dir(OUTPUT_DIR)

    df = load_climate_data(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")

    text_column = find_text_column(df)
    print(f"\nDetected text column: {text_column}")

    english_texts_df = select_balanced_language_texts(
        df=df,
        text_column=text_column,
        language_code="en"
    )

    arabic_texts_df = select_balanced_language_texts(
        df=df,
        text_column=text_column,
        language_code="ar"
    )

    print_selected_texts_preview(
        english_texts_df,
        text_column,
        "English"
    )

    print_selected_texts_preview(
        arabic_texts_df,
        text_column,
        "Arabic"
    )

    print("\nTask 3 completed successfully.")
    print(f"English texts selected: {len(english_texts_df)}")
    print(f"Arabic texts selected: {len(arabic_texts_df)}")

    english_texts = english_texts_df[text_column].tolist()
    arabic_texts = arabic_texts_df[text_column].tolist()

    all_texts = english_texts + arabic_texts

    tokenizer, model, device = load_embedding_model(MODEL_NAME)

    print("\nExtracting multilingual BERT embeddings...")
    embeddings = embed_texts(
        texts=all_texts,
        tokenizer=tokenizer,
        model=model,
        device=device
    )

    print("\nTask 4 completed successfully.")
    print(f"Total texts embedded: {len(all_texts)}")
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nComputing cosine similarity matrix...")

    labels = make_similarity_labels(
        english_df=english_texts_df,
        arabic_df=arabic_texts_df
    )

    similarity_matrix = compute_similarity_matrix(embeddings)

    similarity_df = save_similarity_matrix(
        similarity_matrix=similarity_matrix,
        labels=labels,
        output_path=SIMILARITY_MATRIX_OUTPUT
    )

    cross_lingual_scores = build_cross_lingual_scores(
        similarity_matrix=similarity_matrix,
        english_df=english_texts_df,
        arabic_df=arabic_texts_df,
        text_column=text_column
    )

    cross_lingual_scores.to_csv(
        CROSS_LINGUAL_SCORES_OUTPUT,
        index=False
    )

    similarity_summary = summarize_similarity_groups(
        similarity_matrix=similarity_matrix,
        english_count=len(english_texts_df)
    )

    similarity_summary.to_csv(
        SIMILARITY_SUMMARY_OUTPUT,
        index=False
    )

    expected_pair_scores = build_expected_pair_scores(
        similarity_matrix=similarity_matrix,
        english_df=english_texts_df,
        arabic_df=arabic_texts_df,
        text_column=text_column
    )

    expected_pair_scores.to_csv(
        EXPECTED_PAIRS_OUTPUT,
        index=False
    )

    print("\nTask 5 completed successfully.")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Saved full similarity matrix to: {SIMILARITY_MATRIX_OUTPUT}")
    print(f"Saved cross-lingual scores to: {CROSS_LINGUAL_SCORES_OUTPUT}")
    print(f"Saved similarity summary to: {SIMILARITY_SUMMARY_OUTPUT}")
    print(f"Saved expected pair scores to: {EXPECTED_PAIRS_OUTPUT}")

    print("\nSimilarity summary:")
    print(similarity_summary.to_string(index=False))

    print("\nTop 10 English-Arabic pairs:")
    print(cross_lingual_scores.head(10)[[
        "english_number",
        "arabic_number",
        "english_category",
        "arabic_category",
        "same_category",
        "similarity_score"
    ]].to_string(index=False))

    print("\nExpected same-topic pair scores:")
    print(expected_pair_scores[[
        "topic",
        "english_number",
        "arabic_number",
        "similarity_score"
    ]].to_string(index=False))


    print("\nCreating similarity heatmap...")

    compact_heatmap_labels, label_map_df = build_heatmap_metadata(
        english_df=english_texts_df,
        arabic_df=arabic_texts_df,
        text_column=text_column
    )

    label_map_df.to_csv(LABEL_MAP_OUTPUT, index=False)

    heatmap_path = plot_similarity_heatmap(
        similarity_matrix=similarity_matrix,
        labels=compact_heatmap_labels,
        output_path=HEATMAP_OUTPUT,
        english_count=len(english_texts_df),
        decimals=2,
        annotate_values=True
    )

    print(f"Saved heatmap label map to: {LABEL_MAP_OUTPUT}")

    print("\nHeatmap label mapping:")
    print(label_map_df.to_string(index=False))

    print("\nTask 6 completed successfully.")
    print(f"Saved heatmap to: {heatmap_path}")


if __name__ == "__main__":
    main()