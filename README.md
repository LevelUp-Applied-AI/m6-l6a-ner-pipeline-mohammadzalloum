# Module 6 Week A — NER Pipeline and Multilingual NER Stretch

This repository contains the Module 6 Week A Named Entity Recognition (NER) lab and stretch work for the climate articles dataset. The base lab builds English NER pipelines using spaCy and Hugging Face, while the stretch assignment compares multilingual NER behavior across English and Arabic climate text.

## Project Overview

The dataset contains climate-related articles in both English and Arabic. The base NER pipeline focuses mainly on English text, while the multilingual stretch extends the analysis to Arabic using multilingual models.

The stretch compares two multilingual NER systems:

- spaCy `xx_ent_wiki_sm`
- Hugging Face `Davlan/xlm-roberta-base-wikiann-ner`

The goal is to understand how NER quality changes across English and Arabic text, especially for entity types such as organizations, locations, people, and miscellaneous named entities.

## Repository Structure

```text
.
├── data/
│   ├── climate_articles.csv
│   └── gold_entities.csv
├── outputs/
│   ├── stretch_multilingual_sample.csv
│   ├── multilingual_model_smoke_test.csv
│   ├── multilingual_raw_entities.csv
│   ├── multilingual_document_entity_counts.csv
│   ├── multilingual_ner_comparison.csv
│   ├── multilingual_label_counts.csv
│   ├── multilingual_example_entities.csv
│   ├── multilingual_qualitative_review.csv
│   ├── multilingual_failure_candidates.csv
│   └── multilingual_qualitative_notes.md
├── tests/
├── ner_pipeline.py
├── stretch_multilingual_ner.py
├── stretch_analysis.md
├── requirements.txt
├── setup.sh
└── README.md
```

## Main Files

### `ner_pipeline.py`

Base Lab 6A NER pipeline. It includes:

- Dataset loading and exploration
- Text preprocessing
- spaCy English NER extraction
- Hugging Face English NER extraction
- NER output comparison
- Gold-standard evaluation
- Challenge extensions such as category-level analysis, entity normalization, co-occurrence analysis, and custom evaluation utilities

### `stretch_multilingual_ner.py`

Stretch assignment script for multilingual NER comparison. It includes:

- Balanced sample creation: 20 English texts and 20 Arabic texts
- spaCy multilingual NER using `xx_ent_wiki_sm`
- Hugging Face multilingual NER using `Davlan/xlm-roberta-base-wikiann-ner`
- Raw multilingual entity extraction
- Document-level entity counts
- Entity density calculations
- No-entity-rate analysis
- Label count comparison
- Example entity extraction
- Qualitative review and failure candidate detection

### `stretch_analysis.md`

Final written analysis for the multilingual NER stretch. It summarizes the comparison results and explains what the findings mean for bilingual NLP applications in the MENA region.

## Setup

### 1. Create and activate the virtual environment

```bash
bash setup.sh
source .venv/bin/activate
```

If the virtual environment already exists, `setup.sh` will reuse it.

### 2. Install dependencies

The setup script installs dependencies from `requirements.txt`.

Required packages include:

```text
spacy
pandas
numpy
matplotlib
pytest
transformers
torch
```

### 3. Download spaCy models

For the base English NER pipeline:

```bash
python -m spacy download en_core_web_sm
```

For the multilingual stretch:

```bash
python -m spacy download xx_ent_wiki_sm
```

If `python` points to the system interpreter instead of the virtual environment, use:

```bash
.venv/bin/python -m spacy download xx_ent_wiki_sm
```

## How to Run

### Run the base NER pipeline

```bash
python ner_pipeline.py
```

Or, if your system Python is not using the virtual environment:

```bash
.venv/bin/python ner_pipeline.py
```

### Run the multilingual stretch

```bash
python stretch_multilingual_ner.py
```

Or:

```bash
.venv/bin/python stretch_multilingual_ner.py
```

The first Hugging Face run may take longer because the model weights need to be downloaded.

## Outputs

The multilingual stretch produces the following main outputs:

| Output File | Description |
|---|---|
| `outputs/stretch_multilingual_sample.csv` | Balanced sample of 20 English and 20 Arabic texts |
| `outputs/multilingual_model_smoke_test.csv` | Small test confirming both multilingual models work on English and Arabic |
| `outputs/multilingual_raw_entities.csv` | All extracted entities from the full multilingual sample |
| `outputs/multilingual_document_entity_counts.csv` | Entity counts per text and model, including zero-entity cases |
| `outputs/multilingual_ner_comparison.csv` | Main comparison table by language and model |
| `outputs/multilingual_label_counts.csv` | Entity label counts by language and model |
| `outputs/multilingual_example_entities.csv` | Example entities for each language/model combination |
| `outputs/multilingual_qualitative_review.csv` | Side-by-side qualitative comparison of spaCy and Hugging Face outputs |
| `outputs/multilingual_failure_candidates.csv` | Candidate boundary/type/tokenization errors for manual review |
| `outputs/multilingual_qualitative_notes.md` | Draft notes supporting the final stretch analysis |

## Key Multilingual Results

The multilingual stretch used a balanced sample of 40 articles:

- 20 English texts
- 20 Arabic texts
- 5 texts per category per language

Main comparison results:

| Language | Model | Total Entities | Entities / 100 Words | No-Entity Rate |
|---|---|---:|---:|---:|
| Arabic | Hugging Face XLM-RoBERTa | 70 | 7.31 | 0.00 |
| Arabic | spaCy `xx_ent_wiki_sm` | 20 | 2.09 | 0.35 |
| English | Hugging Face XLM-RoBERTa | 99 | 7.80 | 0.00 |
| English | spaCy `xx_ent_wiki_sm` | 105 | 8.27 | 0.00 |

## Main Findings

English NER was relatively stable across both multilingual models. spaCy extracted slightly more English entities, while Hugging Face produced a similar count and focused mainly on location and organization labels.

Arabic NER showed a much larger model gap. Hugging Face extracted 70 Arabic entities and found at least one entity in every Arabic text. spaCy extracted only 20 Arabic entities and found no entities in 7 out of 20 Arabic texts.

Qualitatively, Hugging Face captured useful Arabic locations and organizations such as:

- `الأردن`
- `المملكة العربية السعودية`
- `الجمعية الملكية لحماية الطبيعة`
- `دائرة الأرصاد الجوية الأردنية`
- `المنظمة العالمية للأرصاد الجوية`

spaCy produced several suspicious Arabic spans, such as:

- `وأكد التقرير` labeled as `PER`
- `وتهدف` labeled as `MISC`
- `وساهمت` labeled as `PER`

This suggests that Arabic entity boundary detection and organization recognition are harder, especially because Arabic has no capitalization and often uses longer multi-word entity names.

## Notes on Arabic Evaluation

Arabic precision, recall, and F1 are not computed because the dataset does not include Arabic gold-standard entity annotations. Arabic evaluation is therefore qualitative and based on inspection of extracted entities, missed entities, suspicious spans, and no-entity-rate patterns.

## Testing

Run the test suite with:

```bash
python -m pytest tests/ -v
```

Or:

```bash
.venv/bin/python -m pytest tests/ -v
```

## Development Notes

Do not commit local environment or cache files such as:

```text
.venv/
__pycache__/
.pytest_cache/
.cache/
huggingface/
```

The repository keeps the generated CSV and Markdown outputs required for the stretch analysis, but avoids committing downloaded model files and local caches.

## Summary

This project shows why multilingual NER models should be evaluated separately by language. English results looked stable across both models, but Arabic results showed a large gap between spaCy and Hugging Face. For bilingual NLP applications in Jordan and the MENA region, a stronger multilingual transformer model, Arabic-specific post-processing, and manual review are important for reliable production use.
