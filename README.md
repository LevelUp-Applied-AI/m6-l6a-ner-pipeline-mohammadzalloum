# Module 6A Stretch — Custom NER Rules

This repository contains the **Module 6 Week A Stretch: Custom NER Rules** assignment. The project extends a baseline spaCy Named Entity Recognition pipeline with a custom `EntityRuler` designed for climate-domain terminology.

The goal is to show how a general-purpose NER model can be improved with carefully designed rule-based patterns for domain-specific concepts such as climate conferences, policy agreements, assessment reports, temperature thresholds, climate funds, and policy mechanisms.

---

## Project Overview

General-purpose NER models often miss or misclassify domain-specific terms. In climate text, concepts such as `COP28`, `Paris Agreement`, `Sixth Assessment Report`, `NDCs`, and `Loss and Damage Fund` carry important meaning, but they are not always captured cleanly by the base spaCy model.

This stretch assignment adds a custom spaCy `EntityRuler` and compares three NER configurations:

1. **Base spaCy** — the original `en_core_web_sm` NER pipeline.
2. **EntityRuler before NER** — custom rules run before the statistical NER model.
3. **EntityRuler after NER** — custom rules run after the statistical NER model.

The comparison measures how the rule-based component changes entity extraction behavior and how it affects standard-label precision, recall, and F1 on the gold-standard subset.

---

## Dataset

The dataset contains climate-related article snippets with multilingual content.

| Metric | Value |
|---|---:|
| Rows | 200 |
| Columns | 5 |
| English texts | 132 |
| Arabic texts | 68 |

Category distribution:

| Category | Count |
|---|---:|
| adaptation | 61 |
| science | 50 |
| impact | 46 |
| policy | 43 |

The stretch script evaluates only English texts because the spaCy model used in this project is English-specific.

---

## Custom Entity Types

The custom `EntityRuler` defines climate-domain entity labels beyond the standard spaCy schema.

| Custom Label | Description |
|---|---|
| `CLIMATE_EVENT` | Climate conferences and summits, such as COP events |
| `AGREEMENT` | Climate agreements, treaties, and pledges |
| `REPORT` | Climate reports and assessments |
| `THRESHOLD` | Temperature targets and climate thresholds |
| `POLICY` | Policy terms and national commitments |
| `POLICY_MECHANISM` | Climate policy instruments |
| `CLIMATE_FUND` | Climate finance funds |

Example patterns include:

- `COP28`
- `COP27`
- `COP26`
- `Paris Agreement`
- `Kigali Amendment`
- `Montreal Protocol`
- `Global Methane Pledge`
- `Sixth Assessment Report`
- `Emissions Gap Report`
- `Adaptation Gap Report`
- `Global Methane Assessment`
- `1.5 degrees Celsius`
- `2 degrees`
- `NDCs`
- `Carbon Border Adjustment Mechanism`
- `Loss and Damage Fund`
- `Green Climate Fund`

---

## Project Files

| File | Purpose |
|---|---|
| `stretch_custom_ner.py` | Main stretch script for custom EntityRuler experiments |
| `stretch_analysis.md` | Written analysis of results, examples, and limitations |
| `data/climate_articles.csv` | Climate article dataset |
| `data/gold_entities.csv` | Gold-standard entity annotations |
| `outputs/stretch_custom_ner_results.txt` | Full terminal output from the stretch run |
| `outputs/stretch_pattern_inventory.csv` | Pattern match counts and example text IDs |
| `outputs/stretch_entity_counts_by_system.csv` | Entity label counts for each NER system |
| `outputs/stretch_custom_label_counts.csv` | Custom label counts by pipeline configuration |
| `outputs/stretch_standard_label_metrics.csv` | Precision, recall, F1, and deltas on standard labels |
| `outputs/stretch_custom_rule_examples.csv` | Dataset examples where custom rules fired |

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

If the base lab Hugging Face pipeline is used, install PyTorch CPU build separately:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## How to Run

Run the stretch script and save the output:

```bash
python stretch_custom_ner.py | tee outputs/stretch_custom_ner_results.txt
```

The script will:

1. Load the climate articles and gold-standard annotations.
2. Build the base spaCy pipeline.
3. Build an EntityRuler-before-NER pipeline.
4. Build an EntityRuler-after-NER pipeline.
5. Extract entities from English texts.
6. Compare entity counts by system and label.
7. Evaluate standard-label precision, recall, and F1.
8. Save output CSV files under `outputs/`.

---

## Results Summary

### Total Entity Counts

| System | Total Entities |
|---|---:|
| base_spacy | 1202 |
| ruler_before_ner | 1212 |
| ruler_after_ner | 1211 |

### Custom Label Counts

| Custom Label | base_spacy | ruler_before_ner | ruler_after_ner |
|---|---:|---:|---:|
| AGREEMENT | 0 | 11 | 0 |
| CLIMATE_EVENT | 0 | 9 | 5 |
| CLIMATE_FUND | 0 | 2 | 1 |
| POLICY | 0 | 4 | 3 |
| POLICY_MECHANISM | 0 | 1 | 0 |
| REPORT | 0 | 4 | 0 |
| THRESHOLD | 0 | 5 | 0 |

### Standard-Label Evaluation

Custom labels were excluded from the numeric gold-standard evaluation because the gold file contains only standard spaCy labels.

| System | Precision | Recall | F1 | F1 Delta vs Base |
|---|---:|---:|---:|---:|
| base_spacy | 0.662 | 0.652 | 0.657 | 0.000 |
| ruler_before_ner | 0.763 | 0.652 | 0.703 | +0.046 |
| ruler_after_ner | 0.662 | 0.652 | 0.657 | 0.000 |

The best configuration was `ruler_before_ner`, which improved F1 from `0.657` to `0.703` by increasing precision while keeping recall unchanged.

---

## Key Findings

The `ruler_before_ner` pipeline captured the most domain-specific climate concepts because the custom rules ran before spaCy's statistical NER model. This allowed the EntityRuler to claim complete climate-domain spans such as `Paris Agreement`, `Sixth Assessment Report`, `1.5 degrees Celsius`, and `Carbon Border Adjustment Mechanism` before spaCy split or labeled parts of them differently.

The `ruler_after_ner` pipeline preserved spaCy's original predictions more strongly, but it captured fewer custom entities because many matching spans had already been assigned by the statistical NER model.

The improvement in standard-label F1 came mainly from precision. The number of true positives stayed at 45, while predicted standard entities decreased from 68 to 59 in the before-NER configuration. This suggests that some noisy standard-label predictions were replaced by more specific custom labels.

---

## Example Custom Rule Matches

| text_id | Entity Text | Custom Label | Category |
|---:|---|---|---|
| 5 | Paris Agreement | AGREEMENT | policy |
| 5 | Bonn Climate Change Conference | CLIMATE_EVENT | policy |
| 5 | NDCs | POLICY | policy |
| 1 | Sixth Assessment Report | REPORT | policy |
| 1 | 1.5 degrees Celsius | THRESHOLD | policy |
| 2 | COP28 | CLIMATE_EVENT | policy |
| 6 | Carbon Border Adjustment Mechanism | POLICY_MECHANISM | policy |
| 9 | Green Climate Fund | CLIMATE_FUND | policy |
| 48 | Loss and Damage Fund | CLIMATE_FUND | policy |
| 71 | Emissions Gap Report | REPORT | policy |

---

## Limitations

The gold-standard file contains only standard labels, not custom labels such as `AGREEMENT`, `REPORT`, `CLIMATE_EVENT`, or `THRESHOLD`. For that reason, custom labels were evaluated qualitatively through dataset examples rather than as exact gold-standard matches.

Some patterns may still be broader than ideal. For example, `2 degrees` can match both policy targets and physical temperature anomaly descriptions. A future improvement could use more precise token patterns such as `2 degrees of warming` or `2 degrees Celsius above`.

---

## Run Checks

Compile the stretch script:

```bash
python -m py_compile stretch_custom_ner.py
```

Check generated outputs:

```bash
ls outputs
```

Recommended Git add command for this stretch PR:

```bash
git add stretch_custom_ner.py stretch_analysis.md README.md
git add -f outputs/stretch_custom_ner_results.txt
git add -f outputs/stretch_pattern_inventory.csv
git add -f outputs/stretch_entity_counts_by_system.csv
git add -f outputs/stretch_custom_label_counts.csv
git add -f outputs/stretch_standard_label_metrics.csv
git add -f outputs/stretch_custom_rule_examples.csv
```

---

## Branch and Submission

This stretch assignment should be submitted from a separate branch:

```bash
git checkout -b stretch-custom-ner
```

After committing and pushing, open a pull request from:

```text
stretch-custom-ner -> main
```

Submit the stretch PR URL to TalentLMS under **Module 6 Week A -> Stretch 6A-S1**.
