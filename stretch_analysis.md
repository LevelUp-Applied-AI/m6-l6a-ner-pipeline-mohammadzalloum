# Stretch Custom NER Analysis

## Overview

This stretch assignment extends the Lab 6A NER pipeline with a custom spaCy `EntityRuler` for climate-specific terminology. The goal was to improve recognition of domain-specific concepts that the base spaCy model may miss or misclassify, such as climate conferences, agreements, reports, targets, funds, and policy mechanisms.

The custom rules were tested in two pipeline positions:

1. `ruler_before_ner`: EntityRuler runs before spaCy's statistical NER.
2. `ruler_after_ner`: EntityRuler runs after spaCy's statistical NER.

The base spaCy model was also used as the comparison baseline.

## Pattern Inventory

The custom EntityRuler included more than 10 pattern entries across several custom entity types, including:

- `CLIMATE_EVENT`
- `AGREEMENT`
- `REPORT`
- `THRESHOLD`
- `POLICY`
- `POLICY_MECHANISM`
- `CLIMATE_FUND`

Several patterns appeared multiple times in the dataset:

| Custom Pattern | Label | Match Count |
|---|---|---:|
| Paris Agreement | AGREEMENT | 6 |
| 2 degrees | THRESHOLD | 5 |
| COP28 | CLIMATE_EVENT | 4 |
| COP26 | CLIMATE_EVENT | 2 |
| NDCs | POLICY | 2 |
| Nationally Determined Contribution | POLICY | 2 |
| Montreal Protocol | AGREEMENT | 2 |
| Sixth Assessment Report | REPORT | 1 |
| Emissions Gap Report | REPORT | 1 |
| Adaptation Gap Report | REPORT | 1 |
| Global Methane Assessment | REPORT | 1 |
| Carbon Border Adjustment Mechanism | POLICY_MECHANISM | 1 |
| Loss and Damage Fund | CLIMATE_FUND | 1 |
| Green Climate Fund | CLIMATE_FUND | 1 |

One pattern, `2°C target`, did not appear in the current dataset, but it was kept as a reasonable climate-domain pattern for future texts.

## Entity Count Comparison

| System | Total Entities |
|---|---:|
| base_spacy | 1202 |
| ruler_before_ner | 1212 |
| ruler_after_ner | 1211 |

The total number of entities changed only slightly, but the entity label distribution changed meaningfully when the EntityRuler ran before NER.

## Custom Label Counts

| Custom Label | base_spacy | ruler_before_ner | ruler_after_ner |
|---|---:|---:|---:|
| AGREEMENT | 0 | 11 | 0 |
| CLIMATE_EVENT | 0 | 9 | 5 |
| CLIMATE_FUND | 0 | 2 | 1 |
| POLICY | 0 | 4 | 3 |
| POLICY_MECHANISM | 0 | 1 | 0 |
| REPORT | 0 | 4 | 0 |
| THRESHOLD | 0 | 5 | 0 |

The `ruler_before_ner` pipeline captured the most domain-specific climate concepts. It identified agreements such as `Paris Agreement`, reports such as `Sixth Assessment Report`, thresholds such as `1.5 degrees Celsius`, and policy concepts such as `NDCs`.

The `ruler_after_ner` pipeline captured fewer custom entities because spaCy's statistical NER had already created overlapping entities. Since the after-NER EntityRuler did not overwrite existing entities, many custom phrase matches were blocked by spaCy's earlier predictions.

## Standard-Label Evaluation on Gold Subset

The gold standard contains standard spaCy labels only, so custom labels were excluded from the numeric evaluation. Precision, recall, and F1 were measured only on overlapping standard labels.

| System | Precision | Recall | F1 | F1 Delta vs Base |
|---|---:|---:|---:|---:|
| base_spacy | 0.662 | 0.652 | 0.657 | 0.000 |
| ruler_before_ner | 0.763 | 0.652 | 0.703 | +0.046 |
| ruler_after_ner | 0.662 | 0.652 | 0.657 | 0.000 |

The `ruler_before_ner` pipeline improved F1 from `0.657` to `0.703`. Precision increased from `0.662` to `0.763`, while recall stayed the same at `0.652`.

This happened because the number of predicted standard-label entities decreased from 68 to 59, while the number of true positives stayed at 45. In other words, some spans that spaCy previously labeled as standard entities were converted into more specific custom labels and excluded from the standard-label evaluation. This reduced standard-label false positives without changing the number of matched gold entities.

## Dataset Examples

The custom EntityRuler captured several meaningful climate-domain concepts:

- `text_id=5`: `Paris Agreement` was captured as `AGREEMENT`.
- `text_id=5`: `Bonn Climate Change Conference` was captured as `CLIMATE_EVENT`.
- `text_id=5`: `NDCs` and `nationally determined contributions` were captured as `POLICY`.
- `text_id=1`: `Sixth Assessment Report` was captured as `REPORT`.
- `text_id=1`: `1.5 degrees Celsius` was captured as `THRESHOLD`.
- `text_id=2`: `COP28` was captured as `CLIMATE_EVENT`.
- `text_id=6`: `Carbon Border Adjustment Mechanism` was captured as `POLICY_MECHANISM`.
- `text_id=9`: `Green Climate Fund` was captured as `CLIMATE_FUND`.
- `text_id=48`: `Loss and Damage Fund` was captured as `CLIMATE_FUND`.
- `text_id=71`: `Emissions Gap Report` was captured as `REPORT`.

These examples show that the custom rules helped extract full climate-domain phrases instead of relying only on spaCy's general-purpose entity labels.

## Before vs After Behavior

The before-NER position gave the custom rules priority. This was useful for full phrase concepts such as `Paris Agreement`, `Sixth Assessment Report`, and `1.5 degrees Celsius`, because the EntityRuler could claim the complete span before the statistical NER split or labeled parts of it.

The after-NER position preserved spaCy's original predictions more strongly. This produced the same standard-label evaluation metrics as the base model, but fewer custom domain entities were added. For example, `ruler_after_ner` captured `COP28`, `COP26`, `Loss and Damage Fund`, and `NDCs`, but it did not capture `AGREEMENT`, `REPORT`, `THRESHOLD`, or `POLICY_MECHANISM` examples in the same way as the before-NER pipeline.

## Noise and Limitations

The custom rules were mostly specific and useful, but some patterns can still introduce noise. For example, the pattern `2 degrees` matched both climate warming thresholds and temperature anomaly descriptions. This is still climate-relevant, but it is broader than a strict policy target. A more precise future version could use token patterns such as `2 degrees of warming` or `2 degrees Celsius above` to separate policy thresholds from physical temperature anomalies.

Another limitation is that the gold standard only contains standard labels, not custom labels such as `AGREEMENT`, `REPORT`, or `CLIMATE_EVENT`. Therefore, custom labels were evaluated qualitatively using dataset examples rather than counted as exact gold-standard matches.

## Conclusion

The custom EntityRuler improved the domain usefulness of the NER pipeline. The strongest configuration was `ruler_before_ner`, which captured 36 custom climate-domain entities and improved standard-label F1 from `0.657` to `0.703`. The improvement came mainly from better precision, not higher recall. The before-NER configuration is better when the goal is to prioritize domain-specific concepts, while the after-NER configuration is safer when preserving the base spaCy labels is more important.