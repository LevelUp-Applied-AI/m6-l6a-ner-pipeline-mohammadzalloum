# Multilingual NER Qualitative Notes

## Label schema

This analysis keeps the multilingual models' native labels: PER, LOC, ORG, and MISC. The Arabic side is qualitative because there is no Arabic gold-standard annotation file.

## Quantitative context

- Arabic + HF total entities: 70
- Arabic + spaCy total entities: 20
- English + HF total entities: 99
- English + spaCy total entities: 105
- Arabic + HF entity density per 100 words: 7.31
- Arabic + spaCy entity density per 100 words: 2.09
- English + HF entity density per 100 words: 7.8
- English + spaCy entity density per 100 words: 8.27
- Arabic + HF no-entity rate: 0.0
- Arabic + spaCy no-entity rate: 0.35
- Arabic texts where HF found entities but spaCy found none: 7

## Strong Arabic HF examples

- محطة طاقة شمسية (ORG)
- الأردن (LOC)
- استراتيجية التكيف الوطنية لسنغافورة (ORG)
- مدينة المدينة المنورة (LOC)
- المملكة العربية السعودية (LOC)
- محمية الأزرق المائية (LOC)

## Arabic spaCy review examples

These are candidates for false positives or boundary/type errors:

- وأكد التقرير (PER)
- دول (ORG)
- وتهدف (MISC)
- وساهمت (PER)
- وأفادت دائرة (PER)
- وقُدرت (MISC)

## HF review examples

These are not automatically wrong, but they should be manually inspected because they are low-confidence or very short spans:

- دان (PER)
- أو (LOC)
- ائر (LOC)
- جز (LOC)
- س (LOC)
- يا (LOC)

## English examples

- Netherlands (LOC)
- Rhine (LOC)
- Meuse (LOC)
- IJssel (LOC)
- Europe (LOC)
- Room for the River program (ORG)

## Draft interpretation

The English results are relatively stable across both multilingual models. spaCy extracted slightly more English entities, while HF produced a similar number and focused mostly on LOC and ORG labels.

The Arabic results show a much larger model gap. HF extracted substantially more Arabic entities and found at least one entity in every Arabic text. spaCy extracted far fewer Arabic entities and left several Arabic texts with no entities. Qualitatively, HF captured useful Arabic locations and organizations, while spaCy produced several suspicious Arabic spans that look like verbs or ordinary phrases rather than named entities. This suggests that Arabic entity boundary detection and organization recognition are harder, and that HF is more suitable for this bilingual climate dataset.
