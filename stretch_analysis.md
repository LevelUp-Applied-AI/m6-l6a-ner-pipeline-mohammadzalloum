# Multilingual NER Comparison Analysis

## Overview

This stretch compares multilingual Named Entity Recognition performance across English and Arabic climate articles. I used a balanced sample of 40 articles from `climate_articles.csv`: 20 English texts and 20 Arabic texts, with 5 texts from each category (`adaptation`, `impact`, `policy`, and `science`) per language. This made the comparison fairer because both languages were represented across the same topic categories.

The analysis keeps each multilingual model's native labels (`PER`, `LOC`, `ORG`, and `MISC`). I used two multilingual NER systems: spaCy `xx_ent_wiki_sm` and Hugging Face `Davlan/xlm-roberta-base-wikiann-ner`. Arabic evaluation is qualitative because the dataset does not include Arabic gold-standard entity annotations.

## Comparison Table

| Language | Model | Texts | Total Words | Total Entities | Entities / 100 Words | No-Entity Rate | Entity Type Counts |
|---|---|---:|---:|---:|---:|---:|---|
| Arabic | HF XLM-RoBERTa | 20 | 957 | 70 | 7.31 | 0.00 | LOC=38; ORG=31; PER=1 |
| Arabic | spaCy xx_ent_wiki_sm | 20 | 957 | 20 | 2.09 | 0.35 | PER=9; MISC=6; ORG=4; LOC=1 |
| English | HF XLM-RoBERTa | 20 | 1269 | 99 | 7.80 | 0.00 | LOC=64; ORG=34; PER=1 |
| English | spaCy xx_ent_wiki_sm | 20 | 1269 | 105 | 8.27 | 0.00 | LOC=55; MISC=21; ORG=20; PER=9 |

## English vs Arabic Findings

The English results were relatively stable across both multilingual models. spaCy extracted 105 English entities, while the Hugging Face model extracted 99. Their entity density was also close: 8.27 entities per 100 words for spaCy and 7.80 for Hugging Face. Both models found useful English climate entities such as `Green Climate Fund` as `ORG`, `Songdo` as `LOC`, `South Korea` as `LOC`, and `Hurricane Otis` as `MISC`. This suggests that English NER was easier and more consistent in this dataset, likely because English named entities often have capitalization, clearer token boundaries, and stronger representation in multilingual model training data.

The Arabic results showed a much larger model gap. Hugging Face extracted 70 Arabic entities with a density of 7.31 entities per 100 words, while spaCy extracted only 20 Arabic entities with a density of 2.09. Hugging Face found at least one entity in every Arabic text, while spaCy found no entities in 7 out of 20 Arabic texts. Qualitatively, Hugging Face captured useful Arabic locations and organizations such as `الأردن` (`LOC`), `المملكة العربية السعودية` (`LOC`), `الجمعية الملكية لحماية الطبيعة` (`ORG`), `دائرة الأرصاد الجوية الأردنية` (`ORG`), and `المنظمة العالمية للأرصاد الجوية` (`ORG`). In contrast, spaCy produced several suspicious Arabic spans such as `وأكد التقرير` as `PER`, `وتهدف` as `MISC`, and `وساهمت` as `PER`, which look like ordinary verbs or phrases rather than named entities.

## Harder Entity Types in Arabic

The hardest Arabic entity types were organizations and multi-word location or institution names. Arabic does not use capitalization, so the model cannot rely on visual cues like `Jordan`, `United Nations`, or `Green Climate Fund`. Arabic entities also often appear as longer phrases, such as `الهيئة الحكومية الدولية المعنية بتغير المناخ` or `الجمعية الملكية لحماية الطبيعة`, where the model has to identify the full boundary of the organization. This makes boundary detection harder than in English. In addition, Arabic words may include prefixes such as `و`, `ب`, or `ل`, which can blur where an entity begins. The spaCy model struggled with this more clearly, often labeling short verbal phrases as `PER` or `MISC` instead of finding the actual named organization or location.

Hugging Face handled Arabic organizations and locations much better than spaCy, but it was not perfect. Some outputs were low-confidence or looked like tokenization/boundary errors, such as `دان` (`PER`), `أو` (`LOC`), `جز` (`LOC`), and `س` (`LOC`). This means Hugging Face had stronger Arabic coverage, but its Arabic output still needs review, filtering, or post-processing. A production system should not blindly trust every extracted Arabic entity, especially very short spans or low-confidence predictions.

## MENA Application Implications

For bilingual NLP applications in Jordan and the wider MENA region, these results show that using an English-only or weak multilingual pipeline is not enough. The same climate dataset contains English reports and Arabic sources, but the model behavior was very different across languages. spaCy worked well on English but missed many Arabic entities, including 7 Arabic texts where it found nothing at all. In a real system for news monitoring, climate policy tracking, or government report analysis, this would create an English bias: English documents would be richly indexed, while Arabic documents would lose important organizations, countries, cities, and institutional names.

The Hugging Face XLM-RoBERTa model is a better starting point for bilingual Arabic-English NER because it produced more balanced entity density across the two languages and captured Arabic `LOC` and `ORG` entities more reliably. However, the qualitative errors show that a real MENA NLP system should include Arabic-specific post-processing, manual review for high-impact outputs, and possibly fine-tuning on regional Arabic data. Custom rules could also help for recurring climate terms such as Arabic names of ministries, funds, agreements, conferences, and international organizations. Overall, the best production approach would combine a stronger multilingual model like XLM-RoBERTa with Arabic normalization, confidence filtering, domain-specific EntityRuler patterns, and human evaluation.

## Key Takeaways

- English NER was stable across both multilingual models.
- Hugging Face was much stronger than spaCy on Arabic text.
- Arabic `ORG` and multi-word `LOC` entities were harder because of missing capitalization, longer phrase boundaries, and Arabic morphology.
- spaCy had a high Arabic no-entity rate: 7 out of 20 Arabic texts.
- Hugging Face had better Arabic coverage but still made boundary and tokenization errors.
- For MENA bilingual NLP systems, multilingual models should be evaluated separately by language and supported with Arabic-specific cleaning, rules, and manual review.