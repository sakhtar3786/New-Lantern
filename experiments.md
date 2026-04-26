# Experiments & Write-Up

## Problem Statement

Given a current radiology examination (described by modality + body region + clinical context) and a list of prior examinations for the same patient, predict which prior studies a radiologist should review when reading the current exam.

## Approach

### Baseline: Rule-based keyword matching
**Method:** Extract modality (MRI, CT, X-RAY, US, PET) and body region (BRAIN, CHEST, SPINE, KNEE…) from study descriptions using regex. Mark a prior as relevant if modality and region both match the current study.

**Result:** ~72% accuracy on public eval. Fails on cross-modality relevance (e.g., CT Head is relevant when current is MRI Brain Stroke) and related-region cases (e.g., thoracic spine relevant to cervical spine study).

### LLM-based classification (claude-sonnet-4)
**Method:** Send one LLM call per case containing all prior studies together. The system prompt encodes radiological relevance heuristics: same body region, related modalities, clinical context (stroke follow-up, cancer staging, trauma). Model returns a JSON boolean array.

**Key design decisions:**
- **One call per case, not per prior** — eliminates timeout risk on private eval with 27K+ priors
- **In-memory caching** — keyed on (current_desc, prior_desc, days_diff); avoids repeat LLM calls for identical study pairs
- **Robust JSON parsing** — regex extracts `[...]` from response so preamble/explanation doesn't break parsing
- **Fallback to false** — if the LLM response can't be parsed, default to not relevant (safer for workflow)

**Result:** ~89–93% accuracy on public eval (varies by batch ordering and model temperature).

## What Worked
- Batching all priors per case into one LLM prompt dramatically reduced latency
- Explicit radiological heuristics in the system prompt (same region, cross-modality, clinical context) significantly improved over keyword rules
- Caching prevented duplicate API costs on retries and repeated study pairs
- Including days-before-current in the prompt helped the model weight recency

## What Failed
- Pure rule-based approaches miss nuanced cross-modality relevance
- Per-prior LLM calls: timed out on even small batches (5+ cases × 30 priors = 150 calls)
- Very long study description lists occasionally caused the model to return fewer booleans than priors; padding logic fixed this

## Next-Step Improvements

### 1. Fine-tuned classifier
Train a lightweight binary classifier (e.g., sentence-transformers + logistic regression) on the labeled public split. Encode `(current_desc, prior_desc)` as separate embeddings, concatenate with derived features (days_diff, modality_match, region_match), and train with cross-entropy loss. Expected: <10ms per prediction, no API cost, higher consistency.

### 2. Structured feature extraction
Parse study descriptions into structured fields: modality, body_region, laterality, contrast, clinical_indication. Use a lookup table of body-region compatibility (e.g., brain→head/neck, chest→thorax/cardiac) and modality compatibility (MRI↔CT for same region = relevant; PET/CT→oncology context). This gives deterministic, interpretable decisions.

### 3. Temporal weighting
Priors >5 years old are often less relevant unless the current study is a long-term follow-up. Add a learned temporal discount curve.

### 4. LLM + classifier ensemble
Use the rule-based classifier as a fast first pass, only escalating uncertain cases (confidence < 0.7) to the LLM. This cuts API costs by ~80% while maintaining high accuracy on clear-cut cases.

### 5. Semantic embedding similarity
Compute cosine similarity between sentence-transformer embeddings of current vs. prior study descriptions. Threshold at ~0.65 for relevance. Fast, cheap, no API, and captures semantic relationships the rule-based approach misses.

## Architecture

```
POST /predict
     |
     v
Flask app (gunicorn, 2 workers, 360s timeout)
     |
     v
For each case:
  1. Check cache for all (current, prior, days_diff) pairs
  2. Batch uncached priors → single Claude API call
  3. Parse JSON boolean array from response
  4. Store results in cache
  5. Append predictions
     |
     v
Return {"predictions": [...]}
```

## Environment
- Python 3.11, Flask 3.x, Anthropic SDK
- Model: claude-sonnet-4-20250514
- Deployment: any platform supporting gunicorn (Railway, Render, Fly.io)
