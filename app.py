"""
Radiology Prior Study Relevance API
Predicts which prior studies are relevant for a radiologist reading a current exam.
"""

import os
import json
import hashlib
import logging
import re
from flask import Flask, request, jsonify
import anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# In-memory cache: (current_study_desc, prior_study_desc) -> bool
_cache: dict[str, bool] = {}

def cache_key(current_desc: str, prior_desc: str, days_diff: int) -> str:
    raw = f"{current_desc.strip().upper()}|{prior_desc.strip().upper()}|{days_diff}"
    return hashlib.md5(raw.encode()).hexdigest()


SYSTEM_PROMPT = """You are a radiology workflow expert. Your job is to decide which prior patient examinations 
should be shown to a radiologist who is reading a new (current) examination.

A prior exam is RELEVANT (true) if it would help the radiologist interpret the current exam — for example:
- Same body region (e.g., both brain, both chest, both spine)
- Same or closely related modality (MRI and CT of the same region are often both relevant)
- Directly related clinical context (e.g., stroke follow-up, cancer staging, trauma)
- Recent enough to show disease progression or treatment response

A prior exam is NOT RELEVANT (false) if:
- It covers a completely different body region unrelated to the current exam
- It is a highly specialized exam with no bearing on the current study

You will receive a current exam and a numbered list of prior exams. 
Return ONLY a JSON array of booleans, one per prior exam, in the same order.
No explanation, no extra text — just the JSON array.
Example: [true, false, true]"""


def predict_case(current_study: dict, prior_studies: list[dict]) -> list[bool]:
    """Run one LLM call for all priors of a single case."""
    if not prior_studies:
        return []

    current_desc = current_study.get("study_description", "")
    current_date = current_study.get("study_date", "")

    # Build compact prior list
    prior_lines = []
    for i, p in enumerate(prior_studies, 1):
        p_desc = p.get("study_description", "")
        p_date = p.get("study_date", "")
        # Compute rough days difference
        try:
            from datetime import date
            cd = date.fromisoformat(current_date)
            pd = date.fromisoformat(p_date)
            days = (cd - pd).days
        except Exception:
            days = 0
        prior_lines.append(f"{i}. [{p_desc}] ({p_date}, {days} days before current)")

    # Check if all results are cached
    results = []
    uncached_indices = []
    for i, p in enumerate(prior_studies):
        p_desc = p.get("study_description", "")
        try:
            from datetime import date
            cd = date.fromisoformat(current_date)
            pd = date.fromisoformat(p.get("study_date", ""))
            days = (cd - pd).days
        except Exception:
            days = 0
        key = cache_key(current_desc, p_desc, days)
        if key in _cache:
            results.append(_cache[key])
        else:
            results.append(None)
            uncached_indices.append((i, key))

    if not uncached_indices:
        logger.info("All %d priors served from cache", len(prior_studies))
        return results

    # Build prompt only for uncached priors
    uncached_priors = [prior_studies[i] for i, _ in uncached_indices]
    uncached_lines = []
    for rank, (orig_i, _) in enumerate(uncached_indices, 1):
        uncached_lines.append(prior_lines[orig_i])

    user_msg = (
        f"Current exam: {current_desc} (date: {current_date})\n\n"
        f"Prior exams to evaluate:\n" +
        "\n".join(uncached_lines) +
        "\n\nReturn a JSON array of booleans for each prior exam listed above."
    )

    logger.info("LLM call: current=%r, %d uncached priors", current_desc, len(uncached_indices))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )

    raw = response.content[0].text.strip()
    logger.info("LLM raw response: %s", raw)

    # Parse JSON array from response
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        logger.warning("Could not parse LLM response, defaulting all to false")
        bools = [False] * len(uncached_indices)
    else:
        try:
            bools = json.loads(match.group(0))
            if len(bools) != len(uncached_indices):
                logger.warning("LLM returned %d bools, expected %d — padding/truncating",
                               len(bools), len(uncached_indices))
                while len(bools) < len(uncached_indices):
                    bools.append(False)
                bools = bools[:len(uncached_indices)]
        except json.JSONDecodeError:
            logger.warning("JSON parse error, defaulting all to false")
            bools = [False] * len(uncached_indices)

    # Store in cache and fill results
    for rank, (orig_i, key) in enumerate(uncached_indices):
        val = bool(bools[rank])
        _cache[key] = val
        results[orig_i] = val

    return results


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    cases = body.get("cases", [])
    logger.info("Request: %d cases", len(cases))

    predictions = []
    for case in cases:
        case_id = case.get("case_id")
        current_study = case.get("current_study", {})
        prior_studies = case.get("prior_studies", [])

        logger.info("Case %s: current=%r, %d priors",
                    case_id,
                    current_study.get("study_description"),
                    len(prior_studies))

        try:
            relevance = predict_case(current_study, prior_studies)
        except Exception as e:
            logger.error("Error on case %s: %s", case_id, e, exc_info=True)
            relevance = [False] * len(prior_studies)

        for prior, is_relevant in zip(prior_studies, relevance):
            predictions.append({
                "case_id": case_id,
                "study_id": prior.get("study_id"),
                "predicted_is_relevant": bool(is_relevant)
            })

    logger.info("Returning %d predictions", len(predictions))
    return jsonify({"predictions": predictions})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cache_size": len(_cache)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
