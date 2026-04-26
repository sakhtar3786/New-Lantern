"""
Score your endpoint against the downloaded public eval JSON.

Usage:
  python eval.py --file public_eval.json --url http://localhost:8000/predict

Reports accuracy and lists wrong predictions.
"""

import argparse
import json
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_eval(file_path: str, url: str, batch_size: int = 50):
    with open(file_path) as f:
        data = json.load(f)

    all_cases = data["cases"]
    print(f"Loaded {len(all_cases)} cases")

    # Build ground truth: (case_id, study_id) -> bool
    ground_truth = {}
    for case in all_cases:
        case_id = case["case_id"]
        for prior in case.get("prior_studies", []):
            gt = prior.get("ground_truth_is_relevant")
            if gt is not None:
                ground_truth[(case_id, prior["study_id"])] = gt

    print(f"Ground truth entries: {len(ground_truth)}")

    # Send in batches
    all_predictions = []
    for i in range(0, len(all_cases), batch_size):
        batch = all_cases[i:i+batch_size]
        payload = {k: v for k, v in data.items() if k != "cases"}
        payload["cases"] = batch
        print(f"Sending batch {i//batch_size + 1}: cases {i}–{i+len(batch)-1}")
        resp = requests.post(url, json=payload, timeout=360)
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])
        all_predictions.extend(preds)

    print(f"Total predictions received: {len(all_predictions)}")

    correct = 0
    incorrect = 0
    skipped = 0

    for key, gt_val in ground_truth.items():
        matched = next(
            (p for p in all_predictions
             if p["case_id"] == key[0] and p["study_id"] == key[1]),
            None
        )
        if matched is None:
            skipped += 1
            incorrect += 1
        elif bool(matched["predicted_is_relevant"]) == bool(gt_val):
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0
    print(f"\n=== RESULTS ===")
    print(f"Correct:   {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Skipped:   {skipped}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to public eval JSON")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()
    run_eval(args.file, args.url, args.batch)
