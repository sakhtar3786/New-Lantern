"""
Local test: sends a sample request to the running API and prints predictions.
Run the server first: python app.py
"""
import json
import requests

TEST_PAYLOAD = {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
        {
            "case_id": "1001016",
            "patient_id": "606707",
            "patient_name": "Andrews, Micheal",
            "current_study": {
                "study_id": "3100042",
                "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                "study_date": "2026-03-08"
            },
            "prior_studies": [
                {
                    "study_id": "2453245",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2020-03-08"
                },
                {
                    "study_id": "992654",
                    "study_description": "CT HEAD WITHOUT CNTRST",
                    "study_date": "2021-03-08"
                },
                {
                    "study_id": "111111",
                    "study_description": "X-RAY LEFT KNEE AP AND LATERAL",
                    "study_date": "2023-01-15"
                }
            ]
        }
    ]
}

resp = requests.post("http://localhost:8000/predict", json=TEST_PAYLOAD, timeout=60)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
