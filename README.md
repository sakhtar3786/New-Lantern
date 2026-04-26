# Radiology Prior Relevance API

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py        # dev server on port 8000
```

## Production (gunicorn)
```bash
PORT=8000 gunicorn app:app --workers 2 --timeout 360 --bind 0.0.0.0:8000
```

## Deploy to Railway (recommended — free tier works)
1. Push this folder to a GitHub repo
2. Connect repo on railway.app → New Project → Deploy from GitHub
3. Set env var: `ANTHROPIC_API_KEY=sk-ant-...`
4. Railway auto-detects Procfile and deploys
5. Your public URL will be `https://your-app.up.railway.app/predict`

## Deploy to Render
1. New → Web Service → connect GitHub repo
2. Build: `pip install -r requirements.txt`
3. Start: `gunicorn app:app --workers 2 --timeout 360 --bind 0.0.0.0:$PORT`
4. Add env var `ANTHROPIC_API_KEY`

## Endpoints
- `POST /predict` — main prediction endpoint
- `GET /health` — health check, shows cache size

## Local Testing
```bash
python test_local.py
```

## Eval against public split
```bash
python eval.py --file public_eval.json --url http://localhost:8000/predict
```
