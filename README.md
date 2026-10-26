# Simple Data Processing AI

Local Python program that cleans a CSV, summarizes it, and prints heuristic insights. No API keys.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python process.py
python process.py path\to\your.csv --output output
```

Outputs:

- `output/cleaned.csv` — cleaned table
- `output/report.json` — numeric summary plus insights
