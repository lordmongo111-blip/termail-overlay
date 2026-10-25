# Macro / Regime Terminal — integriertes Dashboard

## Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## FRED-Key
Optional in `.streamlit/secrets.toml` ablegen:
```toml
FRED_API_KEY = "DEIN_KEY"
```
Keine Secrets committen oder teilen. Alternativ kann der Key links in der Sidebar eingegeben werden.

## Änderungen
- Die vorhandenen Module und Signal-Engine-Funktionen bleiben erhalten.
- Die Startseite lädt auf Knopfdruck FX-Proxy-Kurse, NQ/ES, VIX und ausgewählte FRED-Reihen.
- Datenalter, letzter Zeitstempel und Abruffehler werden sichtbar gemacht.
- Risk-on/off ist ausdrücklich ein heuristischer 20-Handelstage-Proxy, kein validiertes Handelssignal.
- Keine Demo-Werte als Live-Marktwerte; fehlende Reihen bleiben fehlend.
