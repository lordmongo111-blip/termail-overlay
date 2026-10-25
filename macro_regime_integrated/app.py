import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Macro / Regime Terminal", page_icon="⌁", layout="wide")

G10 = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
MONTHS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,130,120,.08); padding: 12px; border-radius: 12px;}
.small-muted {color: #7c8982; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def get_fred_series(series_id, api_key, observation_start="2015-01-01"):
    """Fetch FRED observations server-side. Requires a free FRED API key."""
    import requests
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": observation_start, "sort_order": "asc"
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna()

def load_market_history(symbol, period="10y"):
    """Historical price helper via yfinance; availability/terms depend on symbol/provider."""
    import yfinance as yf
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    out = close.rename("close").to_frame().dropna()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.reset_index()

def seasonality(df):
    if df.empty or "close" not in df:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    d["month"] = d["date"].dt.month
    # Monthly close-to-close returns, grouped by calendar month over available history.
    monthly = d.set_index("date")["close"].resample("ME").last().dropna().pct_change().dropna().to_frame("return")
    monthly["month"] = monthly.index.month
    return monthly.groupby("month")["return"].agg(
        avg_return="mean",
        win_rate=lambda x: (x > 0).mean(),
        observations="count",
        median="median",
        volatility="std"
    ).reset_index()

def cot_api_preview():
    """Official CFTC Socrata API query: latest available records vary by dataset."""
    import requests
    # Public CFTC Legacy Futures Only dataset. Querying schema can change; fail visibly.
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {"$limit": 5, "$order": "report_date_as_yyyy_mm_dd DESC"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return pd.DataFrame(response.json())


# Official CFTC Legacy Futures Only mapping used by the signal engine.
# Mapping must be checked against the selected CFTC dataset and contract metadata.
CFTC_CONTRACT_CODES = {
    "6E": "099741",  # Euro FX
    "6B": "096742",  # British Pound
    "6J": "097741",  # Japanese Yen
    "6A": "232741",  # Australian Dollar
    "6C": "090741",  # Canadian Dollar
    "6S": "092741",  # Swiss Franc
    "6N": "112741",  # New Zealand Dollar
}

def load_cot_history(contract_code, limit=52):
    """Load weekly CFTC Legacy Futures Only net positions for one mapped contract."""
    import requests
    if contract_code not in CFTC_CONTRACT_CODES:
        raise ValueError(f"Kein verifiziertes CFTC-Mapping für {contract_code}")
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        "$where": f"cftc_contract_market_code='{CFTC_CONTRACT_CODES[contract_code]}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(max(26, min(int(limit), 104))),
        "$select": (
            "report_date_as_yyyy_mm_dd,market_and_exchange_names,"
            "cftc_contract_market_code,comm_positions_long_all,comm_positions_short_all,"
            "noncomm_positions_long_all,noncomm_positions_short_all,"
            "nonrept_positions_long_all,nonrept_positions_short_all,open_interest_all"
        ),
    }
    response = requests.get(url, params=params, timeout=20, headers={"Accept": "application/json"})
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise ValueError(f"CFTC lieferte keine Datensätze für Code {CFTC_CONTRACT_CODES[contract_code]}")
    df = pd.DataFrame(rows)
    required = ["report_date_as_yyyy_mm_dd", "comm_positions_long_all", "comm_positions_short_all",
                "noncomm_positions_long_all", "noncomm_positions_short_all"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("CFTC-Schema unvollständig: " + ", ".join(missing))
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce", utc=True)
    numeric = ["comm_positions_long_all", "comm_positions_short_all",
               "noncomm_positions_long_all", "noncomm_positions_short_all",
               "nonrept_positions_long_all", "nonrept_positions_short_all", "open_interest_all"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"] + required[1:]).sort_values("date").drop_duplicates("date")
    if len(df) < 26:
        raise ValueError(f"Zu wenig COT-Historie ({len(df)}/26 Wochen)")
    df["commercial_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    df["noncommercial_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    if {"nonrept_positions_long_all", "nonrept_positions_short_all"}.issubset(df.columns):
        df["nonreportable_net"] = df["nonrept_positions_long_all"] - df["nonrept_positions_short_all"]
    else:
        df["nonreportable_net"] = np.nan
    df["open_interest"] = pd.to_numeric(df.get("open_interest_all"), errors="coerce")
    age_days = (pd.Timestamp.now(tz="UTC") - df["date"].iloc[-1]).total_seconds() / 86400
    if age_days > 14:
        raise ValueError(f"COT-Bericht veraltet ({age_days:.1f} Tage; Limit 14)")
    df.attrs["market_name"] = str(df.iloc[-1].get("market_and_exchange_names", contract_code))
    df.attrs["age_days"] = age_days
    return df.reset_index(drop=True)

def cot_positioning_score(cot_df, invert=False):
    """Map latest non-commercial net positioning to a bounded 52-week percentile score."""
    if cot_df is None or len(cot_df) < 26 or "noncommercial_net" not in cot_df:
        raise ValueError("COT-Historie reicht für einen Positionsscore nicht aus")
    values = pd.to_numeric(cot_df["noncommercial_net"], errors="coerce").dropna().tail(52)
    if len(values) < 26:
        raise ValueError("Zu wenige gültige Non-commercial COT-Werte")
    latest = float(values.iloc[-1])
    # Percentile rank of latest net position within the available rolling history.
    percentile = float((values <= latest).mean())
    score = int(round(np.clip((percentile - 0.5) * 200, -100, 100)))
    if invert:
        score = -score
    return score, latest, percentile

def seasonality_component_score(prices, invert=False, min_years=5):
    """Score this calendar month's historical mean return; needs >=5 observations."""
    if prices is None or prices.empty or not {"date", "close"}.issubset(prices.columns):
        raise ValueError("Historische Kursdaten für Saisonalität fehlen")
    d = prices.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date")
    monthly = d.set_index("date")["close"].resample("ME").last().dropna().pct_change().dropna().to_frame("return")
    current_month = pd.Timestamp.now().month
    sample = monthly.loc[monthly.index.month == current_month, "return"].dropna()
    # Count distinct years, not raw rows; only one monthly return per year is valid.
    sample = sample[~sample.index.year.duplicated(keep="last")]
    if len(sample) < min_years:
        years_found = ", ".join(str(y) for y in sample.index.year.tolist()) or "keine"
        raise ValueError(
            f"Zu wenige historische Monatsbeobachtungen ({len(sample)}/{min_years} Jahre; vorhanden: {years_found}). "
            "Für eine belastbare Saisonanalyse werden längere Kursdaten benötigt."
        )
    mean_return = float(sample.mean())
    # Transparent scaling: +/-2% average monthly return maps to +/-100.
    score = int(round(np.clip(mean_return / 0.02 * 100, -100, 100)))
    if invert:
        score = -score
    return score, mean_return, len(sample)


# ---------------- FX signal engine: fail closed on missing/invalid inputs ----------------
FX_PAIRS = {
    "EUR/USD": {"symbol": "EURUSD=X", "base": "EUR", "quote": "USD", "cot": "6E", "cot_invert": False},
    "GBP/USD": {"symbol": "GBPUSD=X", "base": "GBP", "quote": "USD", "cot": "6B", "cot_invert": False},
    "USD/JPY": {"symbol": "JPY=X", "base": "USD", "quote": "JPY", "cot": "6J", "cot_invert": True},
    "AUD/USD": {"symbol": "AUDUSD=X", "base": "AUD", "quote": "USD", "cot": "6A", "cot_invert": False},
    "USD/CAD": {"symbol": "CAD=X", "base": "USD", "quote": "CAD", "cot": "6C", "cot_invert": True},
    "USD/CHF": {"symbol": "CHF=X", "base": "USD", "quote": "CHF", "cot": "6S", "cot_invert": True},
    "NZD/USD": {"symbol": "NZDUSD=X", "base": "NZD", "quote": "USD", "cot": "6N", "cot_invert": False},
    "EUR/GBP": {"symbol": "EURGBP=X", "base": "EUR", "quote": "GBP", "cot": None, "cot_invert": False},
    "EUR/JPY": {"symbol": "EURJPY=X", "base": "EUR", "quote": "JPY", "cot": "6J", "cot_invert": True},
    "GBP/JPY": {"symbol": "GBPJPY=X", "base": "GBP", "quote": "JPY", "cot": "6J", "cot_invert": True},
    "EUR/CHF": {"symbol": "EURCHF=X", "base": "EUR", "quote": "CHF", "cot": "6S", "cot_invert": True},
    "AUD/JPY": {"symbol": "AUDJPY=X", "base": "AUD", "quote": "JPY", "cot": "6J", "cot_invert": True},
    "CAD/JPY": {"symbol": "CADJPY=X", "base": "CAD", "quote": "JPY", "cot": "6J", "cot_invert": True},
}
LONG_SHORT_THRESHOLD = 20
MAX_PRICE_AGE_DAYS = 4
MIN_PRICE_OBSERVATIONS = 220


def validate_price_history(df, max_age_days=MAX_PRICE_AGE_DAYS):
    """Validate real provider-returned daily prices; never create replacement data."""
    if df is None or df.empty or not {"date", "close"}.issubset(df.columns):
        return False, "Kursreihe fehlt", None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date")
    if len(d) < MIN_PRICE_OBSERVATIONS:
        return False, f"Zu wenig Historie ({len(d)}/{MIN_PRICE_OBSERVATIONS} Tageswerte)", None
    if (d["close"] <= 0).any() or d["date"].duplicated().any():
        return False, "Ungültige Kurse oder doppelte Zeitstempel", None
    age = (pd.Timestamp.now(tz="UTC") - d["date"].iloc[-1]).total_seconds() / 86400
    if age > max_age_days:
        return False, f"Kursdaten veraltet ({age:.1f} Tage)", None
    return True, f"Validiert · {len(d)} Tageswerte · letzter Kurs {d['date'].iloc[-1].date()}", d


def technical_confirmation(prices):
    """Daily trend/momentum confirmation, returned only for validated prices."""
    if prices is None or len(prices) < 220:
        return None, "Technische Historie unzureichend"
    close = prices["close"].astype(float)
    last = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    momentum20 = float(close.pct_change(20).iloc[-1])
    if not np.isfinite([last, sma50, sma200, momentum20]).all():
        return None, "Technische Kennzahlen ungültig"
    if last > sma50 and sma50 > sma200 and momentum20 > 0:
        return 1, f"Bullish bestätigt: Kurs>SMA50>SMA200, 20T-Momentum {momentum20:.2%}"
    if last < sma50 and sma50 < sma200 and momentum20 < 0:
        return -1, f"Bearish bestätigt: Kurs<SMA50<SMA200, 20T-Momentum {momentum20:.2%}"
    return 0, f"Kein eindeutiger Trend: 20T-Momentum {momentum20:.2%}"


def combine_fx_signal(component_scores, component_valid, technical_score):
    """Weighted composite: macro 40%, COT 25%, seasonality 10%, technical 25%.

    Every input is required and must be validated. Technical score is -1/0/+1,
    scaled to -100/0/+100; a neutral technical read cannot confirm a direction.
    """
    required = ("macro", "cot", "seasonality")
    missing = [name for name in required if not component_valid.get(name, False)]
    if missing or technical_score is None:
        blocked = missing + (["Technik"] if technical_score is None else [])
        return {"bias": "NEUTRAL", "score": None,
                "reason": "Signal gesperrt: nicht validiert — " + ", ".join(blocked),
                "signal_quality": "NICHT BEWERTBAR"}
    if any(name not in component_scores for name in required):
        return {"bias": "NEUTRAL", "score": None,
                "reason": "Signal gesperrt: für mindestens eine validierte Komponente fehlt der Score",
                "signal_quality": "NICHT BEWERTBAR"}
    score = int(round(
        0.40 * component_scores["macro"] +
        0.25 * component_scores["cot"] +
        0.10 * component_scores["seasonality"] +
        0.25 * (100 * technical_score)
    ))
    if score >= LONG_SHORT_THRESHOLD:
        if technical_score == 1:
            bias, reason = "LONG", "LONG: gewichteter Score ≥ +20 und Technik bestätigt bullish"
        else:
            bias, reason = "NEUTRAL", "LONG blockiert: Technik bestätigt nicht bullish"
    elif score <= -LONG_SHORT_THRESHOLD:
        if technical_score == -1:
            bias, reason = "SHORT", "SHORT: gewichteter Score ≤ −20 und Technik bestätigt bearish"
        else:
            bias, reason = "NEUTRAL", "SHORT blockiert: Technik bestätigt nicht bearish"
    else:
        bias, reason = "NEUTRAL", "Gewichteter Score innerhalb der Schwelle ±20"
    quality = "MITTEL" if bias != "NEUTRAL" else "NIEDRIG"
    return {"bias": bias, "score": score, "reason": reason, "signal_quality": quality}


# FRED-hosted national/OECD series. For non-US currencies, the short-term
# interest-rate series is a proxy (not necessarily the official policy rate).
# CPI series are price indices; their YoY change is calculated below.
MACRO_CURRENCY_SERIES = {
    "USD": {"rate": "FEDFUNDS", "inflation": "CPIAUCSL", "rate_age": 100, "inflation_age": 100, "label": "US Fed funds"},
    "EUR": {"rate": "ECBDFR", "inflation": "CP0000EZCCM086NEST", "rate_age": 60, "inflation_age": 100, "label": "EZB Einlagezins"},
    "GBP": {"rate": "IRSTCI01GBM156N", "inflation": "GBRCPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "UK Kurzfristzins-Proxy"},
    "JPY": {"rate": "IRSTCI01JPM156N", "inflation": "JPNCPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "Japan Kurzfristzins-Proxy"},
    "CHF": {"rate": "IRSTCI01CHM156N", "inflation": "CHECPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "Schweiz Kurzfristzins-Proxy"},
    "CAD": {"rate": "IRSTCI01CAM156N", "inflation": "CANCPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "Kanada Kurzfristzins-Proxy"},
    "AUD": {"rate": "IRSTCI01AUM156N", "inflation": "AUSCPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "Australien Kurzfristzins-Proxy"},
    "NZD": {"rate": "IRSTCI01NZM156N", "inflation": "NZLCPIALLMINMEI", "rate_age": 100, "inflation_age": 100, "label": "Neuseeland Kurzfristzins-Proxy"},
}

def calculate_relative_macro_score(api_key, base_currency, quote_currency):
    """Compare base and quote macro data. Rates are required; stale CPI is disclosed."""
    if base_currency not in MACRO_CURRENCY_SERIES or quote_currency not in MACRO_CURRENCY_SERIES:
        return None, f"Keine Makro-Zuordnung für {base_currency}/{quote_currency}"
    if base_currency == quote_currency:
        return None, "Basis- und Quotewährung sind identisch"

    loaded = {}
    errors = []
    for currency in (base_currency, quote_currency):
        cfg = MACRO_CURRENCY_SERIES[currency]
        # Rate series are mandatory: without both, no relative macro score is calculated.
        sid = cfg["rate"]
        try:
            df = get_fred_series(sid, api_key)
            if df.empty:
                raise ValueError("keine Beobachtungen")
            df = df.sort_values("date").reset_index(drop=True)
            last_date = pd.Timestamp(df.iloc[-1]["date"])
            age = (pd.Timestamp.now().normalize() - last_date.normalize()).days
            if age < 0 or age > cfg["rate_age"]:
                raise ValueError(f"letzter Wert {last_date.date()} ist {age} Tage alt")
            loaded[(currency, "rate")] = df
        except Exception as exc:
            return None, f"{currency} Zinsreihe {sid} nicht verfügbar/aktuell: {exc}"

        # Inflation is useful but some FRED/OECD CPI index series are discontinued or stale.
        # Do not silently accept stale inflation; transparently fall back to a rate-only score.
        inf_sid = cfg["inflation"]
        try:
            inf = get_fred_series(inf_sid, api_key)
            if inf.empty or len(inf) < 13:
                raise ValueError("zu wenige Beobachtungen")
            inf = inf.sort_values("date").reset_index(drop=True)
            inf_date = pd.Timestamp(inf.iloc[-1]["date"])
            inf_age = (pd.Timestamp.now().normalize() - inf_date.normalize()).days
            if inf_age < 0 or inf_age > cfg["inflation_age"]:
                raise ValueError(f"letzter Wert {inf_date.date()} ist {inf_age} Tage alt")
            loaded[(currency, "inflation")] = inf
        except Exception as exc:
            errors.append(f"{currency}-Inflation ({inf_sid}) nicht aktuell: {exc}")

    base_rate = float(loaded[(base_currency, "rate")].iloc[-1]["value"])
    quote_rate = float(loaded[(quote_currency, "rate")].iloc[-1]["value"])
    rate_diff = base_rate - quote_rate
    rate_component = float(np.clip(rate_diff / 2.0 * 100, -100, 100))

    base_infl = quote_infl = None
    if not errors:
        def yoy(currency):
            df = loaded[(currency, "inflation")]
            last = df.iloc[-1]
            target = pd.Timestamp(last["date"]) - pd.DateOffset(years=1)
            prior = df[df["date"] <= target]
            if prior.empty or float(prior.iloc[-1]["value"]) == 0:
                raise ValueError(f"Kein passender Vorjahreswert für {currency}-Inflation")
            return (float(last["value"]) / float(prior.iloc[-1]["value"]) - 1) * 100
        try:
            base_infl, quote_infl = yoy(base_currency), yoy(quote_currency)
        except Exception as exc:
            errors.append(f"Inflationsvergleich nicht möglich: {exc}")

    base_label = MACRO_CURRENCY_SERIES[base_currency]["label"]
    quote_label = MACRO_CURRENCY_SERIES[quote_currency]["label"]
    if base_infl is not None and quote_infl is not None:
        inflation_diff = base_infl - quote_infl
        inflation_component = float(np.clip(-inflation_diff / 2.0 * 100, -100, 100))
        score = int(round(0.65 * rate_component + 0.35 * inflation_component))
        details = (f"{base_currency} ({base_label}) {base_rate:.2f}% vs. "
                   f"{quote_currency} ({quote_label}) {quote_rate:.2f}% (Zinsdifferenz {rate_diff:+.2f} pp); "
                   f"Inflation YoY {base_currency} {base_infl:.2f}% vs. {quote_currency} {quote_infl:.2f}% "
                   f"(Differenz {inflation_diff:+.2f} pp).")
        mode = "Zinsen + Inflation"
    else:
        score = int(round(rate_component))
        details = (f"TEILWEISE: nur Zinsdifferenz verwendet. {base_currency} ({base_label}) "
                   f"{base_rate:.2f}% vs. {quote_currency} ({quote_label}) {quote_rate:.2f}% "
                   f"(Differenz {rate_diff:+.2f} pp). Inflation wurde wegen fehlender/veralteter Reihe "
                   f"nicht verwendet. " + " | ".join(errors))
        mode = "Nur Zinsdaten"

    return {"score": score, "rate_diff": rate_diff, "base_inflation_yoy": base_infl,
            "quote_inflation_yoy": quote_infl, "details": details, "mode": mode,
            "partial": base_infl is None or quote_infl is None}, None

def calculate_eurusd_macro_score(api_key):
    """Backward-compatible wrapper for the EUR/USD relative macro score."""
    return calculate_relative_macro_score(api_key, "EUR", "USD")

def quality_percent(valid_flags):
    if not valid_flags:
        return 0
    return round(100 * sum(bool(x) for x in valid_flags) / len(valid_flags))

st.title("⌁ MACRO / REGIME TERMINAL")
st.caption("Forex only · Makro · COT · Saisonalität · technische Bestätigung")
st.warning("Datenstatus: Ohne konfigurierte Datenquelle werden keine aktuellen Marktmeinungen behauptet. Historische Daten sind nicht automatisch Echtzeitdaten; Anbieter, Verzögerung und Lizenzbedingungen beachten.")

with st.sidebar:
    st.header("Einstellungen")
    page = st.radio("Modul", ["Übersicht", "FX Signal", "ML Lab", "Saisonalität", "COT / Positioning", "NQ & ES", "Quant & Regression", "Datenquellen"])
    st.divider()
    try:
        saved_fred_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        saved_fred_key = ""
    fred_key = st.text_input(
        "FRED API-Key (optional)", value=saved_fred_key, type="password",
        help="Hier eingeben oder sicher in Streamlit Secrets als FRED_API_KEY hinterlegen."
    )
    st.caption(f"Ansicht erstellt: {utc_now()}")
    st.caption("Build: FX Signal Engine v1.2 · FX + COT · Bias nur aus validierten Daten")

# Demo values are intentionally labeled and are not market data.
demo_strength = pd.DataFrame({
    "Währung": G10,
    "Score": [0.8, 0.3, 0.5, -0.9, 0.1, 0.2, -0.4, -0.6]
})
demo_strength["Status"] = "DEMO"

if page == "Übersicht":
    st.subheader("Zentrales Dashboard")
    st.caption("Ein gemeinsamer Überblick über FX, US-Makro und Cross-Asset-Proxys. Daten werden nur auf Knopfdruck abgerufen; Yahoo Finance ist kein garantierter Echtzeitfeed.")
    st.warning("Regime-Einschätzung ist ein transparenter Proxy aus NQ/ES/VIX und kein validiertes Handelssignal. Fehlende Daten bleiben fehlend.")

    if "dashboard_snapshot" not in st.session_state:
        st.session_state["dashboard_snapshot"] = None

    dcol1, dcol2 = st.columns([1, 3])
    with dcol1:
        refresh_dashboard = st.button("Dashboard aktualisieren", type="primary", use_container_width=True)
    with dcol2:
        st.caption("Abruf: tägliche historische Kurse via yfinance sowie ausgewählte FRED-Reihen. Der FRED-Key wird aus Sidebar/Secrets verwendet.")

    if refresh_dashboard:
        price_rows, price_frames, fred_rows, errors = [], {}, [], []
        dashboard_symbols = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
                             "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
                             "NZD/USD": "NZDUSD=X", "Nasdaq-100 Future": "NQ=F",
                             "S&P 500 Future": "ES=F", "VIX": "^VIX"}
        with st.spinner("Lade Markt- und Makrodaten…"):
            for label, symbol in dashboard_symbols.items():
                try:
                    hist = load_market_history(symbol, "1y")
                    if hist.empty:
                        raise ValueError("keine Kursdaten zurückgegeben")
                    hist = hist.copy()
                    hist["date"] = pd.to_datetime(hist["date"], utc=True, errors="coerce")
                    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
                    hist = hist.dropna(subset=["date", "close"]).sort_values("date")
                    if hist.empty or float(hist.iloc[-1]["close"]) <= 0:
                        raise ValueError("keine gültigen Schlusskurse")
                    last = hist.iloc[-1]
                    age_days = (pd.Timestamp.now(tz="UTC") - last["date"]).total_seconds() / 86400
                    ret20 = float(hist["close"].pct_change(20).iloc[-1] * 100) if len(hist) > 20 else np.nan
                    price_rows.append({"Markt": label, "Symbol": symbol, "Letzter Kurs": float(last["close"]),
                                       "Letztes Datum (UTC)": last["date"].date().isoformat(),
                                       "Alter (Tage)": round(age_days, 1), "20T-Rendite (%)": ret20,
                                       "Status": "OK" if age_days <= 4 else "VERALTET"})
                    price_frames[label] = hist
                except Exception as exc:
                    errors.append(f"{label} ({symbol}): {exc}")
            fred_map = {"FEDFUNDS": ("Fed Funds Rate", "% p.a."), "DGS10": ("US Treasury 10Y", "% p.a."),
                        "DGS2": ("US Treasury 2Y", "% p.a."), "DTWEXBGS": ("Breiter USD-Index", "Index")}
            if fred_key:
                for sid, (label, unit) in fred_map.items():
                    try:
                        series = get_fred_series(sid, fred_key)
                        if series.empty:
                            raise ValueError("keine Beobachtungen")
                        last = series.iloc[-1]
                        fred_rows.append({"Indikator": label, "FRED-Serie": sid,
                                          "Letzter Wert": float(last["value"]), "Einheit": unit,
                                          "Beobachtungsdatum": pd.Timestamp(last["date"]).date().isoformat(),
                                          "Status": "OK"})
                    except Exception as exc:
                        errors.append(f"FRED {sid}: {exc}")
            else:
                errors.append("FRED: API-Key fehlt; US-Makroindikatoren wurden übersprungen.")
        st.session_state["dashboard_snapshot"] = {"prices": price_rows, "frames": price_frames,
                                                    "fred": fred_rows, "errors": errors,
                                                    "updated": utc_now()}

    snap = st.session_state.get("dashboard_snapshot")
    if snap is None:
        st.info("Klicke auf „Dashboard aktualisieren“, um aktuelle verfügbare Daten abzurufen. Bis dahin werden bewusst keine Marktwerte oder Scores erfunden.")
        st.markdown("#### Enthaltene Bereiche")
        st.write("FX-Spot-Proxys · NQ/ES und VIX · FRED-Zinsen und USD-Index · Datenalter und Fehlerstatus")
    else:
        price_df = pd.DataFrame(snap["prices"])
        fred_df = pd.DataFrame(snap["fred"])
        st.caption(f"Letzter Dashboard-Abruf: {snap['updated']}")
        if not price_df.empty:
            def _row(label):
                found = price_df.loc[price_df["Markt"] == label]
                return found.iloc[0] if not found.empty else None
            nq, es, vix = _row("Nasdaq-100 Future"), _row("S&P 500 Future"), _row("VIX")
            fresh = int((price_df["Status"] == "OK").sum())
            total = len(price_df)
            regime = "UNBESTIMMT"
            regime_note = "NQ/ES/VIX-Daten fehlen oder sind nicht vollständig frisch."
            if nq is not None and es is not None and vix is not None and all(r["Status"] == "OK" for r in (nq, es, vix)):
                if nq["20T-Rendite (%)"] > 0 and es["20T-Rendite (%)"] > 0 and vix["20T-Rendite (%)"] <= 0:
                    regime, regime_note = "RISK-ON (PROXY)", "NQ und ES positiv über 20 Handelstage, VIX nicht gestiegen."
                elif nq["20T-Rendite (%)"] < 0 and es["20T-Rendite (%)"] < 0 and vix["20T-Rendite (%)"] > 0:
                    regime, regime_note = "RISK-OFF (PROXY)", "NQ und ES negativ über 20 Handelstage, VIX gestiegen."
                else:
                    regime, regime_note = "GEMISCHT / NEUTRAL", "Cross-Asset-Bedingungen widersprechen sich oder sind uneindeutig."
            fx_labels = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
            fx = price_df[price_df["Markt"].isin(fx_labels) & (price_df["Status"] == "OK")]
            strongest = weakest = "—"
            if len(fx) >= 2:
                # Directional 20d proxy is pair-based and is NOT a normalized G10 currency-strength model.
                strongest = str(fx.sort_values("20T-Rendite (%)", ascending=False).iloc[0]["Markt"])
                weakest = str(fx.sort_values("20T-Rendite (%)").iloc[0]["Markt"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Risk Regime", regime)
            k2.metric("FX-Paare verfügbar", f"{len(fx)}/7")
            k3.metric("Frische Marktserien", f"{fresh}/{total}")
            k4.metric("Letzter Abruf", snap["updated"].split(" ")[1])
            st.caption(regime_note)
            st.markdown("#### FX- und Cross-Asset-Märkte")
            st.dataframe(price_df, use_container_width=True, hide_index=True)
            charts = st.columns(2)
            for col, label in zip(charts, ["Nasdaq-100 Future", "S&P 500 Future"]):
                hist = snap["frames"].get(label)
                with col:
                    st.markdown(f"**{label}**")
                    if hist is not None and not hist.empty:
                        st.line_chart(hist.set_index("date")["close"])
                    else:
                        st.caption("Keine Daten verfügbar.")
        else:
            st.error("Keine Kursdaten verfügbar; das Regime bleibt unbestimmt.")
        st.markdown("#### US-Makro (FRED)")
        if not fred_df.empty:
            st.dataframe(fred_df, use_container_width=True, hide_index=True)
        else:
            st.info("Keine FRED-Reihen geladen. FRED_API_KEY in Streamlit Secrets konfigurieren oder links in der Sidebar eingeben.")
        if snap["errors"]:
            with st.expander(f"Datenprobleme ({len(snap['errors'])})", expanded=False):
                for err in snap["errors"]:
                    st.write(f"- {err}")
        st.caption("Hinweis: „Stärkste/Schwächste“ wird hier bewusst nicht als Währungsstärke ausgegeben. Paar-Renditen sind nicht über alle G10-Währungen normalisiert. FX Signal, ML Lab, COT, Saisonalität, NQ & ES und Quant & Regression bleiben eigenständige Module.")

elif page in ("FX Signal", "FX Macro"):
    st.subheader("Forex Bias Engine")
    st.caption("LONG/SHORT erst ab ±20 und nur, wenn Makro, COT, Saisonalität sowie technische Bestätigung validiert sind. Nicht verbundene Faktoren erzwingen NEUTRAL.")
    pair = st.selectbox("Forex-Paar", list(FX_PAIRS.keys()))
    pair_info = FX_PAIRS[pair]
    st.markdown("#### Datenvalidierung & Bias")
    if st.button("Echte Kursdaten laden und validieren", type="primary"):
        try:
            price_df = load_market_history(pair_info["symbol"], "10y")
            price_ok, price_message, validated_prices = validate_price_history(price_df)
            tech_score, tech_message = technical_confirmation(validated_prices) if price_ok else (None, "Technik blockiert: Kursdaten nicht validiert")
            component_valid = {"macro": False, "cot": False, "seasonality": False}
            component_scores = {}
            macro_result, macro_error = (None, "FRED API-Key fehlt")
            if fred_key:
                try:
                    macro_result, macro_error = calculate_relative_macro_score(
                        fred_key, pair_info["base"], pair_info["quote"]
                    )
                    if macro_result is not None:
                        component_valid["macro"] = True
                        component_scores["macro"] = macro_result["score"]
                except Exception as macro_exc:
                    macro_error = str(macro_exc)
            cot_result, cot_error = None, "COT noch nicht geladen"
            season_result, season_error = None, "Saisonalität noch nicht berechnet"
            # COT is loaded from the public CFTC API and accepted only when mapping,
            # history length, schema and report freshness pass validation.
            try:
                cot_df = load_cot_history(pair_info["cot"], limit=52)
                cot_score, cot_net, cot_pct = cot_positioning_score(cot_df, invert=pair_info.get("cot_invert", False))
                cot_result = {"score": cot_score, "net": cot_net, "percentile": cot_pct,
                              "date": cot_df.iloc[-1]["date"].date().isoformat(),
                              "market": cot_df.attrs.get("market_name", pair_info["cot"]),
                              "age_days": cot_df.attrs.get("age_days", None)}
                component_valid["cot"] = True
                component_scores["cot"] = cot_score
            except Exception as cot_exc:
                cot_error = str(cot_exc)
            # Seasonality uses the same actual price history; no synthetic fallback.
            try:
                if price_ok:
                    season_score, mean_return, season_n = seasonality_component_score(
                        validated_prices, invert=pair_info["symbol"] in ("JPY=X", "CAD=X", "CHF=X")
                    )
                    season_result = {"score": season_score, "mean_return": mean_return, "n": season_n}
                    component_valid["seasonality"] = True
                    component_scores["seasonality"] = season_score
                else:
                    season_error = "Kursreihe nicht validiert; Saisonalität blockiert"
            except Exception as season_exc:
                season_error = str(season_exc)
            # Composite score is calculated once in the engine; missing inputs fail closed.
            result = combine_fx_signal(component_scores, component_valid, tech_score)
            flags = [price_ok, component_valid["macro"], component_valid["cot"], component_valid["seasonality"], tech_score is not None]
            q_data = quality_percent(flags)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bias", result["bias"])
            c2.metric("Gesamtscore", "—" if result["score"] is None else f"{result['score']:+d} / 100")
            c3.metric("Datenqualität", f"{q_data}%")
            c4.metric("Signalqualität", result["signal_quality"])
            st.info(result["reason"])
            # Separate provisional market direction from the fully validated signal.
            # This helps every pair show the available evidence without pretending
            # that missing relative macro data has been validated.
            provisional_parts = []
            provisional_weights = []
            if component_valid.get("cot") and "cot" in component_scores:
                provisional_parts.append(0.45 * component_scores["cot"])
                provisional_weights.append(0.45)
            if component_valid.get("seasonality") and "seasonality" in component_scores:
                provisional_parts.append(0.15 * component_scores["seasonality"])
                provisional_weights.append(0.15)
            if tech_score is not None:
                provisional_parts.append(0.40 * (100 * tech_score))
                provisional_weights.append(0.40)
            st.markdown("#### Vorläufige Richtung (verfügbare Daten, noch kein bestätigtes Handelssignal)")
            if provisional_weights:
                provisional_score = int(round(sum(provisional_parts) / sum(provisional_weights)))
                if provisional_score >= LONG_SHORT_THRESHOLD and tech_score == 1:
                    provisional_bias = "BULLISH / LONG-TENDENZ"
                    st.success(f"{provisional_bias} · Score {provisional_score:+d}/100. COT, Saisonalität und Technik soweit verfügbar; Makrovergleich fehlt ggf. noch.")
                elif provisional_score <= -LONG_SHORT_THRESHOLD and tech_score == -1:
                    provisional_bias = "BEARISH / SHORT-TENDENZ"
                    st.error(f"{provisional_bias} · Score {provisional_score:+d}/100. COT, Saisonalität und Technik soweit verfügbar; Makrovergleich fehlt ggf. noch.")
                else:
                    provisional_bias = "NEUTRAL"
                    st.info(f"{provisional_bias} · Score {provisional_score:+d}/100. Die verfügbaren Faktoren zeigen keine ausreichend bestätigte Richtung.")
                st.caption("Diese vorläufige Richtung ersetzt nicht den finalen Bias. Der finale LONG/SHORT-Status bleibt gesperrt, solange erforderliche Makro-, COT-, Saison- oder Technikdaten fehlen bzw. nicht validiert sind.")
            else:
                st.warning("Vorläufige Richtung nicht bewertbar: Es liegen noch keine validierten COT-, Saison- oder Technikwerte vor.")
            # Explain the direction consistently for every selectable FX pair.
            if result["bias"] == "LONG":
                st.success(f"Einordnung für {pair}: LONG bedeutet, dass die Basiswährung ({pair_info['base']}) gegenüber der Kurswährung ({pair_info['quote']}) bevorzugt wird. Das ist nur ein validiertes Setup, keine Garantie.")
            elif result["bias"] == "SHORT":
                st.error(f"Einordnung für {pair}: SHORT bedeutet, dass die Basiswährung ({pair_info['base']}) gegenüber der Kurswährung ({pair_info['quote']}) schwächer eingeschätzt wird.")
            else:
                st.info(f"Einordnung für {pair}: NEUTRAL — aktuell kein freigegebenes LONG-/SHORT-Signal. Fehlende oder nicht bestätigte Faktoren werden nicht durch Schätzwerte ersetzt.")
            st.caption(
                "Die vorläufige Richtung wird für alle auswählbaren Paare aus den jeweils verfügbaren, "
                "validierten Nicht-Makro-Faktoren berechnet. Der finale Bias wird nur freigegeben, wenn "
                "Makro, COT, Saisonalität und Technik vollständig validiert sind. Für die hinterlegten "
                "Währungen wird der relative Makrovergleich aus Zins- und Inflationsreihen berechnet; "
                "außerhalb USD/EUR dienen teils kurzfristige Zinsreihen als Proxy."
            )
            if result["bias"] == "NEUTRAL":
                missing_names = {
                    "macro": "relativer Makrovergleich fehlt oder ist veraltet",
                    "cot": "COT-Positionierung fehlt oder ist nicht frisch",
                    "seasonality": "Saisonalität ist nicht ausreichend belegt",
                }
                missing_readable = [
                    missing_names.get(name, name)
                    for name in ("macro", "cot", "seasonality")
                    if not component_valid.get(name, False)
                ]
                if tech_score is None:
                    missing_readable.append("technische Daten sind nicht validiert")
                elif tech_score == 0:
                    missing_readable.append("Technik bestätigt weder bullish noch bearish")
                st.info("Warum NEUTRAL? " + (
                    "; ".join(missing_readable) if missing_readable
                    else "Der gewichtete Score erreicht die Schwelle nicht oder Technik bestätigt die Richtung nicht."
                ) + ".")
            st.write("**Technische Prüfung:**", tech_message)
            st.write("**Kursdaten:**", price_message)
            st.markdown(f"#### Relativer Makro-Score ({pair_info['base']} gegenüber {pair_info['quote']})")
            if macro_result is not None:
                st.metric(f"Makro-Score {pair}", f"{macro_result['score']:+d} / 100")
                st.caption(macro_result["details"])
                st.caption("Heuristik: Zinsdifferenz 65%, Inflationsdifferenz 35%. Ein positiver Wert begünstigt die Basiswährung relativ zur Quote-Währung; kein alleiniger Trade-Trigger.")
            else:
                st.warning("Makro-Score nicht verfügbar: " + str(macro_error) + ". Prüfe FRED-Key und Serienverfügbarkeit.")
            quality_rows = [
                {"Komponente": "FX-Kursdaten", "Status": "VALIDIERT" if price_ok else "UNGÜLTIG", "Einordnung": price_message},
                {"Komponente": "Relatives Makro beider Währungen", "Status": "VALIDIERT" if component_valid["macro"] else "FEHLT", "Einordnung": macro_result["details"] if macro_result is not None else str(macro_error)},
                {"Komponente": "COT", "Status": "VALIDIERT" if cot_result is not None else "FEHLT", "Einordnung": (f"{cot_result['market']} · Report {cot_result['date']} · Non-commercial net {cot_result['net']:,.0f} · Score {cot_result['score']:+d} · {cot_result['age_days']:.1f} Tage alt" if cot_result is not None else cot_error)},
                {"Komponente": "Saisonalität", "Status": "VALIDIERT" if season_result is not None else "FEHLT", "Einordnung": (f"Kalendermonat {datetime.now().strftime('%B')} · Ø {season_result['mean_return']:.2%} über {season_result['n']} Jahre · Score {season_result['score']:+d}" if season_result is not None else season_error)},
                {"Komponente": "Technische Bestätigung", "Status": "VALIDIERT" if tech_score is not None else "FEHLT", "Einordnung": tech_message},
            ]
            st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
            cota, cotb, cotc = st.columns(3)
            if cot_result is not None:
                cota.metric("COT-Score", f"{cot_result['score']:+d} / 100")
                cotb.metric("Non-commercial net", f"{cot_result['net']:,.0f}")
                cotc.metric("COT-Bericht", cot_result["date"])
                st.caption(f"CFTC Legacy Futures Only · {cot_result['market']} · {cot_result['age_days']:.1f} Tage alt. Non-commercial net ist Futures-Positionierung, kein vollständiges Spot-FX-Orderflow-Maß.")
            else:
                cota.warning("COT nicht validiert: " + cot_error)
            if season_result is not None:
                st.caption(f"Saisonalität: aktueller Kalendermonat Ø {season_result['mean_return']:.2%} aus {season_result['n']} historischen Monatsbeobachtungen; Score {season_result['score']:+d}.")
            else:
                st.caption("Saisonalität nicht freigegeben: " + season_error)
            if price_ok:
                st.line_chart(validated_prices.set_index("date")["close"])
            st.warning("Fail-closed aktiv: Der finale Bias bleibt NEUTRAL, bis Makro, COT und Saisonalität aus echten, korrekt zugeordneten und frischen Daten validiert sind. Der Makro-Score allein ist noch kein Gesamt- oder Handelssignal.")
        except Exception as exc:
            st.error(f"Kursabruf/Validierung fehlgeschlagen: {exc}")
            st.metric("Bias", "NEUTRAL")
            st.metric("Datenqualität", "0% / nicht validiert")
    else:
        st.info("Noch keine aktuelle Kursprüfung durchgeführt. Ohne erfolgreichen Datenabruf bleibt der Bias NEUTRAL.")
    st.divider()
    st.subheader("Makro-Datenlabor (USD, nicht allein signalberechtigt)")
    st.caption("Lädt verfügbare US-Makroserien aus FRED. Das ist noch kein vollständiger G10-Vergleich; andere Länder benötigen eigene Datenzuordnungen.")
    fred_series = {
        "FEDFUNDS": ("US Effective Federal Funds Rate", "% p.a."),
        "CPIAUCSL": ("US CPI Index", "Index"),
        "UNRATE": ("US Arbeitslosenquote", "%"),
        "DGS10": ("US Treasury 10Y", "% p.a."),
        "DGS2": ("US Treasury 2Y", "% p.a."),
        "T10YIE": ("US 10Y Breakeven Inflation", "% p.a."),
        "DTWEXBGS": ("US Dollar Index (breit, nominal)", "Index"),
        "ECBDFR": ("EZB Einlagefazilität (Eurozone)", "% p.a."),
        "CP0000EZCCM086NEST": ("Eurozone HICP (harmonisierter Verbraucherpreisindex)", "Index 2025=100"),
    }
    if not fred_key:
        st.info("Trage links deinen FRED API-Key ein. Danach kannst du die Makroindikatoren laden.")
    selected = st.multiselect("Indikatoren", list(fred_series.keys()), default=["FEDFUNDS", "CPIAUCSL", "UNRATE", "DGS10", "DGS2", "ECBDFR", "CP0000EZCCM086NEST"], format_func=lambda sid: f"{sid} — {fred_series[sid][0]}")
    if st.button("Makroindikatoren laden", type="primary"):
        if not fred_key:
            st.error("Bitte zuerst links einen FRED API-Key eintragen.")
        elif not selected:
            st.warning("Bitte mindestens einen Indikator auswählen.")
        else:
            summary, errors = [], []
            loaded_series = {}
            with st.spinner("Lade FRED-Daten…"):
                for sid in selected:
                    try:
                        series_df = get_fred_series(sid, fred_key)
                        loaded_series[sid] = series_df.copy()
                        if series_df.empty:
                            errors.append(f"{sid}: keine Beobachtungen zurückgegeben.")
                            continue
                        last = series_df.iloc[-1]
                        prev = series_df.iloc[-2] if len(series_df) > 1 else None
                        summary.append({"Serie": sid, "Indikator": fred_series[sid][0], "Letztes Datum": last["date"].date().isoformat(), "Letzter Wert": float(last["value"]), "Einheit": fred_series[sid][1], "Vorwert": float(prev["value"]) if prev is not None else None, "Beobachtungen": len(series_df)})
                    except Exception as exc:
                        errors.append(f"{sid}: {exc}")
            if summary:
                result = pd.DataFrame(summary)
                st.success(f"{len(summary)} von {len(selected)} Serien erfolgreich geladen.")
                st.dataframe(result, use_container_width=True, hide_index=True)
                chosen = st.selectbox("Serie im Verlauf anzeigen", [r["Serie"] for r in summary], format_func=lambda sid: f"{sid} — {fred_series[sid][0]}")
                try:
                    history = get_fred_series(chosen, fred_key)
                    st.line_chart(history.set_index("date")["value"])
                    st.caption(f"Quelle: FRED · Serie {chosen} · letzter verfügbarer Datenpunkt {history.iloc[-1]['date'].date().isoformat()}. Frequenz und Aktualisierung hängen von der Serie ab; keine garantierten Echtzeitdaten.")
                    st.dataframe(history.tail(12), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Verlauf konnte nicht angezeigt werden: {exc}")
            if errors:
                st.warning("Einige Serien konnten nicht geladen werden: " + " | ".join(errors))

            st.info("Die geladenen US-Makroserien dienen derzeit nur der Dateninspektion. Sie erzeugen keinen Forex-Bias, da ein validierter Vergleich mit dem Gegenwährungsraum und Konsistenzchecks fehlen.")
    st.markdown("#### Status der G10-Abdeckung")
    coverage = pd.DataFrame([
        ["USD", "Teilweise verbunden", "US-Makroindikatoren oben über FRED ladbar"],
        ["EUR", "Teilweise verbunden", "EZB-Einlagezins und Eurozonen-HICP via FRED; weitere Reihen fehlen"],
        ["GBP", "Nicht verbunden", "BoE-/ONS-Serien separat integrieren"],
        ["JPY", "Nicht verbunden", "BoJ-/Japan-Statistikserien separat integrieren"],
        ["CHF", "Nicht verbunden", "SNB-/Schweizer Statistikserien separat integrieren"],
        ["CAD", "Nicht verbunden", "BoC-/Statistics Canada-Serien separat integrieren"],
        ["AUD", "Nicht verbunden", "RBA-/ABS-Serien separat integrieren"],
        ["NZD", "Nicht verbunden", "RBNZ-/Stats NZ-Serien separat integrieren"],
    ], columns=["CCY", "Status", "Hinweis"])
    st.dataframe(coverage, use_container_width=True, hide_index=True)

elif page == "ML Lab":
    st.subheader("ML Lab — FX-Richtungsprognose")
    st.caption("Walk-forward-Auswertung je einzelner Währung gegenüber USD bzw. beim USD gegenüber dem breiten Dollarindex. Keine Kursgarantie; Ergebnisse sind nur so belastbar wie Daten und Testdesign.")
    currency_proxies = {
        "EUR — Euro": {"symbol": "EURUSD=X", "invert": False, "reference": "USD"},
        "USD — US-Dollar": {"symbol": "DX-Y.NYB", "invert": False, "reference": "breiter USD-Index (DXY)"},
        "GBP — Britisches Pfund": {"symbol": "GBPUSD=X", "invert": False, "reference": "USD"},
        "JPY — Japanischer Yen": {"symbol": "JPY=X", "invert": True, "reference": "USD (inverse USD/JPY-Reihe)"},
        "CHF — Schweizer Franken": {"symbol": "CHF=X", "invert": True, "reference": "USD (inverse USD/CHF-Reihe)"},
        "CAD — Kanadischer Dollar": {"symbol": "CAD=X", "invert": True, "reference": "USD (inverse USD/CAD-Reihe)"},
        "AUD — Australischer Dollar": {"symbol": "AUDUSD=X", "invert": False, "reference": "USD"},
        "NZD — Neuseeland-Dollar": {"symbol": "NZDUSD=X", "invert": False, "reference": "USD"},
    }
    currency = st.selectbox("Einzelne Währung", list(currency_proxies.keys()), key="ml_currency")
    proxy = currency_proxies[currency]
    horizon = st.selectbox("Prognosehorizont", [10, 30, 60, 90], index=3, format_func=lambda x: f"{x} Handelstage", key="ml_horizon")
    years = st.selectbox("Trainingshistorie", ["5y", "10y", "max"], index=1, key="ml_years")
    st.caption(f"Referenz: {proxy['reference']}. Bei JPY/CHF/CAD wird die inverse USD-Notierung verwendet, damit steigende Werte eine stärkere ausgewählte Währung bedeuten.")

    # Fundamental panel for the selected currency alone (not a currency-pair comparison).
    st.markdown("#### Fundamentaler Währungs-Scanner — jeder Punkt ein Faktor")
    st.caption("Bewertet ausschließlich die oben ausgewählte Einzelwährung. Zins- und Inflationszeitreihen werden mit konservativen Veröffentlichungslags zeitlich an die Kursdaten angelegt und – sofern ausreichend Historie verfügbar ist – direkt als Features im ML-Modell trainiert.")
    fund_ccy = currency.split(" — ")[0]
    cfg = MACRO_CURRENCY_SERIES.get(fund_ccy)
    if not fred_key:
        st.info("Für fundamentale Einzelwährungsdaten bitte links einen FRED API-Key eintragen. Ohne Key werden keine Makro-Bewertungen erfunden.")
    elif not cfg:
        st.warning(f"Keine Makro-Zuordnung für {fund_ccy}.")
    else:
        rows = []
        errors = []
        for label, key, bullish_high in [
            ("Leitzins / Kurzfristzins", "rate", True),
            ("Inflation (CPI/HICP)", "inflation", False),
        ]:
            sid = cfg.get(key)
            if not sid:
                errors.append(f"{label}: keine Serien-ID")
                continue
            try:
                series = get_fred_series(sid, fred_key, observation_start="2015-01-01")
                if series.empty:
                    raise ValueError("keine Beobachtungen")
                latest = series.iloc[-1]
                age = (pd.Timestamp.now().normalize() - pd.Timestamp(latest["date"]).normalize()).days
                max_age = int(cfg.get("rate_age" if key == "rate" else "inflation_age", 120))
                if age > max_age:
                    raise ValueError(f"letzter Wert {latest['date'].date()}, {age} Tage alt")
                vals = pd.to_numeric(series["value"], errors="coerce").dropna()
                if len(vals) < 12:
                    raise ValueError("zu wenige historische Werte")
                median = float(vals.median())
                current = float(latest["value"])
                # Relative-to-own-history classification avoids comparing unrelated national levels.
                delta = current - median
                if abs(delta) < 0.10:
                    bias = "NEUTRAL"
                elif (delta > 0) == bullish_high:
                    bias = "BULLISH"
                else:
                    bias = "BEARISH"
                rows.append({"Faktor": label, "Aktueller Wert": current, "Historischer Median": median,
                             "Abweichung zum Median": delta, "Einordnung": bias, "Datenstand": str(pd.Timestamp(latest["date"]).date()),
                             "Alter (Tage)": age, "Serie": sid})
            except Exception as exc:
                errors.append(f"{label} ({sid}): {exc}")
        if rows:
            fdf = pd.DataFrame(rows)
            colors = {"BULLISH":"#16a34a", "BEARISH":"#dc2626", "NEUTRAL":"#94a3b8"}
            fdf["Punktfarbe"] = fdf["Einordnung"].map(colors)
            fig_f = go.Figure()
            for _, rr in fdf.iterrows():
                fig_f.add_trace(go.Scatter(x=[rr["Abweichung zum Median"]], y=[rr["Faktor"]], mode="markers",
                    marker=dict(size=18, color=rr["Punktfarbe"]), name=rr["Einordnung"],
                    customdata=[[rr["Einordnung"], rr["Aktueller Wert"], rr["Historischer Median"], rr["Datenstand"]]],
                    hovertemplate="%{y}<br>Abweichung: %{x:.3f}<br>Bias: %{customdata[0]}<br>Aktuell: %{customdata[1]:.3f}<br>Median: %{customdata[2]:.3f}<br>Datenstand: %{customdata[3]}<extra></extra>"))
            fig_f.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_f.update_layout(title=f"Fundamentale Faktor-Punkte — {fund_ccy} allein", xaxis_title="Aktueller Wert minus eigener historischer Median", yaxis_title="Fundamentalfaktor", height=300, showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)
            st.dataframe(fdf[["Faktor", "Einordnung", "Aktueller Wert", "Historischer Median", "Abweichung zum Median", "Datenstand", "Alter (Tage)", "Serie"]], use_container_width=True, hide_index=True)
            bulls = int((fdf["Einordnung"] == "BULLISH").sum())
            bears = int((fdf["Einordnung"] == "BEARISH").sum())
            bias = "BULLISH" if bulls > bears else ("BEARISH" if bears > bulls else "NEUTRAL")
            a, b, c = st.columns(3)
            a.metric(f"Makro-Bias {fund_ccy}", bias)
            b.metric("Bullishe Faktoren", bulls)
            c.metric("Bearishe Faktoren", bears)
        if errors:
            st.warning("Nicht verfügbare/veraltete Einzelwährungsdaten: " + " | ".join(errors))
        st.caption("Interpretation: Ein im eigenen historischen Vergleich hoher Zins wird hier als carry-unterstützend gewertet; überdurchschnittliche Inflation als möglicher Gegenwind. Die Scanner-Punkte sind eine Heuristik. Für das ML-Modell werden stattdessen historische Rohzeitreihen plus Veränderungen mit konservativem Zeitversatz verwendet; die Prognosegüte muss im Holdout geprüft werden.")

    if st.button("Modell trainieren und Backtest ausführen", type="primary", key="run_ml_lab"):
        try:
            with st.spinner("Historische Kurse und Makrozeitreihen laden, Features zeitlich abgleichen und testen…"):
                raw = load_market_history(proxy["symbol"], years)
                if not raw.empty and proxy["invert"]:
                    raw = raw.copy()
                    raw["close"] = 1.0 / pd.to_numeric(raw["close"], errors="coerce")
                if raw.empty or len(raw) < 500:
                    st.error("Zu wenige Kursdaten für einen sinnvollen ML-Test (mindestens 500 Tageswerte erforderlich).")
                else:
                    d = raw.copy()
                    d["date"] = pd.to_datetime(d["date"])
                    d = d.sort_values("date").set_index("date")
                    close = d["close"].astype(float)
                    features = pd.DataFrame(index=d.index)
                    for lag in [1, 2, 5, 10, 20]:
                        features[f"ret_{lag}d"] = close.pct_change(lag)
                    for window in [5, 10, 20, 50, 100, 200]:
                        ma = close.rolling(window).mean()
                        features[f"ma_gap_{window}"] = close / ma - 1
                    features["vol_20d"] = close.pct_change().rolling(20).std()
                    features["mom_60d"] = close.pct_change(60)
                    # Calendar/seasonality features are known at forecast time (no future leakage).
                    month = pd.Series(d.index.month, index=d.index, dtype=float)
                    features["month_sin"] = np.sin(2 * np.pi * month / 12.0)
                    features["month_cos"] = np.cos(2 * np.pi * month / 12.0)
                    # Relative short/long volatility and trend persistence.
                    features["vol_ratio_5_20"] = close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std()
                    features["ret_20_minus_60"] = close.pct_change(20) - close.pct_change(60)

                    # Integrate historical macro data for the selected currency itself.
                    # Observation dates are shifted forward conservatively to reduce release-date look-ahead:
                    # policy-rate proxies +7 calendar days; CPI/HICP index proxies +45 days.
                    # These are safety approximations, not exact release calendars / ALFRED vintages.
                    macro_feature_names = []
                    macro_status = "Makro-Features nicht verfügbar; technisches Modell wird verwendet."
                    macro_cfg = MACRO_CURRENCY_SERIES.get(fund_ccy)
                    if fred_key and macro_cfg:
                        try:
                            rate_hist = get_fred_series(macro_cfg["rate"], fred_key, observation_start="2000-01-01")
                            infl_hist = get_fred_series(macro_cfg["inflation"], fred_key, observation_start="2000-01-01")
                            if len(rate_hist) >= 6 and len(infl_hist) >= 15:
                                rate_hist = rate_hist.sort_values("date").drop_duplicates("date").copy()
                                rate_hist["available_date"] = pd.to_datetime(rate_hist["date"]) + pd.Timedelta(days=7)
                                rate_hist["macro_rate_level"] = pd.to_numeric(rate_hist["value"], errors="coerce")
                                rate_hist["macro_rate_change_3m"] = rate_hist["macro_rate_level"] - rate_hist["macro_rate_level"].shift(3)
                                rate_merge = rate_hist[["available_date", "macro_rate_level", "macro_rate_change_3m"]].dropna(subset=["macro_rate_level"])

                                infl_hist = infl_hist.sort_values("date").drop_duplicates("date").copy()
                                infl_hist["available_date"] = pd.to_datetime(infl_hist["date"]) + pd.Timedelta(days=45)
                                infl_hist["infl_index"] = pd.to_numeric(infl_hist["value"], errors="coerce")
                                # CPI/HICP series are normally monthly; require a full 12-observation comparison.
                                infl_hist["macro_inflation_yoy"] = (infl_hist["infl_index"] / infl_hist["infl_index"].shift(12) - 1.0) * 100.0
                                infl_hist["macro_inflation_change_3m"] = infl_hist["macro_inflation_yoy"] - infl_hist["macro_inflation_yoy"].shift(3)
                                infl_merge = infl_hist[["available_date", "macro_inflation_yoy", "macro_inflation_change_3m"]].dropna(subset=["macro_inflation_yoy"])

                                market_dates = pd.DataFrame({"market_date": pd.to_datetime(features.index)}).sort_values("market_date")
                                aligned = pd.merge_asof(market_dates, rate_merge.sort_values("available_date"),
                                                        left_on="market_date", right_on="available_date", direction="backward")
                                aligned = pd.merge_asof(aligned.sort_values("market_date"), infl_merge.sort_values("available_date"),
                                                        left_on="market_date", right_on="available_date", direction="backward", suffixes=("", "_infl"))
                                aligned.index = features.index
                                candidate_cols = ["macro_rate_level", "macro_rate_change_3m", "macro_inflation_yoy", "macro_inflation_change_3m"]
                                coverage = aligned[candidate_cols].notna().all(axis=1).mean()
                                if coverage >= 0.60:
                                    for col in candidate_cols:
                                        features[col] = aligned[col].to_numpy()
                                    macro_feature_names = candidate_cols
                                    macro_status = f"Makro integriert: {fund_ccy} · 4 Features · {coverage:.0%} gemeinsame Abdeckung. Veröffentlichungslags: Zins +7 Tage, Inflation +45 Tage."
                                else:
                                    macro_status = f"Makro nicht ins Training übernommen: nur {coverage:.0%} gemeinsame Abdeckung (benötigt mindestens 60 %). Technisches Modell als Fallback."
                            else:
                                macro_status = "Makro-Historie zu kurz; technisches Modell als Fallback."
                        except Exception as macro_exc:
                            macro_status = f"Makroabruf fehlgeschlagen ({macro_exc}); technisches Modell als Fallback."
                    elif not fred_key:
                        macro_status = "Kein FRED-Key hinterlegt; technisches Modell als Fallback."
                    st.caption(macro_status)

                    target_return = close.shift(-int(horizon)) / close - 1
                    dataset = features.copy()
                    dataset["target"] = (target_return > 0).astype(int)
                    dataset["future_return"] = target_return
                    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
                    # Drop tail where the future outcome is not yet observable.
                    dataset = dataset.iloc[:-int(horizon)]
                    if len(dataset) < 300:
                        st.error("Nach Feature-Bau bleiben zu wenige Beobachtungen. Bitte längere Historie oder kürzeren Horizont wählen.")
                    else:
                        split = int(len(dataset) * 0.8)
                        X, y = dataset.drop(columns=["target", "future_return"]), dataset["target"]
                        X_train, X_test = X.iloc[:split], X.iloc[split:]
                        y_train, y_test = y.iloc[:split], y.iloc[split:]
                        try:
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
                            from sklearn.calibration import calibration_curve
                            model = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=10,
                                                           class_weight="balanced_subsample", random_state=42, n_jobs=-1)
                            model.fit(X_train, y_train)
                            prob_up = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
                            pred = (prob_up >= 0.5).astype(int)
                            actual = y_test.to_numpy()
                            acc = accuracy_score(actual, pred)
                            bacc = balanced_accuracy_score(actual, pred)
                            baseline = max(float((actual == 1).mean()), float((actual == 0).mean()))
                            brier = brier_score_loss(actual, prob_up)
                            logloss = log_loss(actual, np.clip(prob_up, 1e-6, 1 - 1e-6), labels=[0, 1])
                            # Calibration is measured only on the chronological holdout; bins can be noisy for small samples.
                            frac_positive, mean_predicted = calibration_curve(actual, prob_up, n_bins=8, strategy="quantile")
                            current_prob = float(model.predict_proba(X.iloc[[-1]])[0, list(model.classes_).index(1)])
                            # Transparent directional label for the selected single currency.
                            # This ML estimate is not the validated FX Bias Engine signal.
                            if current_prob >= 0.55:
                                ml_bias = "BULLISH"
                                ml_explanation = (
                                    f"Das Modell schätzt die Wahrscheinlichkeit einer positiven "
                                    f"{horizon}-Tage-Rendite auf {current_prob:.1%}. Das liegt mindestens "
                                    "5 Prozentpunkte über 50 % und spricht im Modell für Aufwärtsdruck."
                                )
                            elif current_prob <= 0.45:
                                ml_bias = "BEARISH"
                                ml_explanation = (
                                    f"Das Modell schätzt die Wahrscheinlichkeit einer positiven "
                                    f"{horizon}-Tage-Rendite auf {current_prob:.1%}. Das liegt mindestens "
                                    "5 Prozentpunkte unter 50 % und spricht im Modell für Abwärtsdruck."
                                )
                            else:
                                ml_bias = "NEUTRAL"
                                ml_explanation = (
                                    f"Mit {current_prob:.1%} liegt die Modellwahrscheinlichkeit nahe an 50 %. "
                                    "Die Richtung ist damit nicht deutlich genug, um bullish oder bearish zu nennen."
                                )
                            st.markdown("#### Aktuelle ML-Einschätzung")
                            b1, b2, b3 = st.columns(3)
                            b1.metric("Richtung", ml_bias)
                            b2.metric("P(Aufwärts)", f"{current_prob:.1%}")
                            b3.metric("Prognosehorizont", f"{horizon} Handelstage")
                            if ml_bias == "BULLISH":
                                st.success(ml_explanation)
                            elif ml_bias == "BEARISH":
                                st.error(ml_explanation)
                            else:
                                st.info(ml_explanation)
                            st.caption(
                                "Die ML-Richtung beschreibt die einzelne ausgewählte Währung gegenüber ihrer "
                                "Referenz. Sie ist kein freigegebenes LONG-/SHORT-Signal des FX Bias Engine."
                            )

                            k1,k2,k3,k4 = st.columns(4)
                            k1.metric("Modell: P(Aufwärts)", f"{current_prob:.1%}")
                            k2.metric("Holdout Accuracy", f"{acc:.1%}")
                            k3.metric("Balanced Accuracy", f"{bacc:.1%}")
                            k4.metric("Mehrheits-Baseline", f"{baseline:.1%}")
                            c1, c2 = st.columns(2)
                            c1.metric("Brier Score ↓", f"{brier:.3f}", help="Niedriger ist besser; 0 wäre perfekt. Bewertet die Wahrscheinlichkeiten, nicht nur die Richtung.")
                            c2.metric("Log Loss ↓", f"{logloss:.3f}", help="Niedriger ist besser; falsche Prognosen mit hoher Sicherheit werden stärker bestraft.")
                            cal_fig = go.Figure()
                            cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfekte Kalibrierung"))
                            cal_fig.add_trace(go.Scatter(x=mean_predicted, y=frac_positive, mode="lines+markers", name="Holdout-Kalibrierung"))
                            cal_fig.update_layout(title="Edge & Calibration — Zuverlässigkeit der Wahrscheinlichkeiten", xaxis_title="Mittlere prognostizierte P(Aufwärts)", yaxis_title="Tatsächlicher Anteil Aufwärts", xaxis=dict(range=[0,1], tickformat=".0%"), yaxis=dict(range=[0,1], tickformat=".0%"), height=360)
                            st.plotly_chart(cal_fig, use_container_width=True)
                            st.caption("Interpretation: Punkte nahe der gestrichelten Diagonale sind besser kalibriert. Die Grafik nutzt ausschließlich den chronologischen Holdout; bei wenigen Fällen je Bin ist sie instabil.")
                            st.caption(f"Testfenster: {X_test.index.min().date()} bis {X_test.index.max().date()} · {len(X_test)} Testfälle · Ziel: Rendite nach {horizon} Handelstagen > 0.")
                            if bacc <= 0.52 or bacc <= (baseline - 0.03):
                                st.warning("Der Test zeigt keinen überzeugenden Richtungs-Vorteil gegenüber einer einfachen Baseline. Die aktuelle Wahrscheinlichkeit nicht als eigenständiges Handelssignal verwenden.")
                            else:
                                st.info("Der Test ist historisch besser als die genutzte Baseline, garantiert aber keine zukünftige Prognosegüte.")
                            importance = pd.DataFrame({"Feature": X.columns, "Wichtigkeit": model.feature_importances_}).sort_values("Wichtigkeit", ascending=False).head(10)
                            st.markdown("#### ML-Faktoren-Scanner — jeder Punkt ein Modellfaktor")
                            st.caption("Jeder Punkt steht für einen tatsächlich verwendeten Kurs-/Trend-/Volatilitäts- oder Saisonfaktor. Die Farbe zeigt, ob das Ersetzen dieses Faktors durch seinen historischen Trainingsmedian die aktuelle Modellwahrscheinlichkeit senkt oder erhöht. Das ist eine Sensitivitätsanalyse, kein kausaler Effekt. Makro, Zinsen, COT und News sind in diesem Modell noch nicht enthalten.")
                            # One-at-a-time sensitivity: compare current probability with the
                            # probability when one feature is set to its training median.
                            current_row = X.iloc[[-1]].copy()
                            current_p = float(model.predict_proba(current_row)[0, list(model.classes_).index(1)])
                            factor_rows = []
                            train_medians = X_train.median(numeric_only=True)
                            for feature_name in X.columns:
                                altered = current_row.copy()
                                altered.loc[:, feature_name] = train_medians.get(feature_name, current_row.iloc[0][feature_name])
                                altered_p = float(model.predict_proba(altered)[0, list(model.classes_).index(1)])
                                delta_pp = (current_p - altered_p) * 100
                                factor_rows.append({
                                    "Faktor": feature_name,
                                    "Wichtigkeit": float(model.feature_importances_[list(X.columns).index(feature_name)]),
                                    "Sensitivität (Prozentpunkte)": delta_pp,
                                    "Einordnung": "BULLISH" if delta_pp >= 1.0 else ("BEARISH" if delta_pp <= -1.0 else "NEUTRAL"),
                                    "Aktueller Wert": float(current_row.iloc[0][feature_name]),
                                })
                            factor_df = pd.DataFrame(factor_rows)
                            factor_df["Einflussgröße"] = factor_df["Sensitivität (Prozentpunkte)"].abs()
                            factor_df = factor_df.sort_values("Wichtigkeit", ascending=False)
                            dot_fig = go.Figure()
                            for _, fr in factor_df.iterrows():
                                color = "#16a34a" if fr["Einordnung"] == "BULLISH" else ("#dc2626" if fr["Einordnung"] == "BEARISH" else "#94a3b8")
                                dot_fig.add_trace(go.Scatter(
                                    x=[fr["Sensitivität (Prozentpunkte)" ]], y=[fr["Faktor"]],
                                    mode="markers", name=fr["Einordnung"],
                                    marker=dict(size=10 + 24 * min(float(fr["Wichtigkeit"]), 1.0), color=color),
                                    customdata=[[fr["Wichtigkeit"], fr["Aktueller Wert"], fr["Einordnung"]]],
                                    hovertemplate="%{y}<br>Sensitivität: %{x:.2f} pp<br>Wichtigkeit: %{customdata[0]:.3f}<br>Aktueller Wert: %{customdata[1]:.4f}<br>%{customdata[2]}<extra></extra>"
                                ))
                            dot_fig.add_vline(x=0, line_dash="dash", line_color="gray")
                            dot_fig.update_layout(title="Faktor-Punkte: Einflussrichtung auf P(Aufwärts)", xaxis_title="Aktuelle P(Aufwärts) minus P(Aufwärts) nach Median-Ersatz (Prozentpunkte)", yaxis_title="Modellfaktor", height=max(400, 34 * len(factor_df)), showlegend=False)
                            st.plotly_chart(dot_fig, use_container_width=True)
                            st.dataframe(factor_df[["Faktor", "Einordnung", "Sensitivität (Prozentpunkte)", "Wichtigkeit", "Aktueller Wert"]], use_container_width=True, hide_index=True)
                            st.markdown("#### Welche Faktoren beeinflussen das Modell?")
                            st.dataframe(importance, use_container_width=True, hide_index=True)
                            st.markdown("#### Letzte historische Testprognosen")
                            hist = pd.DataFrame({"P(Aufwärts)": prob_up, "Tatsächlich aufwärts": actual,
                                                 "Modellrichtung": np.where(pred == 1, "AUFWÄRTS", "ABWÄRTS"),
                                                 "Zielrendite": dataset.loc[X_test.index, "future_return"]})
                            hist["P(Aufwärts)"] = hist["P(Aufwärts)"] * 100
                            st.dataframe(hist.tail(25), use_container_width=True)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=hist.index, y=hist["P(Aufwärts)"], mode="markers", name="P(Aufwärts)"))
                            fig.add_hline(y=50, line_dash="dash", line_color="gray")
                            fig.update_layout(title=f"{currency} — historische Out-of-sample Wahrscheinlichkeiten ({horizon}T)", yaxis_title="P(Aufwärts) %", xaxis_title="Testdatum")
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Methodik: chronologischer 80/20-Holdout, Random Forest, technische Preis-/Volatilitäts-/Momentum-Features plus Monats-Saisonalität, keine zufällige Durchmischung. Brier Score, Log Loss und Kalibrierungsdiagramm werden auf dem Holdout berechnet. Der neue Faktoren-Scanner zeigt Sensitivität der aktuell verwendeten technischen und Saison-Features. Wichtig: COT, Makro/Yields und News sind noch NICHT als historische Modellfeatures eingebunden; sie dürfen daher nicht als vom Modell berücksichtigte Faktoren interpretiert werden. Transaktionskosten, Slippage und Overlapping-Horizon-Effekte sind nicht modelliert; dies ist kein handelbarer Netto-Backtest.")
                        except ImportError:
                            st.error("scikit-learn fehlt. Ergänze `scikit-learn` in requirements.txt und deploye die App erneut.")
        except Exception as exc:
            st.error(f"ML Lab fehlgeschlagen: {exc}")

elif page == "Saisonalität":
    st.subheader("Monatliche Saisonalität")
    symbols = {name: info["symbol"] for name, info in FX_PAIRS.items()}
    label = st.selectbox("Instrument / Proxy", list(symbols))
    years = st.selectbox("Historie", ["5y", "10y", "max"], index=1)
    if st.button("Historische Daten laden und berechnen"):
        with st.spinner("Lade historische Daten…"):
            try:
                prices = load_market_history(symbols[label], years)
                if prices.empty:
                    st.error("Keine Daten erhalten. Symbol oder Anbieter prüfen.")
                else:
                    result = seasonality(prices)
                    result["Monat"] = result["month"].map(lambda m: MONTHS[int(m)-1])
                    result["Ø Rendite (%)"] = result["avg_return"] * 100
                    result["Positive Jahre (%)"] = result["win_rate"] * 100
                    st.caption(f"Quelle: yfinance/Yahoo Finance Wrapper · Abruf {utc_now()} · Historische Daten, kein garantierter Echtzeitfeed. Symbol: {symbols[label]}")
                    fig = px.bar(result, x="Monat", y="Ø Rendite (%)", color="Ø Rendite (%)", title=f"{label} — durchschnittliche Monatsrendite")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(result[["Monat", "Ø Rendite (%)", "Positive Jahre (%)", "observations", "median", "volatility"]], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Download/Berechnung fehlgeschlagen: {e}")
    else:
        st.info("Klicke auf „Historische Daten laden und berechnen“, um Saisonalität aus historischen Kursdaten zu berechnen. Die Daten sind nicht garantiert vollständig oder börsenoffiziell.")

elif page == "COT / Positioning":
    st.subheader("COT / Positioning")
    st.caption("CFTC-Daten sind Futures-Positionierung, nicht der gesamte globale Spot-FX-Markt. Contract code mapping muss je Datensatz geprüft werden.")
    cot_contract = st.selectbox("CFTC-Futures-Kontrakt", list(CFTC_CONTRACT_CODES), format_func=lambda c: {
        "6E": "Euro FX (6E)", "6B": "British Pound (6B)", "6J": "Japanese Yen (6J)",
        "6A": "Australian Dollar (6A)", "6C": "Canadian Dollar (6C)",
        "6S": "Swiss Franc (6S)", "6N": "New Zealand Dollar (6N)"
    }.get(c, c))
    if st.button("CFTC-Historie laden", type="primary"):
        try:
            cot = load_cot_history(cot_contract, limit=52)
            st.success(f"{len(cot)} wöchentliche Datensätze · letzter Bericht {cot.iloc[-1]['date'].date()} · {cot.attrs.get('age_days', 0):.1f} Tage alt.")
            st.caption("Quelle: CFTC Public Reporting API · Legacy Futures Only. Net = Long minus Short. Non-reportable ist keine direkt ausgewiesene Retail-Kategorie.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Commercial net", f"{cot.iloc[-1]['commercial_net']:,.0f}")
            c2.metric("Non-commercial net", f"{cot.iloc[-1]['noncommercial_net']:,.0f}")
            c3.metric("Non-reportable net", "—" if pd.isna(cot.iloc[-1]['nonreportable_net']) else f"{cot.iloc[-1]['nonreportable_net']:,.0f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cot["date"], y=cot["commercial_net"], mode="lines", name="Commercial net"))
            fig.add_trace(go.Scatter(x=cot["date"], y=cot["noncommercial_net"], mode="lines", name="Non-commercial net"))
            fig.add_trace(go.Scatter(x=cot["date"], y=cot["nonreportable_net"], mode="lines", name="Non-reportable net"))
            fig.update_layout(title=f"{cot.attrs.get('market_name', cot_contract)} — Netto-Positionen", yaxis_title="Kontrakte", legend_title="Gruppe")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cot[["date", "commercial_net", "noncommercial_net", "nonreportable_net", "open_interest"]].tail(20).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"CFTC-Daten nicht verfügbar/validiert: {e}")
    st.markdown("#### Kontrakte für FX-Recherche")
    st.dataframe(pd.DataFrame([
        ["EUR", "Euro FX", "6E"], ["GBP", "British Pound", "6B"], ["JPY", "Japanese Yen", "6J"],
        ["CAD", "Canadian Dollar", "6C"], ["AUD", "Australian Dollar", "6A"],
        ["CHF", "Swiss Franc", "6S"], ["NZD", "New Zealand Dollar", "6N"], ["USD", "Kein einzelner direkter G10-COT-Proxy", "—"]
    ], columns=["Währung", "CFTC-Futures-Kontrakt", "Ticker/Code"]), use_container_width=True, hide_index=True)
    st.warning("Der CFTC-Endpunkt und seine Felder können sich ändern. Vor produktiver Nutzung Reporttyp (Legacy/Disaggregated/TFF), Datenfelder, Berichtswoche und Kontrakt-ID validieren.")

elif page == "NQ & ES":
    st.subheader("NQ & ES Macro Lens")
    symbols = {"NQ=F": "Nasdaq-100 Futures (Datenanbieter-Symbol)", "ES=F": "S&P 500 Futures (Datenanbieter-Symbol)"}
    symbol = st.selectbox("Historisches Symbol", list(symbols), format_func=lambda x: symbols[x])
    period = st.selectbox("Zeitraum", ["3mo", "1y", "5y"], index=1)
    if st.button("Historie laden"):
        try:
            df = load_market_history(symbol, period)
            if df.empty:
                st.error("Keine historischen Preise erhalten.")
            else:
                st.caption(f"Historischer Preis-Proxy · Abruf {utc_now()} · keine Zusicherung von Echtzeit oder Börsenlizenz.")
                st.line_chart(df.set_index("date")["close"])
                df["return_pct"] = df["close"].pct_change()*100
                st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Marktdaten konnten nicht geladen werden: {e}")
    st.markdown("#### Makro-Checkliste")
    st.write("- US-Realrenditen / Treasury yields")
    st.write("- Fed-Erwartungen und Inflationsüberraschungen")
    st.write("- Marktbreite, Volatilität und Kreditspreads")
    st.write("- Earnings breadth / Revisionen")
    st.info("NQ-/ES-Futures-Symbole über kostenlose Wrapper können verzögert, unvollständig oder anders gerollt sein. Für Trading-Anwendungen Broker- oder lizenzierte Börsendaten nutzen.")

elif page == "Quant & Regression":
    st.subheader("Quant & Regression")
    st.caption("Demo-Regression wird aus synthetischen Daten gerechnet und ist keine Marktaussage.")
    rng = np.random.default_rng(42)
    n = st.slider("Beobachtungen", 30, 500, 120, 10)
    x = rng.normal(0, 1, n)
    y = -0.45*x + rng.normal(0, 0.8, n)
    fit = np.polyfit(x, y, 1)
    pred = np.polyval(fit, x)
    ss_res = np.sum((y-pred)**2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta (synthetisch)", f"{fit[0]:.3f}")
    c2.metric("R² (synthetisch)", f"{r2:.3f}")
    c3.metric("Observations", n)
    plot_df = pd.DataFrame({"x": x, "y": y, "OLS": pred}).sort_values("x")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Synthetische Punkte", opacity=.6))
    fig.add_trace(go.Scatter(x=plot_df["x"], y=plot_df["OLS"], mode="lines", name="OLS-Linie"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Risk regime inputs")
    st.write("Noch nicht bewertet: Aktienbreite, Realrenditen, Kreditspreads, Dollartrend, Volatilität. Keine Risk-on/off-Klassifikation ohne verifizierte Inputs.")

else:
    st.subheader("Datenquellen & Aktualität")
    st.markdown("#### Quellenregister")
    sources = pd.DataFrame([
        ["Makro/Zinsen", "FRED API", "API-Key nötig; Frequenz je Serie; kein Konsenssurprise ohne Zusatzanbieter"],
        ["Zentralbankpfade", "Fed, EZB, BoE, BoJ, SNB, BoC, RBA, RBNZ", "Offizielle Veröffentlichungen; Parsing/Mapping erforderlich"],
        ["COT", "CFTC Public Reporting · Legacy Futures Only", "Wöchentlich; Kontraktcode/Schema und Aktualität werden zur Laufzeit geprüft"],
        ["Historische Preise", "yfinance Wrapper / Yahoo Finance", "Nicht garantiert offiziell, vollständig, lizenziert oder echtzeitfähig"],
        ["FX-Kurse", "FX-Datenanbieter / Broker", "Aktualität, Historie und Lizenzbedingungen validieren"],
        ["Konsensdaten", "Separater Makro-Kalender-Anbieter", "Meist separat/lizenziert"]
    ], columns=["Bereich", "Quelle", "Hinweis"])
    st.dataframe(sources, use_container_width=True, hide_index=True)
    st.markdown("#### Freshness-Regeln für Produktion")
    st.write("- Jede Beobachtung braucht Quelle, Zeitstempel, Einheit und Frequenz.")
    st.write("- Fehlende/veraltete Daten dürfen nicht stillschweigend durch Nullen ersetzt werden.")
    st.write("- Jedes Signal muss auf die tatsächlich verwendeten Datenpunkte zurückführbar sein.")
    st.write("- API-Schlüssel in Streamlit Secrets oder Server-Umgebung speichern, niemals in öffentliche Frontend-Dateien.")
    st.markdown("#### FRED API-Key konfigurieren")
    st.code('[secrets.toml]\nFRED_API_KEY = "dein_api_key"', language="toml")
    st.caption("Die Beispielabfrage akzeptiert den Schlüssel im Sidebar-Feld. Für dauerhaftes Hosting besser st.secrets verwenden und das Secret nicht ins GitHub-Repository hochladen.")

st.divider()
st.caption("Macro / Regime Terminal — FX + COT Signal Engine. NEUTRAL bei unzureichenden oder nicht validierten Daten. COT ist wöchentlich, kein Spot-FX-Orderflow. Keine Anlageberatung.")
