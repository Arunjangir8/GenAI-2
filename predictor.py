import pickle
import pandas as pd
from pathlib import Path
from config import CITY_META, PRICE_SANITY, VALID_PROPERTY_TYPES, VALID_STATUSES, VALID_CITIES

MODEL_PATH  = "./Models/linear_regression_model.pkl"
SCALER_PATH = "./Models/minmax_scaler.pkl"

with open(MODEL_PATH,  "rb") as f: lr_model = pickle.load(f)
with open(SCALER_PATH, "rb") as f: scaler   = pickle.load(f)


def _build_ordinal_map(values) -> dict:
    vals = sorted({str(v).strip() for v in values if str(v).strip()})
    return {v: i for i, v in enumerate(vals)}


def _load_feature_maps() -> tuple[dict, dict, dict, dict]:
    """Build deterministic category->int maps from Raw CSVs (LabelEncoder-like)."""
    raw_dir = Path(__file__).resolve().parent / "Raw"
    csv_files = sorted(raw_dir.glob("*.csv"))

    city_vals = []
    location_vals = []
    status_vals = []

    for p in csv_files:
        try:
            df = pd.read_csv(p, usecols=["city", "location", "Status"])
        except Exception:
            continue

        city_vals.extend(df["city"].dropna().astype(str).str.strip().tolist())
        location_vals.extend(df["location"].dropna().astype(str).str.strip().tolist())
        status_vals.extend(df["Status"].dropna().astype(str).str.strip().tolist())

    city_map = _build_ordinal_map(city_vals or VALID_CITIES)
    location_map = _build_ordinal_map(location_vals)
    status_map = _build_ordinal_map(status_vals or VALID_STATUSES)
    property_map = _build_ordinal_map(VALID_PROPERTY_TYPES)

    return city_map, location_map, status_map, property_map


CITY_MAP, LOCATION_MAP, STATUS_MAP, PROPERTY_MAP = _load_feature_maps()


def _encode(value: str, mapping: dict, fallback: int = 0) -> int:
    key = str(value or "").strip()
    if key in mapping:
        return mapping[key]

    # case-insensitive fallback
    kfold = key.casefold()
    for k, v in mapping.items():
        if k.casefold() == kfold:
            return v
    return fallback


def predict_rent(props: dict) -> dict:
    city = props.get("city", "Delhi")
    location = props.get("location", "")
    status = props.get("status", "Semi-Furnished")
    property_type = props.get("property_type", "Apartment")
    rooms = float(props.get("rooms", 2))

    city_code = _encode(city, CITY_MAP, fallback=0)
    location_code = _encode(location, LOCATION_MAP, fallback=0)
    status_code = _encode(status, STATUS_MAP, fallback=0)
    property_code = _encode(property_type, PROPERTY_MAP, fallback=0)

    X = pd.DataFrame([{
        "location":          float(location_code),
        "city":              float(city_code),
        "latitude":          float(props.get("latitude",  28.6)),
        "longitude":         float(props.get("longitude", 77.2)),
        "numBathrooms":      float(props.get("bathrooms", 2)),
        "numBalconies":      float(props.get("balconies", 1)),
        "isNegotiable":      float(props.get("is_negotiable", 0)),
        "SecurityDeposit":   float(props.get("security_deposit", 25000)),
        "Status":            float(status_code),
        "Size_ft²":          float(props.get("size_sqft", 1000)),
        "BHK":               rooms,
        "rooms_num":         rooms,
        "property_type":     float(property_code),
        "verification_days": 5.0,
    }])

    if hasattr(scaler, "feature_names_in_"):
        X = X.reindex(columns=list(scaler.feature_names_in_), fill_value=0.0)

    pred = round(float(lr_model.predict(scaler.transform(X))[0]), -2)
    sanity = PRICE_SANITY.get(city, {"min": 4_000, "max": 5_00_000})
    pred = max(sanity["min"], min(sanity["max"], pred))

    print(f"🏠 LR Predicted Rent: ₹{pred:,.0f}/month")
    annual     = pred * 12
    avg_psf    = 8000 if city == "Pune" else (15000 if city == "Delhi" else 20000)
    prop_val   = float(props.get("size_sqft", 1000)) * avg_psf
    city_meta  = CITY_META.get(city, {})

    return {
        "Linear Regression": pred,
        "Ensemble":          pred,
        "model_source":      "linear_regression_model",
        "analytics": {
            "monthly_rent":        pred,
            "annual_rent":         annual,
            "est_property_value":  prop_val,
            "gross_yield_pct":     round((annual / prop_val) * 100, 2),
            "price_to_rent_ratio": round(prop_val / annual, 1),
            "city_avg_yield":      city_meta.get("avg_yield", 3.5),
            "yoy_growth":          city_meta.get("yoy_growth", 8.0),
            "vacancy_rate":        city_meta.get("vacancy_rate", 10.0),
        }
    }


def format_inr(value: float) -> str:
    if value >= 1e7: return f"₹{value/1e7:.2f} Cr"
    if value >= 1e5: return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.0f}"