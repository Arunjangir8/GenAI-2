"""
config.py - Central configuration for the Real Estate Advisory Agent
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def _clean_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if value is None:
        return default
    return str(value).strip().strip('"').strip("'")

GROQ_API_KEY   = _clean_env("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL     = "llama-3.3-70b-versatile"
GROQ_TEMP      = 0.2
GROQ_MAX_TOKENS = 2048

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH  = "faiss_index"
TOP_K_RETRIEVAL   = 5

CITY_META = {
    "Delhi": {
        "avg_yield":    3.5,
        "yoy_growth":   8.2,
        "vacancy_rate": 12.5,
        "tier":         1,
        "rera_state":   "RERA Delhi"
    },
    "Mumbai": {
        "avg_yield":    3.0,
        "yoy_growth":   10.1,
        "vacancy_rate": 9.8,
        "tier":         1,
        "rera_state":   "MahaRERA"
    },
    "Pune": {
        "avg_yield":    4.2,
        "yoy_growth":   12.4,
        "vacancy_rate": 8.3,
        "tier":         2,
        "rera_state":   "MahaRERA"
    }
}

PRICE_SANITY = {
    "Delhi":  {"min": 5_000,  "max": 3_00_000},
    "Mumbai": {"min": 8_000,  "max": 5_00_000},
    "Pune":   {"min": 4_000,  "max": 2_00_000},
}

VALID_PROPERTY_TYPES = [
    "Apartment", "Independent Floor", "Independent House",
    "Studio Apartment", "Builder Floor", "Villa",
    "Penthouse", "Service Apartment"
]

VALID_STATUSES = ["Furnished", "Semi-Furnished", "Unfurnished"]
VALID_CITIES   = ["Delhi", "Mumbai", "Pune"]