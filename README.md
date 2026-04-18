# 🏙️ RealAdvisor AI — Agentic Real Estate Advisory System
### Milestone 2: Agentic AI Real Estate Advisory Assistant

---

## 📋 Overview

An autonomous AI-powered real estate advisory system built with **LangGraph**, **Groq AI (LLaMA 3.3 70B)**, **FAISS RAG**, and the **Milestone 1 ML models**. The system reasons through property valuation, retrieves live market insights, analyzes comparable properties, assesses investment risks, and generates structured advisory reports.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     SYSTEM COMPONENTS                              │
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │  Streamlit   │    │  LangGraph   │    │   Groq AI            │ │
│  │  Web UI      │───►│  StateGraph  │───►│   LLaMA 3.3 70B      │ │
│  │  (app.py)    │    │  (7 nodes)   │    │   (Risk, Advice,     │ │
│  └──────────────┘    └──────┬───────┘    │    Report)           │ │
│                             │            └──────────────────────┘ │
│          ┌──────────────────┼─────────────────────┐               │
│          ▼                  ▼                     ▼               │
│  ┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐   │
│  │  ML Models   │  │  FAISS RAG   │   │  Knowledge Base      │   │
│  │  (Milestone1)│  │  + Sentence  │   │  (Delhi/Mumbai/Pune  │   │
│  │  Lin.Reg +   │  │  Transformers│   │   Market Docs)       │   │
│  │  Rand.Forest │  └──────────────┘   └──────────────────────┘   │
│  └──────────────┘                                                  │
└────────────────────────────────────────────────────────────────────┘
```

## 🔄 LangGraph Agent Workflow

```
[User Input]
     │
     ▼
┌─────────────────┐
│ 1. validate_input│  → Sanitize, enrich, set defaults
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. predict_price │  → Run LR + RF models → Ensemble rent estimate
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ 3. retrieve_market_  │  → FAISS similarity search → city market docs
│    data (RAG)        │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 4. analyze_compar-   │  → Score & rank comparable properties
│    ables             │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 5. assess_risk       │  → Groq LLM: risk factor analysis
│    (LLM)             │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 6. generate_advice   │  → Groq LLM: BUY/HOLD/AVOID recommendation
│    (LLM)             │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 7. compile_report    │  → Groq LLM: Full structured advisory report
│    (LLM)             │
└────────┬─────────────┘
         │
         ▼
    [Final Output]
    • Predicted Rent (₹)
    • Gross Yield %
    • Comparable Analysis
    • Risk Assessment
    • Investment Recommendation (BUY/HOLD/AVOID)
    • Full Advisory Report (downloadable .md)
```

---

## 🚀 Setup & Installation

### 1. Clone and navigate
```bash
cd real_estate_agent
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at: https://console.groq.com

### 5. (Optional) Save Milestone 1 models
Save your trained models to the `models/` directory:
```python
# In your Milestone 1 notebook, add this at the end:
import pickle, os
os.makedirs('models', exist_ok=True)
pickle.dump(rf_model,   open('models/rf_model.pkl', 'wb'))
pickle.dump(lr_model,   open('models/lr_model.pkl', 'wb'))
pickle.dump(scaler,     open('models/scaler.pkl', 'wb'))
pickle.dump(le_city,    open('models/le_city.pkl', 'wb'))
pickle.dump(le_property,open('models/le_property.pkl', 'wb'))
pickle.dump(le_status,  open('models/le_status.pkl', 'wb'))
pickle.dump(le_location,open('models/le_location.pkl', 'wb'))
```

If models are not found, the system uses a calibrated rule-based estimator.

### 6. Run the application
```bash
streamlit run app.py
```

Open: http://localhost:8501

---

## 📁 File Structure

```
real_estate_agent/
├── app.py                  # Streamlit UI (main entry point)
├── agent_graph.py          # LangGraph StateGraph (7 nodes)
├── predictor.py            # ML model wrapper + rule-based fallback
├── rag_system.py           # FAISS RAG with HuggingFace embeddings
├── knowledge_base.py       # Market documents + comparable data
├── config.py               # Configuration constants
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # (Create this) Saved ML models from M1
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   └── ...
└── faiss_index/            # Auto-created on first run
    └── index.faiss
```

---

## 🎯 Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM Provider | Groq (LLaMA 3.3 70B) | Fast inference, free tier, no OpenAI cost |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, runs locally, no API key |
| Vector Store | FAISS | Open source, fast, persists locally |
| Agent Framework | LangGraph | Explicit state, debuggable, production-grade |
| Temperature | 0.2 | Low temp → fewer hallucinations |
| Anti-hallucination | System prompts + data-only prompts | Grounding LLM in provided data |

---

## 🛡️ Anti-Hallucination Strategies

1. **Data-grounded prompts**: All LLM prompts include specific numeric data from models/RAG
2. **Low temperature (0.2)**: Reduces creative/fabricated outputs
3. **Structured output format**: Forces LLM to follow specific report structure
4. **Conditional language**: Prompts instruct LLM to use "likely", "historically", "based on data"
5. **Sanity checks**: Predicted prices clamped to known city price ranges
6. **Fallback chain**: LLM failure → template-based output (system never crashes)

---

## 📊 Output Report Structure

The generated advisory report includes:
1. **Executive Summary** — 2-3 sentence overview
2. **Property Valuation** — LR, RF, Ensemble predictions + analytics
3. **Market Overview** — RAG-retrieved city-specific insights
4. **Comparable Property Analysis** — 4 scored comparable properties
5. **Risk Assessment** — 3-4 specific risk factors with severity
6. **Investment Recommendation** — BUY / HOLD / AVOID with justification
7. **Action Plan** — 3 concrete next steps
8. **Disclaimer** — Financial/legal notice

---

## 📝 Notes

- First run downloads the SentenceTransformer model (~90MB) and builds the FAISS index
- Subsequent runs load the cached index instantly
- The system works without Groq API key (uses template-based outputs as fallback)
- All 3 cities (Delhi, Mumbai, Pune) are supported matching Milestone 1 dataset