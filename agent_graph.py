"""
LangGraph-powered Agentic Real Estate Advisory Workflow.

Nodes (in order):
  1. validate_input       → Validate & enrich property details
  2. predict_price        → Run ML / rule-based price prediction
  3. retrieve_market_data → RAG retrieval of market context
  4. analyze_comparables  → Find & compare similar properties
  5. assess_risk          → Evaluate investment risks
  6. generate_advice      → LLM-powered investment recommendation
  7. compile_report       → Structure final advisory report

State flows through all nodes via RealEstateState TypedDict.
"""

from __future__ import annotations

import json
from typing import TypedDict, Optional, Dict, Any, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMP, GROQ_MAX_TOKENS, CITY_META
from predictor import predict_rent, format_inr
from rag_system import get_rag
from knowledge_base import COMPARABLE_PROPERTIES

class RealEstateState(TypedDict):
    # Input
    property_details:   Dict[str, Any]
    user_preferences:   Dict[str, Any]

    # Step outputs
    validation_result:  str
    prediction_result:  Dict[str, Any]
    market_context:     str
    comparables:        List[Dict[str, Any]]
    risk_assessment:    str
    investment_advice:  str
    final_report:       str

    # Metadata
    current_step:   str
    error:          Optional[str]
    step_logs:      List[str]

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=GROQ_TEMP,
        max_tokens=GROQ_MAX_TOKENS,
    )

def validate_input(state: RealEstateState) -> RealEstateState:
    """Validate, sanitize, and enrich incoming property details."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    prefs = state["user_preferences"]

    logs.append("🔍 Step 1: Validating property input...")
    validated = {
        "city":             props.get("city", "Delhi"),
        "location":         props.get("location", ""),
        "size_sqft":        float(props.get("size_sqft", 1000)),
        "rooms":            int(props.get("rooms", 2)),
        "bathrooms":        int(props.get("bathrooms", 2)),
        "balconies":        int(props.get("balconies", 1)),
        "bhk_flag":         int(props.get("bhk_flag", 1)),
        "status":           props.get("status", "Semi-Furnished"),
        "property_type":    props.get("property_type", "Apartment"),
        "is_negotiable":    int(props.get("is_negotiable", 0)),
        "security_deposit": float(props.get("security_deposit", 0)),
        "latitude":         float(props.get("latitude", 0)),
        "longitude":        float(props.get("longitude", 0)),
    }

    prefs_clean = {
        "investment_horizon":  prefs.get("investment_horizon", "medium"),   # short/medium/long
        "risk_appetite":       prefs.get("risk_appetite", "moderate"),      # low/moderate/high
        "purpose":             prefs.get("purpose", "investment"),          # investment/self-use
        "budget_lakhs":        float(prefs.get("budget_lakhs", 50)),
        "expected_yield_pct":  float(prefs.get("expected_yield_pct", 4.0)),
    }

    validation_msg = (
        f"✅ Property validated:\n"
        f"  City: {validated['city']} | {validated['rooms']}BHK {validated['property_type']}\n"
        f"  Size: {validated['size_sqft']:,.0f} sq ft | Status: {validated['status']}\n"
        f"  Location: {validated['location'] or 'Not specified'}\n"
        f"  Investor profile: {prefs_clean['risk_appetite']} risk | {prefs_clean['investment_horizon']}-term"
    )

    logs.append(validation_msg)

    return {
        **state,
        "property_details": validated,
        "user_preferences": prefs_clean,
        "validation_result": validation_msg,
        "current_step": "predict_price",
        "step_logs": logs,
        "error": None,
    }

def predict_price(state: RealEstateState) -> RealEstateState:
    """Run price prediction models."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]

    logs.append("📊 Step 2: Running price prediction models...")

    try:
        result = predict_rent(props)
        ens    = result.get("Ensemble", result.get("Rule-Based", 20000))
        logs.append(
            f"  Ensemble prediction: {format_inr(ens)}/month\n"
            f"  Source: {result.get('model_source', 'unknown')}"
        )
    except Exception as e:
        logs.append(f"  ⚠️ Prediction error: {e}")
        result = {"Ensemble": 25000, "model_source": "error_fallback", "analytics": {}}

    return {
        **state,
        "prediction_result": result,
        "current_step": "retrieve_market_data",
        "step_logs": logs,
    }


def retrieve_market_data(state: RealEstateState) -> RealEstateState:
    """Retrieve city-specific market insights via RAG."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    city  = props["city"]

    logs.append(f"🗂️  Step 3: Retrieving market data for {city}...")

    try:
        rag = get_rag()
        query = (
            f"{city} rental market trends investment outlook "
            f"{props['property_type']} {props['rooms']}BHK {props['location']}"
        )
        context = rag.retrieve(query, city=city, k=5)
        logs.append(f"  Retrieved {len(context.split(chr(10)))} lines of market context")
    except Exception as e:
        logs.append(f"  ⚠️ RAG error: {e}")
        city_meta = CITY_META.get(city, {})
        context = (
            f"Market data for {city}: avg yield {city_meta.get('avg_yield', 3.5)}%, "
            f"YoY growth {city_meta.get('yoy_growth', 8.0)}%, "
            f"vacancy {city_meta.get('vacancy_rate', 10.0)}%."
        )

    return {
        **state,
        "market_context": context,
        "current_step": "analyze_comparables",
        "step_logs": logs,
    }


def analyze_comparables(state: RealEstateState) -> RealEstateState:
    """Find and score comparable properties."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    city  = props["city"]
    rooms = props["rooms"]
    size  = props["size_sqft"]

    logs.append(f"🏘️  Step 4: Finding comparable properties in {city}...")

    city_comps = COMPARABLE_PROPERTIES.get(city, [])

    scored = []
    for comp in city_comps:
        room_diff = abs(comp["bhk"] - rooms)
        size_diff = abs(comp["size"] - size) / max(size, 1)
        similarity = 1.0 - (room_diff * 0.4 + size_diff * 0.6)
        scored.append({**comp, "similarity_score": round(similarity, 2)})

    comparables = sorted(scored, key=lambda x: x["similarity_score"], reverse=True)[:4]
    logs.append(f"  Found {len(comparables)} comparable properties")

    return {
        **state,
        "comparables": comparables,
        "current_step": "assess_risk",
        "step_logs": logs,
    }


def _preview_text(text: str, limit: int = 1200) -> str:
    clean = str(text or "").strip().replace("\n", " | ")
    if len(clean) <= limit:
        return clean
    return clean[:limit] + " ...[truncated]"


def _masked_api_key(key: str) -> str:
    k = str(key or "").strip()
    if not k:
        return "<missing>"
    if len(k) <= 10:
        return k[0] + "***" + k[-1]
    return f"{k[:4]}...{k[-4:]} (len={len(k)})"

RISK_SYSTEM_PROMPT = """You are a senior real estate risk analyst specializing in Indian property markets (Delhi, Mumbai, Pune).
Your role is to provide concise, factual risk assessments. Be specific, not generic.
Avoid hallucination — base all claims on the provided market data.
Structure: 3-4 bullet points max. Each bullet: risk factor + severity (Low/Medium/High) + brief reason."""

def assess_risk(state: RealEstateState) -> RealEstateState:
    """LLM-powered risk assessment using market context."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    prefs = state["user_preferences"]
    pred  = state["prediction_result"]
    ctx   = state["market_context"]
    comps = state["comparables"]

    logs.append("⚠️  Step 5: Assessing investment risks...")

    llm = _get_llm()

    analytics = pred.get("analytics", {})
    ens_rent  = pred.get("Ensemble", pred.get("Rule-Based", 20000))

    comp_summary = "\n".join([
        f"  - {c['name']}: ₹{c['rent']:,}/mo, {c['size']} sqft, yield {c['yield']}%"
        for c in comps[:3]
    ])

    prompt = f"""
Property Details:
  City: {props['city']} | Location: {props['location']}
  Type: {props['rooms']}BHK {props['property_type']} | Size: {props['size_sqft']} sqft
  Furnishing: {props['status']}

Predicted Monthly Rent: ₹{ens_rent:,.0f}
Gross Rental Yield: {analytics.get('gross_yield_pct', 'N/A')}%
Price-to-Rent Ratio: {analytics.get('price_to_rent_ratio', 'N/A')}x
Investor Profile: {prefs['risk_appetite']} risk appetite | {prefs['investment_horizon']}-term horizon

Comparable Properties (market):
{comp_summary}

Market Context:
{ctx[:800]}

Provide a concise risk assessment (3-4 bullet points) identifying key risks for this specific property investment.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=RISK_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        risk_text = response.content
        logs.append("  Risk assessment generated")
        logs.append(f"  🧠 LLM Risk Response: {_preview_text(risk_text)}")
        print("\n===== LLM RISK RESPONSE =====\n" + str(risk_text) + "\n=============================\n")
    except Exception as e:
        logs.append(f"  ⚠️ LLM error: {e}")
        risk_text = (
            f"• Vacancy Risk (Medium): {props['city']} {props['property_type']} segment "
            f"has ~{analytics.get('vacancy_rate', 10)}% vacancy rate.\n"
            f"• Yield Risk (Medium): Gross yield of {analytics.get('gross_yield_pct', 3.5)}% "
            f"is {'above' if analytics.get('gross_yield_pct', 3.5) > 4 else 'below'} city average.\n"
            f"• Liquidity Risk (High): Real estate illiquidity may affect exit strategy.\n"
            f"• Regulatory Risk (Low): RERA registration provides buyer protection."
        )
        logs.append(f"  🧩 Fallback Risk Response: {_preview_text(risk_text)}")
        print("\n===== FALLBACK RISK RESPONSE =====\n" + str(risk_text) + "\n==================================\n")

    return {
        **state,
        "risk_assessment": risk_text,
        "current_step": "generate_advice",
        "step_logs": logs,
    }

ADVICE_SYSTEM_PROMPT = """You are an expert Indian real estate investment advisor with 20 years of experience
in Delhi, Mumbai, and Pune markets. You provide clear, actionable, evidence-based investment recommendations.

Rules to prevent hallucination:
1. Only use data explicitly provided in the prompt.
2. Always cite the specific metric you're basing recommendations on.
3. Use conditional language when market conditions are uncertain ("likely", "historically", "based on data").
4. Include a clear BUY / HOLD / AVOID recommendation with justification.
5. Be concise — max 300 words. Structure with: Summary, Key Metrics, Recommendation, Action Steps."""

def generate_advice(state: RealEstateState) -> RealEstateState:
    """Generate comprehensive investment recommendation using LLM."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    prefs = state["user_preferences"]
    pred  = state["prediction_result"]
    ctx   = state["market_context"]
    comps = state["comparables"]
    risks = state["risk_assessment"]

    logs.append("💡 Step 6: Generating investment recommendation...")

    llm = _get_llm()

    analytics = pred.get("analytics", {})
    ens_rent  = pred.get("Ensemble", pred.get("Rule-Based", 20000))
    lr_rent   = pred.get("Linear Regression", ens_rent * 0.95)
    rf_rent   = pred.get("Random Forest", ens_rent * 1.05)

    comp_detail = "\n".join([
        f"  {i+1}. {c['name']}: ₹{c['rent']:,}/mo | {c['size']}sqft | "
        f"Yield {c['yield']}% | {c['furnished']} | Similarity {c['similarity_score']:.0%}"
        for i, c in enumerate(comps[:4])
    ])

    prompt = f"""
=== PROPERTY UNDER ANALYSIS ===
City: {props['city']} | Location: {props['location'] or 'Not specified'}
Property: {props['rooms']}BHK {props['property_type']} | {props['size_sqft']:,.0f} sq ft | {props['status']}
Security Deposit: ₹{props['security_deposit']:,.0f} | Negotiable: {'Yes' if props['is_negotiable'] else 'No'}

=== MODEL PREDICTIONS ===
Linear Regression: ₹{lr_rent:,.0f}/month
Random Forest: ₹{rf_rent:,.0f}/month  
Ensemble (recommended): ₹{ens_rent:,.0f}/month
Annual Rent: ₹{analytics.get('annual_rent', ens_rent*12):,.0f}
Gross Yield: {analytics.get('gross_yield_pct', 'N/A')}%
Price-to-Rent Ratio: {analytics.get('price_to_rent_ratio', 'N/A')}x
City Avg Yield: {analytics.get('city_avg_yield', 'N/A')}%
YoY Rental Growth: {analytics.get('yoy_growth', 'N/A')}%

=== COMPARABLE PROPERTIES ===
{comp_detail}

=== RISK ASSESSMENT ===
{risks}

=== INVESTOR PROFILE ===
Purpose: {prefs['purpose']} | Horizon: {prefs['investment_horizon']}-term
Risk Appetite: {prefs['risk_appetite']} | Budget: ₹{prefs['budget_lakhs']} Lakhs
Expected Yield: {prefs['expected_yield_pct']}%

=== MARKET CONTEXT (RAG) ===
{ctx[:1000]}

Provide a structured investment recommendation following the format:
**RECOMMENDATION: [BUY / HOLD / AVOID]**
**Summary:** (2-3 sentences)
**Key Metrics Analysis:** (3-4 bullet points referencing specific data above)
**Action Steps:** (3 numbered steps)
"""

    try:
        response = llm.invoke([
            SystemMessage(content=ADVICE_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        advice = response.content
        logs.append("  Investment advice generated")
        logs.append(f"  🧠 LLM Advice Response: {_preview_text(advice)}")
        print("\n===== LLM ADVICE RESPONSE =====\n" + str(advice) + "\n===============================\n")
    except Exception as e:
        logs.append(f"  ⚠️ LLM error: {e}")
        yield_above = analytics.get("gross_yield_pct", 3.5) >= analytics.get("city_avg_yield", 3.5)
        advice = (
            f"**RECOMMENDATION: {'BUY' if yield_above else 'HOLD'}**\n\n"
            f"**Summary:** The {props['rooms']}BHK {props['property_type']} in {props['city']} "
            f"shows a gross yield of {analytics.get('gross_yield_pct', 'N/A')}% "
            f"which is {'above' if yield_above else 'below'} the city average.\n\n"
            f"**Action Steps:**\n"
            f"1. Verify RERA registration of the property\n"
            f"2. Negotiate rent to ₹{ens_rent*0.95:,.0f} (5% below estimate)\n"
            f"3. Insist on formal 11-month Leave & Licence agreement"
        )
        logs.append(f"  🧩 Fallback Advice Response: {_preview_text(advice)}")
        print("\n===== FALLBACK ADVICE RESPONSE =====\n" + str(advice) + "\n====================================\n")

    return {
        **state,
        "investment_advice": advice,
        "current_step": "compile_report",
        "step_logs": logs,
    }


REPORT_SYSTEM_PROMPT = """You are a professional real estate report writer. Create structured, formal advisory reports.
Format: Use markdown with clear sections. Be precise with numbers. Include disclaimer at end.
Tone: Professional, data-driven, balanced. Avoid superlatives and marketing language."""

def compile_report(state: RealEstateState) -> RealEstateState:
    """Compile all analysis into a structured advisory report."""
    logs  = state.get("step_logs", [])
    props = state["property_details"]
    prefs = state["user_preferences"]
    pred  = state["prediction_result"]
    comps = state["comparables"]
    risks = state["risk_assessment"]
    advice = state["investment_advice"]

    logs.append("📝 Step 7: Compiling final advisory report...")

    llm       = _get_llm()
    analytics = pred.get("analytics", {})
    ens_rent  = pred.get("Ensemble", pred.get("Rule-Based", 20000))

    comp_table = "\n".join([
        f"| {c['name']} | ₹{c['rent']:,} | {c['size']} | {c['yield']}% | {c['furnished']} |"
        for c in comps[:4]
    ])

    prompt = f"""
Create a complete investment advisory report for:

PROPERTY: {props['rooms']}BHK {props['property_type']} in {props['location']}, {props['city']}
Size: {props['size_sqft']:,.0f} sq ft | {props['status']}
Predicted Rent: ₹{ens_rent:,.0f}/month | Annual: ₹{analytics.get('annual_rent', ens_rent*12):,.0f}
Gross Yield: {analytics.get('gross_yield_pct', 'N/A')}% (City Avg: {analytics.get('city_avg_yield', 'N/A')}%)
Price-to-Rent: {analytics.get('price_to_rent_ratio', 'N/A')}x | YoY Growth: {analytics.get('yoy_growth', 'N/A')}%

INVESTOR: {prefs['risk_appetite']} risk | {prefs['investment_horizon']}-term | Budget ₹{prefs['budget_lakhs']}L

COMPARABLE PROPERTIES:
| Property | Rent/mo | Size sqft | Yield | Furnished |
|----------|---------|-----------|-------|-----------|
{comp_table}

RISK ASSESSMENT:
{risks}

INVESTMENT RECOMMENDATION:
{advice}

Generate a professional advisory report with these EXACT sections:
# Real Estate Investment Advisory Report

## 1. Executive Summary
## 2. Property Valuation
## 3. Market Overview  
## 4. Comparable Property Analysis
## 5. Risk Assessment
## 6. Investment Recommendation
## 7. Action Plan
## 8. Disclaimer

Include all data provided. Be concise and factual. Word limit: 600 words.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        report = response.content
        logs.append("  ✅ Report compiled successfully")
        logs.append(f"  🧠 LLM Report Response: {_preview_text(report)}")
        print("\n===== LLM REPORT RESPONSE =====\n" + str(report) + "\n===============================\n")
    except Exception as e:
        logs.append(f"  ⚠️ Report LLM error: {e}. Using template.")
        report = _template_report(props, prefs, pred, comps, risks, advice)
        logs.append(f"  🧩 Fallback Report Response: {_preview_text(report)}")
        print("\n===== FALLBACK REPORT RESPONSE =====\n" + str(report) + "\n====================================\n")

    return {
        **state,
        "final_report": report,
        "current_step": "done",
        "step_logs": logs,
    }


def _template_report(props, prefs, pred, comps, risks, advice):
    """Fallback template report when LLM unavailable."""
    analytics = pred.get("analytics", {})
    ens_rent  = pred.get("Ensemble", pred.get("Rule-Based", 20000))
    return f"""
# Real Estate Investment Advisory Report

## 1. Executive Summary
Analysis of {props['rooms']}BHK {props['property_type']} in {props['location']}, {props['city']}.
Predicted rent: ₹{ens_rent:,.0f}/month with gross yield of {analytics.get('gross_yield_pct', 'N/A')}%.

## 2. Property Valuation
- **Predicted Monthly Rent (Ensemble):** ₹{ens_rent:,.0f}
- **Annual Rental Income:** ₹{analytics.get('annual_rent', ens_rent*12):,.0f}
- **Gross Yield:** {analytics.get('gross_yield_pct', 'N/A')}%
- **Price-to-Rent Ratio:** {analytics.get('price_to_rent_ratio', 'N/A')}x

## 3. Market Overview
City YoY growth: {analytics.get('yoy_growth', 'N/A')}% | Vacancy: {analytics.get('vacancy_rate', 'N/A')}%

## 4. Comparable Property Analysis
{chr(10).join([f"- {c['name']}: ₹{c['rent']:,}/mo, Yield {c['yield']}%" for c in comps[:3]])}

## 5. Risk Assessment
{risks}

## 6. Investment Recommendation
{advice}

## 7. Action Plan
1. Verify RERA registration
2. Conduct site visit and title verification
3. Negotiate terms based on comparable data above

## 8. Disclaimer
This report is generated by an AI system for informational purposes only and does not constitute
financial or legal advice. Consult a SEBI-registered financial advisor and a qualified lawyer before
making any real estate investment decisions. Past market performance does not guarantee future returns.
"""



def build_graph() -> StateGraph:
    graph = StateGraph(RealEstateState)

    # Add all nodes
    graph.add_node("validate_input",       validate_input)
    graph.add_node("predict_price",        predict_price)
    graph.add_node("retrieve_market_data", retrieve_market_data)
    graph.add_node("analyze_comparables",  analyze_comparables)
    graph.add_node("assess_risk",          assess_risk)
    graph.add_node("generate_advice",      generate_advice)
    graph.add_node("compile_report",       compile_report)

    # Linear pipeline edges
    graph.set_entry_point("validate_input")
    graph.add_edge("validate_input",       "predict_price")
    graph.add_edge("predict_price",        "retrieve_market_data")
    graph.add_edge("retrieve_market_data", "analyze_comparables")
    graph.add_edge("analyze_comparables",  "assess_risk")
    graph.add_edge("assess_risk",          "generate_advice")
    graph.add_edge("generate_advice",      "compile_report")
    graph.add_edge("compile_report",       END)

    return graph.compile()


_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_advisory(property_details: dict, user_preferences: dict) -> RealEstateState:
    """Entry point — run the full advisory pipeline."""
    print("🤖 agent_graph.run_advisory started")

    initial_state: RealEstateState = {
        "property_details":  property_details,
        "user_preferences":  user_preferences,
        "validation_result": "",
        "prediction_result": {},
        "market_context":    "",
        "comparables":       [],
        "risk_assessment":   "",
        "investment_advice": "",
        "final_report":      "",
        "current_step":      "validate_input",
        "error":             None,
        "step_logs":         [],
    }

    graph  = get_graph()
    result = graph.invoke(initial_state)
    print(f"✅ agent_graph completed ({len(result.get('step_logs', []))} step logs)")
    return result


def stream_advisory(property_details: dict, user_preferences: dict):
    """Stream advisory pipeline — yields (node_name, state) after each node."""
    initial_state: RealEstateState = {
        "property_details":  property_details,
        "user_preferences":  user_preferences,
        "validation_result": "",
        "prediction_result": {},
        "market_context":    "",
        "comparables":       [],
        "risk_assessment":   "",
        "investment_advice": "",
        "final_report":      "",
        "current_step":      "validate_input",
        "error":             None,
        "step_logs":         [],
    }
    graph = get_graph()
    final_state = initial_state
    for step_output in graph.stream(initial_state):
        node_name  = list(step_output.keys())[0]
        final_state = step_output[node_name]
        yield node_name, final_state
    return final_state