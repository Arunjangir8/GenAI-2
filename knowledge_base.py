"""
knowledge_base.py
Real-estate market knowledge documents for RAG retrieval.
These are used to populate the FAISS vector store.
"""

MARKET_DOCUMENTS = [

    # Delhi 
    {
        "id": "delhi_overview",
        "city": "Delhi",
        "category": "market_overview",
        "text": """
Delhi Real Estate Market Overview 2024-25:
Delhi's rental market remains one of India's most active with average rental yields of 3.0–4.0%.
Key micro-markets include Dwarka (affordable, family-friendly), South Delhi (premium, high demand),
Noida Extension (fast-growing), and Gurugram (IT corridor, high appreciation).
Year-on-year rental growth in Delhi NCR stands at approximately 8–10% driven by infrastructure
upgrades like the Delhi Metro Phase 4 expansion, new expressways, and the Dwarka Expressway.
Supply of new housing units in Delhi NCR reached 52,000 in FY2024, with demand outpacing supply
in the 2–3 BHK segment (1000–1500 sq ft). Vacancy rates hover around 12.5% for luxury (>₹50,000/mo),
and 6% for mid-segment (₹15,000–₹30,000/mo).
"""
    },
    {
        "id": "delhi_localities",
        "city": "Delhi",
        "category": "locality_insights",
        "text": """
Delhi Top Rental Localities 2025:
1. Dwarka: Avg rent ₹18,000–₹30,000 for 2BHK. Strong metro connectivity (Blue Line & upcoming Phase 4).
   Suitable for families. Appreciation potential: 8% YoY.
2. South Extension / Greater Kailash: Premium zone, avg ₹35,000–₹80,000 for 2–3BHK.
   High-end retail, schools, hospitals nearby. Rental yield ~2.8%.
3. Saket: Well-connected, avg ₹25,000–₹55,000. Close to Select Citywalk Mall, metro on Yellow Line.
4. Lajpat Nagar: Lively market area, avg ₹20,000–₹40,000. Strong demand from working professionals.
5. Rohini: North Delhi hub, affordable ₹12,000–₹22,000 for 2BHK. Good metro access (Red Line).
6. Janakpuri: West Delhi residential, avg ₹15,000–₹28,000. Stable rental demand, family-oriented.
7. Vasant Kunj: Premium residential, avg ₹30,000–₹65,000. Close to Aerocity and NH-48.
"""
    },
    {
        "id": "delhi_investment",
        "city": "Delhi",
        "category": "investment_analysis",
        "text": """
Delhi Investment Outlook 2025:
Best areas for rental investment: Dwarka Sector 10–22 (Metro Phase 4 impact, 10–12% appreciation expected),
Narela (emerging, industrial zone proximity), L-Zone (Dwarka Expressway).
Risk factors: High stamp duty (6–8%), circle rate increases, older inventory in Central Delhi.
RERA Delhi compliance is mandatory for all builders. Verified listings show ~15% premium over
non-RERA properties. Security deposit norms: 2–3 months rent is standard in Delhi NCR.
For investors: 2BHK units in Dwarka priced ₹60–80 Lakh offer gross yields of 3.8–4.5%.
Studio/1BHK near universities (JNU, DU, IIT Delhi) offer short-term rental yields up to 5%.
"""
    },

    # Mumbai 
    {
        "id": "mumbai_overview",
        "city": "Mumbai",
        "category": "market_overview",
        "text": """
Mumbai Real Estate Market Overview 2024-25:
Mumbai is India's most expensive rental market with average rent per sq ft of ₹55–₹110 in prime zones.
The Mumbai Metropolitan Region (MMR) recorded a 10.1% YoY rental increase in 2024.
Key demand drivers: Financial services, Bollywood, IT/ITES at BKC and Powai, port connectivity.
The Mumbai Trans-Harbour Link (Atal Setu) opened in Jan 2024 significantly reducing travel time
from Navi Mumbai to South Mumbai. Thane and Navi Mumbai are seeing 15%+ appreciation.
New supply is constrained by FSI regulations and coastal zone restrictions. Redevelopment of
old SRA/chawl buildings is the major supply source in central Mumbai. Rental yields average 2.8–3.5%.
"""
    },
    {
        "id": "mumbai_localities",
        "city": "Mumbai",
        "category": "locality_insights",
        "text": """
Mumbai Top Rental Localities 2025:
1. Bandra West: Premium lifestyle hub, avg ₹60,000–₹1,50,000 for 2BHK. Preferred by HNIs,
   expats, media professionals. Sea view premiums add 20–30%.
2. Powai: IT hub near Hiranandani, avg ₹35,000–₹75,000 for 2BHK. Strong corporate demand.
   Proposed metro line will boost values further.
3. Andheri (West/East): Transit hub, avg ₹28,000–₹65,000. East side popular with SMEs/startups.
   Close to MIDC and Sahar (airport). 
4. Dadar: Central Mumbai, avg ₹40,000–₹85,000. Excellent connectivity via WR+CR railways.
5. Thane: ₹18,000–₹40,000 for 2BHK. Fast appreciation due to new infrastructure.
   Ghodbunder Road corridor seeing 12–15% annual growth.
6. Navi Mumbai: Affordable gateway, ₹14,000–₹32,000 for 2BHK. NAINA development, Navi Mumbai
   Airport (expected 2025) to drive 18–20% appreciation.
7. Malad / Goregaon: Mid-segment, ₹25,000–₹55,000. Film City proximity, Link Road access.
"""
    },
    {
        "id": "mumbai_investment",
        "city": "Mumbai",
        "category": "investment_analysis",
        "text": """
Mumbai Investment Outlook 2025:
Top picks for rental investment: Navi Mumbai (pre-airport appreciation play), Thane (growing IT hub),
Panvel (Atal Setu + new airport beneficiary). Avoid over-priced Bandra/Juhu for yield; better for capital gains.
MahaRERA is one of India's strongest regulatory frameworks. 95% of projects must be registered.
Security deposits in Mumbai are typically 3–6 months, higher than other cities.
Stamp duty: 5% for properties >₹30 Lakh in Mumbai. Leave and Licence (L&L) agreements are standard.
Key risk: High entry costs mean long payback periods (25–30 years). Best strategy: mid-segment
(₹80 Lakh–₹1.5 Cr range) in Thane/Navi Mumbai for 4–5% gross yields.
"""
    },

    #  Pune
    {
        "id": "pune_overview",
        "city": "Pune",
        "category": "market_overview",
        "text": """
Pune Real Estate Market Overview 2024-25:
Pune is India's fastest-growing major rental market with 12.4% YoY growth in 2024.
IT/ITES sector employs 7+ lakh people, driving strong 2–3 BHK demand in Hinjewadi, Wakad, Kharadi.
Average rental yield is highest among the three cities at 4.0–5.0% for well-located 2BHKs.
Pune Metro Phase 1 (PCMC to Swargate) operational; Phase 2 (Hinjewadi to Shivajinagar)
under construction and expected 2025-26, will dramatically boost Hinjewadi/Wakad values.
New supply: 48,000 units launched in FY2024 with absorption rate of 82% — healthy market.
The ₹20,000–₹35,000/month segment (2BHK, 900–1200 sq ft) has virtually zero vacancy in key IT zones.
"""
    },
    {
        "id": "pune_localities",
        "city": "Pune",
        "category": "locality_insights",
        "text": """
Pune Top Rental Localities 2025:
1. Hinjewadi: IT hub (Rajiv Gandhi IT Park), avg ₹18,000–₹40,000 for 2BHK. 
   Excellent rental demand, 15% YoY appreciation. Upcoming metro is a major catalyst.
2. Kharadi: East Pune IT corridor (EON IT Park), avg ₹20,000–₹45,000. Strong corporate rentals.
   Kalyani Nagar and Viman Nagar proximity adds lifestyle value.
3. Kothrud: Premium residential, avg ₹22,000–₹50,000. Posh address, family-oriented.
   Good schools, hospitals. Stable long-term rentals.
4. Baner: Upscale suburb near Hinjewadi, avg ₹25,000–₹55,000. Aundh Road connectivity.
   High demand from IT middle management.
5. Wakad: Emerging IT zone, avg ₹15,000–₹32,000. Affordable alternative to Hinjewadi.
   Excellent highway connectivity (Mumbai-Pune Expressway).
6. Hadapsar: Magarpatta IT hub, avg ₹14,000–₹28,000. Large township projects.
7. Undri / Pisoli: Affordable south Pune, avg ₹10,000–₹22,000. Growing steadily.
"""
    },
    {
        "id": "pune_investment",
        "city": "Pune",
        "category": "investment_analysis",
        "text": """
Pune Investment Outlook 2025:
Best investment picks: Hinjewadi Phase 3 (pre-metro speculation), Kharadi (sustained IT demand),
Wagholi (budget play, industrial growth). Avoid: areas beyond 20 km from IT hubs (poor liquidity).
MahaRERA governs Pune; 90%+ new projects are registered. Tenant protections are strong.
Security deposit: 2–3 months standard. Lease agreements are typically 11 months + renewal.
Stamp duty: 5% (waived partially for women buyers—4%). GST applicable on new properties.
Best yields: 2BHK in Hinjewadi (₹55–70 Lakh investment) gives gross yield of 4.5–5.2%.
Return on Investment projections: Hinjewadi 3BHK purchased at ₹90L → total returns (rent+appreciation)
of 18–22% over 3 years is realistic given metro completion + demand growth.
"""
    },

    #  National Regulations
    {
        "id": "rera_regulations",
        "city": "All",
        "category": "regulations",
        "text": """
RERA (Real Estate Regulatory Authority) - Key Points for Investors 2025:
Under the Real Estate (Regulation and Development) Act, 2016:
- All projects with >500 sq m plot area or >8 apartments must be registered with state RERA.
- Builders must park 70% of funds in escrow; protects buyers from fund diversion.
- Possession delay penalty: Builder pays same interest as home loan rate to buyer.
- Buyer can claim refund with interest if builder fails to deliver within promised date.
- Five-year warranty on structural defects after possession.
States: MahaRERA (Maharashtra) is most active; RERA Delhi covers NCR; UP-RERA covers Noida/Greater Noida.
For renters: The Model Tenancy Act 2021 aims to balance landlord-tenant rights. States implementing it:
Andhra Pradesh, Assam, Tamil Nadu, Uttar Pradesh, Uttarakhand (2024 status).
Delhi and Maharashtra have their own Rent Control Acts, offering security to existing tenants.
"""
    },
    {
        "id": "investment_principles",
        "city": "All",
        "category": "investment_principles",
        "text": """
Real Estate Investment Principles for Indian Markets:
1. Gross Rental Yield = Annual Rent / Property Price × 100. Acceptable range: 3–5% in metros.
2. Net Yield = (Annual Rent - Maintenance - Taxes) / Property Price × 100. Typically 0.5–1% lower.
3. Price-to-Rent Ratio: Property Price / Annual Rent. Lower = better for buyers (15–20 good, 25+ consider renting).
4. Appreciation Potential: Tier-1 cities average 7–12% annually; Tier-2 cities 10–18% in growth corridors.
5. Liquidity Risk: Real estate is illiquid. Emergency exit may mean 10–15% haircut on price.
6. Leverage: Most investors use 70–80% LTV loans. EMI should not exceed 40% of monthly income.
7. Tax benefits: Section 24(b) allows ₹2L deduction on home loan interest for self-occupied.
   Rental income taxed as "Income from House Property" with 30% standard deduction.
8. Diversification: Don't invest >40% of portfolio in a single property/city.
9. Due Diligence: Verify RERA registration, title deed, encumbrance certificate, OC/CC.
10. NRI Investment: FEMA allows NRIs to purchase residential property; rent can be repatriated.
"""
    },
    {
        "id": "market_trends_2025",
        "city": "All",
        "category": "macro_trends",
        "text": """
Indian Real Estate Macro Trends 2025:
1. Co-living: Growing 35% YoY. Stanza Living, NestAway, OYO Homes expanding. Good for studio/1BHK investors.
2. Work-from-Home Legacy: Larger homes (3–4BHK) in suburbs seeing sustained demand even post-COVID.
3. Green Buildings: IGBC/LEED certified projects command 8–12% rental premium. Growing awareness.
4. Proptech: Platforms like NoBroker, Housing.com, Magicbricks increasing market transparency.
   Price discovery improving; negotiation margins narrowing (from 10% to 5% in metro markets).
5. Luxury Segment: HNI demand for branded residences (Lodha, Godrej, DLF luxury) surging 40% YoY.
6. Interest Rates: RBI repo rate at 6.25% (2025). Effective home loan rates: 8.5–9.5%.
   Impact: Buyers prefer renting when EMI > 1.5× equivalent rent. Supports rental demand.
7. Data Centers & Warehousing: Industrial real estate growing; spillover effect on residential
   demand in Pune (Chakan), Delhi NCR (Kundli-Manesar-Palwal), Mumbai (Bhiwandi).
8. Infrastructure Push: PM Gati Shakti, Smart Cities Mission adding long-term value uplift
   in 100+ cities. Best plays: Ahmedabad, Surat, Indore, Lucknow for next decade.
"""
    }
]

COMPARABLE_PROPERTIES = {
    "Delhi": [
        {"name": "Dwarka 2BHK Independent Floor", "size": 950, "bhk": 2, "rent": 22000, "yield": 3.9, "furnished": "Semi-Furnished"},
        {"name": "Rohini 2BHK Apartment", "size": 1050, "bhk": 2, "rent": 18000, "yield": 3.5, "furnished": "Unfurnished"},
        {"name": "Saket 3BHK Apartment", "size": 1500, "bhk": 3, "rent": 45000, "yield": 3.2, "furnished": "Furnished"},
        {"name": "Janakpuri 2BHK Builder Floor", "size": 900, "bhk": 2, "rent": 20000, "yield": 3.7, "furnished": "Semi-Furnished"},
        {"name": "Greater Kailash 3BHK", "size": 1800, "bhk": 3, "rent": 70000, "yield": 2.8, "furnished": "Furnished"},
        {"name": "Vasant Kunj 2BHK", "size": 1100, "bhk": 2, "rent": 35000, "yield": 3.0, "furnished": "Furnished"},
    ],
    "Mumbai": [
        {"name": "Andheri 2BHK Apartment", "size": 750, "bhk": 2, "rent": 42000, "yield": 3.1, "furnished": "Semi-Furnished"},
        {"name": "Powai 2BHK Hiranandani", "size": 850, "bhk": 2, "rent": 55000, "yield": 2.9, "furnished": "Furnished"},
        {"name": "Dadar 1BHK Apartment", "size": 500, "bhk": 1, "rent": 32000, "yield": 3.0, "furnished": "Semi-Furnished"},
        {"name": "Thane 2BHK New Complex", "size": 900, "bhk": 2, "rent": 28000, "yield": 3.8, "furnished": "Semi-Furnished"},
        {"name": "Malad 2BHK Apartment", "size": 800, "bhk": 2, "rent": 38000, "yield": 3.2, "furnished": "Furnished"},
        {"name": "Navi Mumbai 3BHK", "size": 1200, "bhk": 3, "rent": 30000, "yield": 4.1, "furnished": "Unfurnished"},
    ],
    "Pune": [
        {"name": "Hinjewadi 2BHK Apartment", "size": 950, "bhk": 2, "rent": 26000, "yield": 4.8, "furnished": "Semi-Furnished"},
        {"name": "Kharadi 2BHK Apartment", "size": 1000, "bhk": 2, "rent": 28000, "yield": 4.5, "furnished": "Furnished"},
        {"name": "Kothrud 3BHK Independent Floor", "size": 1400, "bhk": 3, "rent": 38000, "yield": 4.0, "furnished": "Furnished"},
        {"name": "Baner 2BHK Apartment", "size": 900, "bhk": 2, "rent": 30000, "yield": 4.6, "furnished": "Furnished"},
        {"name": "Wakad 2BHK Budget", "size": 850, "bhk": 2, "rent": 20000, "yield": 4.2, "furnished": "Semi-Furnished"},
        {"name": "Hadapsar 2BHK", "size": 1000, "bhk": 2, "rent": 22000, "yield": 4.4, "furnished": "Semi-Furnished"},
    ]
}