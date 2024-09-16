import os
from typing import Dict, List

MIN_CLUSTER_SIZE = 20
N_TOPICS = 100
N_TOP_WORDS = 15

# Directory locations

os.environ["PROJECT_DIR"] = "/home/zc_research/narrativesBERT"
data_folder = os.path.join(os.environ["PROJECT_DIR"], "data")
data_filename = 'earnings_calls_20231017.csv'

N_NEIGHBORS = 10
N_COMPONENTS = 5
MIN_DIST = 0.5
MIN_SAMPLES = 5
MIN_CLUSTER_SIZE = 20
N_TOPICS = 100
N_TOP_WORDS = 15
EMBEDDING_MODELS = ['paraphrase-MiniLM-L6-v2', 'all-MiniLM-L6-v2']

SEED_WORDS : Dict[str, List[str]] = {
    # Financial Performance
    "revenue": ["revenue", "income statement", "sales", "top-line", "total revenue" ],
    "growth": ["growth", "expansion", "increase", "rise", "escalation"],
    "profit": ["profit", "net income", "bottom-line", "earnings", "net profit"],
    "cost": ["cost", "expenses", "expenditure", "overhead", "costs"],
    "cash": ["cash", "cash flow", "liquidity", "cash position", "cash balance"],
    "debt": ["debt", "liabilities", "borrowing", "indebtedness", "debt burden"],
    "equity": ["equity", "shareholders", "stockholders", "ownership", "equity holders"],
    "investment": ["investment", "investing", "capital expenditure", "capex", "investment spending"],
    "dividend": ["dividend", "dividend payment", "dividend yield", "dividend payout", "dividend policy"],
    "risk": ["risk", "uncertainty", "volatility", "risk factors", "risk management"],
    "financial_position": ["financial position", "balance sheet", "financial health", "financial stability", "financial standing"],
    "liquidity": ["liquidity", "liquid assets", "current assets", "quick ratio", "current ratio"],
    "gross_profit_margin": ["gross margin", "profit ratio", "markup percentage", "gross profit rate", "sales margin"],
    "Operating margin": ["operating profit margin", "EBIT margin", "operating income margin", "profit margin", "operational efficiency"],
    "free cash flow": ["cash balance", "cash burn rate", "cash convertion cycle", "cash flow", "cash generation", "cash position"],
    "return on equity": ["return on equity", "equity returns", "shareholder return", "net income to equity", "equity performance", "profitability ratio"],
    "return on assets": ["return on assets", "asset returns", "asset performance", "net income to assets", "asset profitability"],
    "return on investment": ["return on investment", "investment returns", "investment performance", "net income to investment", "investment profitability"],
    "productivity": ["automation", "capacity utilization", "cost cutting", "cost efficiency", "cost reduction", "cost saving", "digital transformation", "efficiency", "labor cost", "labor efficiency", "labor layoff", "labor productivity", "labour cost", "labour efficiency", "labour layoff", "labour productivity", "laid off", "lay off"],
    "impairment": ["allowance", "write-off", "impairment charge", "asset impairment", "goodwill impairment"],
    # Guidance and Outlook
    "next quarter guidance": ["short-term forecast", "upcoming quarter outlook", "near-term projections", "quarterly expectations", "forward guidance"],
    "full-year outlook": ["annual forecast", "yearly projection", "long-term guidance", "fiscal year outlook", "12-month projection"],
    "Long-term financial targets": ["multi-year goals", "strategic financial objectives", "extended financial outlook", "long-range targets", "future financial aims"],
    "Industry forecast": ["industry forecast", "sector outlook", "market projections", "industry trends", "vertical predictions", "sector expectations"],
    "Economic forecast": ["economic forecast", "macroeconomic outlook", "economic projections", "economic trends", "economic expectations"],
    # Market Position
    "market share": ["market share", "market dominance", "market leadership", "market position", "business footprint"],
    "competitive landscape": ["competitive landscape", "competitive environment", "industry rivalry", "market competition", "competitor analysis", "competitive environment", "industry dynamics"],
    "Brand strength": ["brand strength", "brand power", "brand health", "brand recognition", "brand equity", "brand value"],
    "Customer acquisition": ["new customer growth", "client onboarding", "customer wins", "new business generation", "expanding customer base"],
    "Customer retention rates": ["client loyalty", "churn rate", "customer stickiness", "repeat business", "customer longevity"],
    "Net Promoter Score (NPS)": ["customer satisfaction index", "loyalty metric", "referral likelihood", "customer advocacy", "satisfaction score"],
    # Product and Service
    "New product launches": ["product releases", "new offerings", "product introductions", "market debuts", "new solutions"],
    "Product mix changes": ["product portfolio shifts", "offering diversification", "product line adjustments", "sales mix"],
    "Service quality": ["customer satisfaction measures", "service performance indicators", "quality assurance metrics", "service level achievements", "customer experience scores"],
    "research_and_development": ["R&D spending", "innovation funding", "product development costs", "research expenditure", "technology investments"],
    "innovation_pipeline": ["future products", "development roadmap", "upcoming innovations", "product incubation", "new concept funnel"],
    "product_roadmap": ["development timeline", "product strategy", "future releases", "product evolution plan", "feature roadmap"],
    # Operational Efficiency:
    "Cost-cutting initiatives": ["expense reduction", "efficiency programs", "cost optimization", "savings measures", "budget trimming"],
    "Operational improvements": ["process enhancements", "efficiency gains", "operational streamlining", "productivity boosts", "performance upgrades"],
    "Productivity metrics": ["efficiency measures", "output indicators", "performance ratios", "productivity KPIs", "operational effectiveness"],
    "Capacity utilization": ["resource usage", "operational efficiency", "production capacity", "facility utilization", "asset efficiency"],
    "Supply chain efficiency": ["logistics performance", "supply network optimization", "procurement effectiveness", "distribution efficiency", "supply chain streamlining"],
    "Inventory turnover": ["stock rotation", "inventory efficiency", "stock velocity", "goods turnover rate", "inventory churn"],
    # Capital Structure and Allocation:
    "Debt levels": ["borrowings", "financial leverage", "liabilities", "indebtedness", "loan balances"],
    "Debt-to-equity ratio": ["leverage ratio", "capital structure", "financial leverage", "gearing ratio", "debt-to-capital ratio"],
    "Share buyback plans": ["stock repurchase program", "share repurchases", "buyback initiative", "stock retirement", "equity reduction"],
    "Dividend policy": ["payout policy", "shareholder distributions", "dividend strategy", "income distribution plan", "yield policy"],
    "Capital expenditure plans": ["CapEx projections", "investment plans", "asset acquisition strategy", "infrastructure spending", "capital outlays"],
    "Working capital management": ["cash flow management", "liquidity management", "short-term asset management", "operational liquidity", "current asset efficiency"],
    # Growth Strategies:
    "Geographic expansion": ["market entry", "territorial growth", "global reach expansion", "new market penetration", "regional diversification"],
    "Merger and acquisition activities": ["M&A strategy", "corporate takeovers", "business combinations", "acquisition plans", "consolidation efforts"],
    "Market penetration strategies": ["market share growth", "customer base expansion", "sales penetration tactics", "market intensification", "deepening market presence"],
    "Diversification efforts": ["business expansion", "new venture development", "portfolio diversification", "risk spreading", "new market entry"],
    "Partnerships and collaborations": ["strategic alliances", "joint ventures", "cooperative agreements", "business partnerships", "collaborative initiatives"],
    # Sales and Marketing:
    "Sales pipeline": ["sales funnel", "prospect pipeline", "revenue pipeline", "deal flow", "sales forecast"],
    "Backlog or order book status": ["unfilled orders", "work in progress", "future revenue", "committed sales", "order queue"],
    "Customer acquisition costs": ["CAC", "cost per customer", "marketing efficiency", "acquisition spend", "customer onboarding costs"],
    "Lifetime value of customers": ["LTV", "customer worth", "long-term customer value", "client profitability", "customer equity"],
    "Marketing effectiveness": ["ROI on marketing", "campaign performance", "marketing efficiency", "promotional impact", "advertising effectiveness"],
    "Sales force productivity": ["sales efficiency", "rep performance", "sales team effectiveness", "selling productivity", "revenue per salesperson"],
    # Segment Performance:
    "Business unit breakdowns": ["divisional performance", "segment analysis", "unit-level results", "departmental breakdown", "operational segment review"],
    "Geographic segment performance": ["regional results", "country-specific performance", "geographical breakdown", "territorial analysis", "location-based performance"],
    "Product category performance": ["product line results", "category-wise analysis", "product segment breakdown", "offering performance", "product mix analysis"],
    "Customer segment analysis": ["client group performance", "customer cohort analysis", "demographic performance", "user segment breakdown", "target market results"],
    # Cost Management:
    "Raw material costs": ["input costs", "material expenses", "commodity prices", "resource costs", "supply expenses"],
    "Labor costs": ["workforce expenses", "employee costs", "payroll expenses", "human resource costs", "wage and salary expenses"],
    "Overhead expenses": ["indirect costs", "fixed costs", "operating expenses", "overhead burden", "non-direct expenses"],
    "Cost of goods sold (COGS)": ["production costs", "direct costs", "manufacturing expenses", "cost of sales", "product costs"],
    "Selling, general, and administrative expenses (SG&A)": ["operating expenses", "overhead costs", "non-production costs", "administrative burden", "commercial expenses"],
    # Risk Management:
    "Regulatory challenges": ["compliance issues", "legal hurdles", "regulatory environment", "policy challenges", "governmental constraints"],
    "Litigation updates": ["legal proceedings", "lawsuit status", "court case developments", "legal dispute updates", "judicial proceedings"],
    "Cybersecurity measures": ["data protection", "information security", "cyber defense", "digital safeguards", "IT security"],
    "Foreign exchange impact": ["currency effects", "forex exposure", "exchange rate influence", "monetary conversion impact", "currency risk"],
    "Interest rate sensitivity": ["rate exposure", "interest risk", "borrowing cost sensitivity", "debt expense fluctuation", "interest rate impact"],
    # Human Capital:
    "Employee headcount": ["workforce size", "staff numbers", "personnel count", "employee strength", "team size"],
    "Employee turnover rate": ["staff attrition", "churn rate", "workforce stability", "retention challenges", "employee departures"],
    "Talent acquisition and retention strategies": ["hiring initiatives", "employee retention programs", "workforce planning", "talent management", "recruitment strategies"],
    "Workforce diversity and inclusion": ["diversity metrics", "inclusivity efforts", "equal opportunity initiatives", "workforce representation", "cultural diversity"],
    "Employee engagement metrics": ["staff satisfaction", "workforce morale", "employee loyalty", "job satisfaction", "team engagement"],
    # Technology and Digital:
    "Digital transformation initiatives": ["digitalization efforts", "tech modernization", "digital evolution", "IT transformation", "technology upgrade"],
    "IT infrastructure investments": ["tech spending", "system upgrades", "IT capex", "technology infrastructure", "computing resources"],
    "E-commerce performance": ["online sales", "digital revenue", "web store results", "internet retail performance", "online marketplace metrics"],
    "Data analytics capabilities": ["business intelligence", "data-driven insights", "analytics infrastructure", "information analysis", "predictive modeling"],
    "Artificial intelligence and machine learning applications": ["AI integration", "ML implementation", "intelligent automation", "cognitive computing", "smart algorithms"],
    # Sustainability and ESG:
    "Environmental initiatives": ["eco-friendly programs", "green initiatives", "sustainability efforts", "environmental stewardship", "ecological projects"],
    "Social responsibility programs": ["community initiatives", "social impact", "corporate citizenship", "philanthropic efforts", "societal contributions"],
    "Governance practices": ["corporate governance", "board practices", "management oversight", "ethical leadership", "shareholder rights"],
    "Carbon footprint reduction efforts": ["emissions reduction", "climate impact mitigation", "greenhouse gas reduction", "carbon neutrality efforts", "environmental impact reduction"],
    "Sustainable sourcing": ["ethical procurement", "responsible sourcing", "supply chain sustainability", "green purchasing", "eco-friendly suppliers"],
    # Intellectual Property:
    "Patent portfolio": ["IP assets", "patent holdings", "invention rights", "proprietary technology", "patented innovations"],
    "Trademark developments": ["brand protection", "trademark portfolio", "intellectual property rights", "brand assets", "trademark strategy"],
    "Licensing agreements": ["IP licensing", "technology transfer", "patent licensing", "trademark licensing", "copyright agreements"],
    "IP litigation": ["patent disputes", "trademark infringement", "copyright cases", "intellectual property lawsuits", "IP legal battles"],
    # Customer-centric Metrics:
    "Customer satisfaction scores": ["client happiness index", "satisfaction ratings", "customer feedback metrics", "service quality scores", "consumer contentment measures"],
    "Churn rate": ["customer attrition", "client loss rate", "turnover rate", "defection rate", "customer departure frequency"],
    "Average revenue per user (ARPU)": ["per-customer revenue", "user monetization", "client value", "revenue intensity", "customer yield"],
    "Customer lifetime value (CLV)": ["lifetime customer worth", "long-term client value", "customer profitability", "total customer value", "client lifetime worth"],
    # Pricing Strategies:
    "Pricing power": ["price elasticity", "pricing leverage", "value capture ability", "price setting ability", "margin potential"],
    "Discount policies": ["price reduction strategies", "promotional pricing", "markdown strategies", "price concessions", "rebate programs"],
    "Dynamic pricing initiatives": ["real-time pricing", "adaptive pricing", "flexible pricing", "demand-based pricing", "price optimization"],
    "Bundle pricing strategies": ["package deals", "product bundling", "combined offering prices", "multi-product discounts", "solution pricing"],
    # Corporate Structure:
    "Organizational changes": ["structural shifts", "corporate reorganization", "business restructuring", "organizational redesign", "company realignment"],
    "Executive leadership transitions": ["C-suite changes", "management shuffle", "leadership succession", "executive appointments", "senior management changes"],
    "Board composition": ["director lineup", "board structure", "governance makeup", "board demographics", "directorship changes"],
    "Subsidiary performance": ["division results", "affiliate performance", "business unit outcomes", "subsidiary contributions", "controlled entity results"],
    # Industry-specific Metrics:
    "Sector-specific KPIs": ["industry benchmarks", "vertical-specific metrics", "sector performance indicators", "industry standards", "niche measurements"],
    "Regulatory compliance metrics": ["compliance scores", "regulatory adherence measures", "conformity indicators", "rule-following metrics", "policy compliance rates"],
    "Industry benchmarking": ["peer comparison", "competitive benchmarking", "industry standards comparison", "market positioning", "sector performance ranking"],
    # Macroeconomic Factors:
    "Economic indicators affecting the business": ["macro trends", "economic influences", "market conditions", "financial environment", "economic climate"]
}