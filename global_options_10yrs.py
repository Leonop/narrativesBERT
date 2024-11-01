# this global_options includes 10 years of earnings call data from 2010 to 2020
# the longer document input, it requires adjusting the parameters in the BERTopic model

import os
from typing import Dict, List
import pandas as pd
# MIN_CLUSTER_SIZE = 20
# N_TOPICS = 100
# N_TOP_WORDS = 15

# Directory locations
UNIQUE_KEYS = ['companyid', 'keydevid', 'transcriptid', 'transcriptcomponentid'] # composite key in earnings call data
PROJECT_DIR = os.getcwd()
data_folder = os.path.join(PROJECT_DIR, "data")
model_folder = os.path.join(PROJECT_DIR, "model")
output_folder = os.path.join(PROJECT_DIR, "output")
output_fig_folder = os.path.join(output_folder, "fig")
data_filename = 'earnings_calls_20231017.csv'
stop_list = pd.read_csv(os.path.join(data_folder, "stoplist.csv"))['stopwords'].tolist()
MODEL_SCORES = os.path.join(output_folder, "model_scores.txt")
DATE_COLUMN = "transcriptcreationdate_utc"
TOPIC_SCATTER_PLOT = os.path.join(output_fig_folder, "topic_scatter_plot.pdf")
num_topic_to_plot = 20 # top_N topics to plot
MODEL_SELECTION_RESULTS = os.path.join(output_folder, "model_selection_results.csv")
TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
START_ROWS = 4000000 # start row to read from the csv file
NROWS = 15000000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_FILTER = 2020 # train the model on data from start year to this year
START_YEAR = 2010 # start year of the data
# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
BATCH_SIZE = 500

# create a list of parameters to search over using GridSearchCV
N_NEIGHBORS = [28] # Number of Neighbors in UMAP, the higher number requires more computational power
N_COMPONENTS = [8] # More dimensions might allow for a richer, more nuanced representation of the data, which can help the model distinguish different topics more effectively.
MIN_DIST = [0.0] # Lower value will make the topic more distinct from each other
MIN_SAMPLES = [20] #The higher value will make the topics output fewer topics, but larger, and robust clusters.
MIN_CLUSTER_SIZE = [40] # A large value will lead to larger, more stable clusters, but fewer clusters.
N_TOPICS = [100] # Number of Topics in Topic Model
TOP_N_WORDS = [20] # Number of Top Words in Topic Model
METRIC = ['cosine']
EMBEDDING_MODELS = ['paraphrase-MiniLM-L6-v2'] #'all-MiniLM-L6-v2'
MAX_DF = [0.95] # remove the top 15% of the most frequent words
MIN_DF = [5] # eliminate very rare words
MIN_COUNT = 2 # SMART_N_GRAM minimum number of times a word must appear in the corpus to be included in the vocabulary
THRESHOLD = 5 # SMART_N_GRAM
NR_TOPICS = [250] # Number of Topics in Topic Model

# SAVE RESULTS 
SAVE_RESULTS_COLS = ["params", "score", "probability"]


SEED_TOPICS = [["revenue", "income statement", "sales", "top-line", "total revenue" ],
    ["growth", "expansion", "increase", "rise", "escalation"],
    ["profit", "net income", "bottom-line", "earnings", "net profit"],
    ["cost", "expenses", "expenditure", "overhead", "costs"],
    ["cash", "cash flow", "liquidity", "cash position", "cash balance"],
    ["debt", "liabilities", "borrowing", "indebtedness", "debt burden"],
    ["equity", "shareholders", "stockholders", "ownership", "equity holders"],
    ["investment", "investing", "capital expenditure", "capex", "investment spending"],
    ["dividend", "dividend payment", "dividend yield", "dividend payout", "dividend policy"],
    ["financial position", "balance sheet", "financial health", "financial stability", "financial standing"],
    ["liquidity", "liquid assets", "current assets", "quick ratio", "current ratio"],
    ["gross margin", "profit ratio", "markup percentage", "gross profit rate", "sales margin"],
    ["operating profit margin", "EBIT margin", "operating income margin", "profit margin", "operational efficiency"],
    ["cash balance", "cash burn rate", "cash convertion cycle", "cash flow", "cash generation", "cash position"],
    ["return on equity", "equity returns", "shareholder return", "net income to equity", "equity performance", "profitability ratio"],
    ["return on assets", "asset returns", "asset performance", "net income to assets", "asset profitability"],
    ["return on investment", "investment returns", "investment performance", "net income to investment", "investment profitability"],
    ["automation", "capacity utilization", "cost cutting", "cost efficiency", "cost reduction", "cost saving", "digital transformation", "efficiency", "labor cost", "labor efficiency", "labor layoff", "labor productivity", "labour cost", "labour efficiency", "labour layoff", "labour productivity", "laid off", "lay off"],
    ["allowance", "write-off", "impairment charge", "asset impairment", "goodwill impairment"],
    ["Coprporate tax", "effective tax rate", "tax liabilities", "tax planning", "tax credits", "deferred taxes"],
    ["short-term forecast", "upcoming quarter outlook", "near-term projections", "quarterly expectations", "forward guidance"],
    ["full-year outlook", "annual forecast", "yearly projection", "long-term guidance", "fiscal year outlook", "12-month projection"],
    ["Long-term", "multi-year goals", "strategic financial objectives", "extended financial outlook", "long-range targets", "future financial aims",
    "throughout leader", "investment horizon", "growth strategy", "strategic horizon", "strategic plan", "strategic vision", "strategic objectives", "value creation", "future roadmap", "future-proofing", "market leadership",
    "competitive advantage", "strategic initiatives", "sustainable growth", "multi-year plan", "long-haul", "forward looking", "forward-looking", "future potential", "anticipatory", "next-gen focus", "broad outlook", "extended timeframe", "comprehensive view",
    "longtitudinal approach", "long range perspective", "down the road", "the future", "going forward", "upcoming quarters", "long run", "long term", "long-term", "in future", "enduring value"],
    ["industry forecast", "sector outlook", "market projections", "industry trends", "vertical predictions", "sector expectations"],
    ["economic forecast", "macroeconomic outlook", "economic projections", "economic trends", "economic expectations", "bernanke",
    "biden", "china economy", "composite pmi", "comsumption", "construction pmi", "consumer confidence", "consumer price index", "consumer sentiment", 
    "consumer spending", "core consumer price index", "core cpi", "core ppi", "core producer price index", "corona", "coronavirus", "covid", "CPI", "difficult time", 
    "economic crisis", "economic growth", "economic sanctions", "economy contraction", "economy crisis", "economy development", "economy expansion", "economy recession", 
    "economy", "emerging market", "energy price", "FED", "federal reserve", "financial crisis", "fiscal policy", "foreign exchange", "GDP", "global economic condition", 
    "government spend", "government spending", "government stimulus", "inflation", "interest rate", "jobs data", "jobs report", "labor market", "labour market", "lockdown", 
    "macro", "manufacturing pmi", "market demand", "market fear", "market sentiment", "market uncertainty",
    "market volatility", "monetary policy", "obama", "pandemic", "PCE", "PMI", "powell", "PPI", "producer price index", 
    "purchasing managers index", "recession", "recovery", "remote work", "retail sales growth", "retail sales",
    "services pmi", "shutdown", "soft demand", "supply chain", "trade agreement", "trade deal", "trade dispute", 
    "trade policy", "trade sanctions", "trade tariff", "trade war", "trump", "unemployment", "yellen"],
    ["market share", "market dominance", "market leadership", "market position", "business footprint"],
    ["competitive landscape", "competitive risk", "competitive environment", "industry rivalry", "market competition", "competitor analysis", "competitive environment", "industry dynamics", 
    "adjust strategy", "barrier entry", "big shift", "brand awareness", "brand loyalty", "brand recognition", "catch up", "challenge", "competition", "competitive advantage", "competitive edge", 
    "competitive environment", "competitive intensity", "competitive landscape", "competitive position", "competitive risk", "competitive threat",
    "competitor", "consolidation", "continuous innovation", "continuous optimization", "continuous transformation", "customer loyalty", "customer retention", 
    "customer satisfaction", "differentiate", "differentiation", "difficult environment", "difficult position", "difficult situation",
    "dynamic environment", "dynamic industry", "dynamic market", "grow change", "headwinds", "heat up", "heats up", "price competition", 
    "price war", "resilience", "resilient", "rival", "rivalry", "threat", "tough decision"],
    ["Challenges", "adjustment period", "back-end loaded", "be behind", "bear the brunt", "bear the burden", "behind the curve", "behind the eight ball", "behind the times", "belly up", "big question mark", "big question",
    "bizarre decision", "bleak", "bloodbath", "brake", "bump in the road", "bumpy", "cautious outlook", "challenge", "choppy", "close its doors", "compression", "conservative", "contract",
    "cost growth", "cost pressure", "creep", "curve ball", "curveball", "cut numbers", "dead horse", "directly affected", "directly hit", "directly impacted", "downtime", "dramatically affected", "dramatically impacted", "evaporate",
    "expense growth", "fall apart", "fall off a cliff", "falloff", "flattish", "flatten", "fluid situation", "gap in the road", "have an issue", "headwind", "heavily impacted", "hiccup", "impacted visibility", "in the hole",
    "in the hot seat", "in the red", "In uncharted waters", "Inventory adjustment", "keep up at night", "lack of clarity", "lack of transparency", "lack of visibility", "less visibility", "limited visibility", "low visibility", "lumpiness", "materially affected",
    "materially hit", "materially impacted", "mixed bag", "mixed result", "nightmare", "sustainability impacted"],
    ["brand strength", "brand power", "brand health", "brand recognition", "brand equity", "brand value"],
    ["new customer growth", "client onboarding", "customer wins", "new business generation", "expanding customer base"],
    ["client loyalty", "churn rate", "customer stickiness", "repeat business", "customer longevity"],
    ["customer satisfaction index", "loyalty metric", "referral likelihood", "customer advocacy", "satisfaction score"],
    ["New product launches", "product releases", "new offerings", "product introductions", "market debuts", "new solutions"],
    ["Product mix changes", "product portfolio shifts", "offering diversification", "product line adjustments", "sales mix"],
    ["Service quality", "customer satisfaction measures", "service performance indicators", "quality assurance metrics", "service level achievements", "customer experience scores"],
    ["research_and_development", "R&D spending", "innovation funding", "product development costs", "research expenditure", "technology investments"],
    ["innovation_pipeline", "future products", "development roadmap", "upcoming innovations", "product incubation", "new concept funnel"],
    ["product_roadmap", "development timeline", "product strategy", "future releases", "product evolution plan", "feature roadmap"],
    ["Cost-cutting initiatives", "expense reduction", "efficiency programs", "cost optimization", "savings measures", "budget trimming", "cost control", "cost cutting",
    "cost efficiency", "cost management", "cost minimization", "cost optimization", "cost reduction", "cost saving", "decrease in investment", "digital transformation", 
    "direction", "downsize", "efficien", "execution", "exit unattractive", "expenditure reduction", "expense control", "expense management", "expense reduction", "focus", 
    "lower spend", "operating margin", "outsource", "overhead reduction", "production cost", "reduce cost", "reduce expenditure", "reduce expense", "reduce investment", 
    "reduce marketing", "reduce overhead", "reduce R&D", "reduce SG&A", "reduce spending", "reducing capital expenditure", "restruct", "restructuring cost", "roadmap", 
    "robotic process automation", "RPA", "spend control", "streamlin", "supply chain optimization", "utilization"],
    ["Operational improvements", "process enhancements", "efficiency gains", "operational streamlining", "productivity boosts", "performance upgrades"],
    ["Productivity metrics", "efficiency measures", "output indicators", "performance ratios", "productivity KPIs", "operational effectiveness"],
    ["Capacity utilization", "resource usage", "operational efficiency", "production capacity", "facility utilization", "asset efficiency"],
    ["Supply chain efficiency", "logistics performance", "supply network optimization", "procurement effectiveness", "distribution efficiency", "supply chain streamlining"],
    ["Inventory turnover", "stock rotation", "inventory efficiency", "stock velocity", "goods turnover rate", "inventory churn"],
    ["Capital Structure", "borrowings", "financial leverage", "liabilities", "indebtedness", "loan balances",  "bond rating",
    "capital market condition", "capital structure management", "capital structure optimization", "capital structure", "capitalization", "cost debt", "cost equity", "credit rating", "debt capacity", "debt covenants",
    "debt financing", "debt maturity", "debt restructur", "debt restructuring", "debt-to-equity ratio", "dividend policy", "equity financing", "financial flexibility", "financing plan", "financing strategy",
    "financing", "interest coverage ratio", "interest coverage", "interest expense", "interest rate risk", "interest rate sensitivity", "interest rate", "leverage", "long-term debt", "merge and acquisition", "share buyback",
    "share repurchase", "short-term debt", "stock buyback", "stock repurchase", "strategic acquisition", "strategic alliance", "strategic investment", "strategic merger", "strategic partnership", "WACC", 
    "weighted average cost of capital", "Debt-to-equity ratio", "leverage ratio", "capital structure", "financial leverage", "gearing ratio", "debt-to-capital ratio"],
    ["Share buyback plans", "stock repurchase program", "share repurchases", "buyback initiative", "stock retirement", "equity reduction"],
    ["Dividend policy", "payout policy", "shareholder distributions", "dividend strategy", "income distribution plan", "yield policy"],
    ["Capital expenditure plans", "Capex projections", "investment plans", "asset acquisition strategy", "infrastructure spending", "capital outlays"],
    ["Working capital management", "cash flow management", "liquidity management", "short-term asset management", "operational liquidity", "current asset efficiency"],
    ["Geographic expansion", "market entry", "territorial growth", "global reach expansion", "new market penetration", "regional diversification"],
    ["Merger and acquisition activities", "M&A strategy", "corporate takeovers", "business combinations", "acquisition plans", "consolidation efforts"],
    ["Market penetration strategies", "market share growth", "customer base expansion", "sales penetration tactics", "market intensification", "deepening market presence"],
    ["Diversification efforts", "business expansion", "new venture development", "portfolio diversification", "risk spreading", "new market entry"],
    ["Partnerships and collaborations", "strategic alliances", "joint ventures", "cooperative agreements", "business partnerships", "collaborative initiatives"],
    ["Sales pipeline", "sales funnel", "prospect pipeline", "revenue pipeline", "deal flow", "sales forecast"],
    ["Backlog or order book status", "unfilled orders", "work in progress", "future revenue", "committed sales", "order queue"],
    ["Customer acquisition costs", "CAC", "cost per customer", "marketing efficiency", "acquisition spend", "customer onboarding costs"],
    ["Lifetime value of customers", "LTV", "customer worth", "long-term customer value", "client profitability", "customer equity"],
    ["Marketing effectiveness", "ROI on marketing", "campaign performance", "marketing efficiency", "promotional impact", "advertising effectiveness"],
    ["Sales force productivity", "sales efficiency", "rep performance", "sales team effectiveness", "selling productivity", "revenue per salesperson"],
    ["Business unit breakdowns", "divisional performance", "segment analysis", "unit-level results", "departmental breakdown", "operational segment review"],
    ["Geographic segment performance", "regional results", "country-specific performance", "geographical breakdown", "territorial analysis", "location-based performance"],
    ["Product category performance", "product line results", "category-wise analysis", "product segment breakdown", "offering performance", "product mix analysis"],
    ["Customer segment analysis", "client group performance", "customer cohort analysis", "demographic performance", "user segment breakdown", "target market results"],
    ["Raw material costs", "input costs", "material expenses", "commodity prices", "resource costs", "supply expenses"],
    ["Labor costs", "workforce expenses", "employee costs", "payroll expenses", "human resource costs", "wage and salary expenses"],
    ["Overhead expenses", "indirect costs", "fixed costs", "operating expenses", "overhead burden", "non-direct expenses"],
    ["Cost of goods sold, COGS", "production costs", "direct costs", "manufacturing expenses", "cost of sales", "product costs"],
    ["Selling, general, and administrative expenses, SG&A", "operating expenses", "overhead costs", "non-production costs", "administrative burden", "commercial expenses"],
    ["Regulatory challenges", "compliance issues", "legal hurdles", "regulatory environment", "policy challenges", "governmental constraints"],
    ["Litigation updates", "legal proceedings", "lawsuit status", "court case developments", "legal dispute updates", "judicial proceedings"],
    ["Cybersecurity measures", "data protection", "information security", "cyber defense", "digital safeguards", "IT security"],
    ["Foreign exchange impact", "currency effects", "forex exposure", "exchange rate influence", "monetary conversion impact", "currency risk"],
    ["Interest rate sensitivity", "rate exposure", "interest risk", "borrowing cost sensitivity", "debt expense fluctuation", "interest rate impact"],
    ["Employee headcount", "workforce size", "staff numbers", "personnel count", "employee strength", "team size", "headcount management", "headcount optimization", 
    "headcount reduction", "headcount freeze", "hiring freeze", "labor cost", "labor efficiency", "labor management", "labor optimization", "labour cost", "labour efficiency", 
    "labour management", "labour optimization"],
    ["Employee turnover rate", "staff attrition", "churn rate", "workforce stability", "retention challenges", "employee departures"],
    ["Talent acquisition and retention strategies", "hiring initiatives", "employee retention programs", "workforce planning", "talent management", "recruitment strategies", 
    "staffing plans", "stock options"],
    ["Workforce diversity and inclusion", "diversity metrics", "inclusivity efforts", "equal opportunity initiatives", "workforce representation", "cultural diversity"],
    ["Employee engagement metrics", "staff satisfaction", "workforce morale", "employee loyalty", "job satisfaction", "team engagement"],
    ["Digital transformation initiatives", "digitalization efforts", "tech modernization", "digital evolution", "IT transformation", "technology upgrade", "agile methodology",
    "AI", "API", "AR", "artificial intelligence", "augmented reality", "automation", "automative", "autonomous vehicle", "AWS", "azure", "big data", "biometrics", "blockchain", "chatbot", "cloud computing", "contactless technology", "crowdsourcing", "cryptocurrency", "cybersecurity",
    "data analytics", "devops", "digital marketing", "digital transformation", "digital twin", "drones", "e-commerce", "edge computing", "fintech", "gaming", "geolocation technology", "google cloud", "health tech", "IAAS",
    "industry 4", "innovation", "innovative", "internet censorship", "internet of things", "IOT", "machine learning", "mobile payments", "mobile technology", "nanotechnology", "open source", "PaaS", "patent", "personalization", "predictive analytics",
    "prototyp", "quantum computing", "quantum encryption", "robot", "robotic", "robotics", "SaaS", "self-driving cars", "smart devices", "smart home technology", "social media", "technology", "user-generated content", "virtual reality", "VR", "wearable technology"],
    ["IT infrastructure investments", "tech spending", "system upgrades", "IT capex", "technology infrastructure", "computing resources"],
    ["E-commerce performance", "online sales", "digital revenue", "web store results", "internet retail performance", "online marketplace metrics"],
    ["Data analytics capabilities", "business intelligence", "data-driven insights", "analytics infrastructure", "information analysis", "predictive modeling"],
    ["Artificial intelligence and machine learning applications", "AI integration", "ML implementation", "intelligent automation", "cognitive computing", "smart algorithms", "adaptive system", "adversal attack",
    "adversal learning", "adversarial networks", "ai tools", "AI", "algorithmic decision-making", "ambient intelligence", "analytics", "artificial intelligence", "association rules", "attention mechanisms", "augmented reality", "automated reasoning", "automation",
    "autonomous systems", "bayesian networks", "BERT", "big data", "chatbot", "chatgpt", "clustering algorithms", "cognitive analytics", "cognitive computing", "computer vision", "content generat", "contextual understand", "conversational ai",
    "data science", "decision trees", "deep learning", "deepfake", "digital assistant", "digital transformation", "dimensionality reduction", "edge computing", "expert systems", "fake news detection", "fine-tuning", "generative models", "human-robot collaboration", "image recognition", "intelligent agents", "intelligent assistant",
    "intelligent automation", "intelligent edge", "intelligent interface", "intelligent optimization", "intelligent sensors", "internet of things", "keywords", "large language model", "LLM", "machine intelligence", "machine learning", "model selection", "natural language processing", "neural architecture search", "neural networks",
    "neural processing", "open ai", "openai", "pattern recognition", "pre-trained", "predictive modeling", "principal component analysis", "random forests", "reinforcement learning", "robotic process automation", "robotics", "self-learning systems", "semi-supervised learning", "smart algorithms", "smart systems", "supervised learning",
    "support vector machines", "text classification", "text completion", "text generation", "text summarization", "transfer learning", "transformer", "unsupervised learning", "virtual reality", "voice recognition"],
    ["automation", "robotics", "RPA", "robotic process automation", "artificial intelligence", "machine learning", "process automat", "digital transformation", "digitalization", "digit process automat", "autonomous driv", "autonomous vehicle", "autonomous system"],
    ["Environmental initiatives", "eco-friendly programs", "green initiatives", "sustainability efforts", "environmental stewardship", "ecological projects", "biodiversity",
    "carbon emissions", "carbon footprint", "carbon neutrality", "carbon offset", "carbon reduction", "carbon tax", "carbon trading", "clean energy", "climate change", "climate risk", 
    "eco-friendly", "energy efficiency", "environmental impact", "environmental sustainability", "ESG", "global warning", "green energy", "green product", "green technology", "greenhouse gas",
    "greenhouse", "low-carbon", "natural resource", "ozone layer", "pollution control", "pollution", "sustainability", "sustainable"],
    ["Social responsibility programs", "community initiatives", "social impact", "corporate citizenship", "philanthropic efforts", "societal contributions", "community",
    "corporate culture", "culture diversity", "gender diversity", "gender equality", "gender inequality", "income inequality", "job creation", 
    "LGBT", "social impact", "social responsibility", "social sustainab", "sustainable development", "workforce diversity"],
    ["Governance practices", "corporate governance", "board practices", "management oversight", "ethical leadership", "shareholder rights", "bonus plans",
    "conservatism", "conservative", "corporate governance", "corporate culture", "discipline", "effective", "efficien", "employee engagement", "employee productivity", 
    "employee retention", "employee satisfaction", "employee stock option", "employee turnover", "equity compensation", "executive compensation", "gender inequality", 
    "incentive compensation", "incentive plan", "income inequality", "insider", "optimal", "organizational culture", "organizational structure", "performance evaluation", 
    "performance-base compensation", "predictability", "predictable", "race equality", "shareholder activis", "stock option plan", "stock option program", "stock option", 
    "succession planning", "talent acquisition", "talent development", "talent management", "talent retention", "transparen", "turnover", "workforce diversity"],
    ["Carbon footprint reduction efforts", "emissions reduction", "climate impact mitigation", "greenhouse gas reduction", "carbon neutrality efforts", "environmental impact reduction"],
    ["Sustainable sourcing", "ethical procurement", "responsible sourcing", "supply chain sustainability", "green purchasing", "eco-friendly suppliers"],
    ["Patent portfolio", "IP assets", "patent holdings", "invention rights", "proprietary technology", "patented innovations"],
    ["Trademark developments", "brand protection", "trademark portfolio", "intellectual property rights", "brand assets", "trademark strategy"],
    ["Licensing agreements", "IP licensing", "technology transfer", "patent licensing", "trademark licensing", "copyright agreements"],
    ["IP litigation", "patent disputes", "trademark infringement", "copyright cases", "intellectual property lawsuits", "IP legal battles"],
    ["Corporate innovation", "innovation", "r&d", "research development", "patent", "breakthrough technologies"],
    ["Customer satisfaction scores", "client happiness index", "satisfaction ratings", "customer feedback metrics", "service quality scores", "consumer contentment measures"],
    ["Churn rate", "customer attrition", "client loss rate", "turnover rate", "defection rate", "customer departure frequency"],
    ["Average revenue per user (ARPU)", "per-customer revenue", "user monetization", "client value", "revenue intensity", "customer yield"],
    ["Customer lifetime value (CLV)", "lifetime customer worth", "long-term client value", "customer profitability", "total customer value", "client lifetime worth"],
    ["Pricing power", "price elasticity", "pricing leverage", "value capture ability", "price setting ability", "margin potential"],
    ["Discount policies", "price reduction strategies", "promotional pricing", "markdown strategies", "price concessions", "rebate programs"],
    ["Dynamic pricing initiatives", "real-time pricing", "adaptive pricing", "flexible pricing", "demand-based pricing", "price optimization"],
    ["Bundle pricing strategies", "package deals", "product bundling", "combined offering prices", "multi-product discounts", "solution pricing"],
    ["Organizational changes", "structural shifts", "corporate reorganization", "business restructuring", "organizational redesign", "company realignment"],
    ["Executive leadership transitions", "C-suite changes", "management shuffle", "leadership succession", "executive appointments", "senior management changes"],
    ["Board composition", "director lineup", "board structure", "governance makeup", "board demographics", "directorship changes"],
    ["Subsidiary performance", "division results", "affiliate performance", "business unit outcomes", "subsidiary contributions", "controlled entity results"],
    ["Sector-specific KPIs", "industry benchmarks", "vertical-specific metrics", "sector performance indicators", "industry standards", "niche measurements"],
    ["Regulatory compliance metrics", "compliance scores", "regulatory adherence measures", "conformity indicators", "rule-following metrics", "policy compliance rates"],
    ["Industry benchmarking", "peer comparison", "competitive benchmarking", "industry standards comparison", "market positioning", "sector performance ranking", "bill",
    "commodity price", "competitive advantage", "competitive environment", "competitive landscape", "competitive position", "competitive risk", "competitive threat", "consumer growth", "consumer trend", "consumption growth", "consumption trend",
    "consumption", "consolidation", "consumer behavior", "consumer confidence", "consumer preference", "consumer sentiment", "consumer spending", "consumer trend", "govern subsidy", "government bill", "government contract",
    "government subsidy", "heat up", "heats up", "industry contraction", "industry expansion", "industry forecast", "industry growth trend", "industry growth", "industry outlook", "industry projection", "industry trend",
    "innovation", "innovative approach", "innovative business", "innovative manner", "innovative product", "innovative service", "innovative solution", "innovative technology", "market demand", "market supply", "market trend",
    "PPI", "price competition", "price trend", "price war", "price", "producer price index", "product innovation", "R&D", "subsidy war", "supply chain", "supply-chain", "sustainability", "technology development", "technology innovation", "technology trends"],
    ["political risk", "risk", "uncertainty", "volatility", "risk factors", "risk management", "a government",
    "a political", "a president", "a yes", "african americans", "american political", "and congress", "and party", "and political", "and senate", "and social", "any state",
    "argued that", "care reform", "civil service", "civil war", "clause of", "congress the", "congress to", "congress", "constitution and", "court has", "due process", "economic policy",
    "elected officials", "executive branch", "executive privilege", "federal bureaucracy", "federal courts", "federal reserve", "first amendment", "for governor", "free market", "general election", 
    "geopolitical attention", "geopolitical change", "geopolitical policy", "geopolitical reform", "geopolitical risk", "geopolitical uncertainty", "government and", "government contract", "government in", 
    "government is 1", "government policy", "government the", "governor and", "governor in", "groups and", "in congress", "in government", "interest group", "interest groups", "islamic state", "judicial review", 
    "law suit", "legal risk", "legal uncertainty", "limits on", "most americans", "national and", "national government", "national security", "NATO", "of citizens", "of civil", "of government", "of politics", "of religion",
    "of representatives", "of social", "of speech", "office of", "other nations", "passed by", "public policy", "political risk", "political uncertainty", "policy goals", "policy is", "policy", "political parties",
    "political party", "political process", "political system", "politics and", "politics is", "politics of", "powers and", "president has", "president obama", "proposed by", "public opinion", "regulatory change", "regulatory environment",
    "regulatory policy", "regulatory reform", "regulatory risk", "regulatory uncertainty", "ruled that", "sanction", "senate and", "shall have", "shall not", "social policy", "state the", "states or", "struck down",
    "support for", "tax risk", "tax tariff", "tea party", "that congress", "the bureaucracy", "the campaign", "the candidates", "the civil", "the congress", "the constitution", "the constitutional", "the democratic", "the electoral", 
    "the epa", "the faa", "the gop", "the governments", "the house", "the islamic", "the legislative", "the legislature", "the nation", "the partys", "the political", "the politics", "the presidency", "the presidential", "the republican", 
    "the south 1", "the states", "the taliban", "the va", "this election", "to enact", "vote for", "war risk", "war uncertainty", "war", "white house", "yes vote"],
    ["Macro Economic Risk", "macro trends", "economic influences", "market conditions", "financial environment", "economic climate", "economic risk",
    "economic uncertainty", "economic change", "economic reform", "economic policy", "economic environment", "economic slowdown", "economic recession", "economic contraction", "economic expansion", "economic growth", "economic recovery",
    "headwind", "financial risk", "market risk", "interest rate risk", "downturn", "slow down", "systematic risk", "bank run", "crisis", "macroeconomic", "market crush", "inflation", "deflation", "global recession", "fiscal policy", "monetary policy", "unemployment"],
    ["liquidity risk", " liquidity risk", " liquidity uncertainty", "financial tightness", "financial constraint", "financial stress", "financial strain", "financial pressure", "financial difficulty", "solvency", "default", "capital adequacy", "credit crunch",
    "credit rating", "credit risk", "counterparty risk", "stress test", "repo market", "interest coverage", "quick ratio", "current ratio", "collateral", "counterparty risk", "cash reserve", "bankruptcy"],
    ["sovereign risk", "american", "british", "china", "chinese", "european", "great britain", "iran", "israel", "north korea", "russia", "sovereign change", "sovereign credit rating", "sovereign credit risk",
    "sovereign debt", "sovereign default", "sovereign downgrade", "sovereign environment", "sovereign policy", "sovereign reform", "sovereign risk", "sovereign uncertainty", "the eu", "the uk", "the us", 
    "united kingdom", "united states"],
    ["credit risk", "credit risk", "credit uncertainty", "credit change", "credit reform", "credit policy", "credit environment", "credit tightness", "credit constraint", "credit stress", "credit strain", "credit pressure", "credit difficulty", "credit downgrade",
    "credit rating", "credit exposure", "credit default", "credit default swap", "credit deterioration", "credit evaluation", "credit limit", "credit loss", "credit monitoring", "credit portfolio", "credit quality", "credit review", "credit scoring", "credit spread",
    "credit surveillance", "creditworthiness", "counterparty risk", "default risk", "loan loss", "loan quality", "non-performing loan", "subprime risk"],
    ["operational risk",  "brand reputation", "business continuity", "business operation", "contingenc", "control environment", "crisis management", "customer satisfaction", "cybersecurity", "CEO transition", "CFO transition", "fraud", "health and safety", "incident management", "information risk",
    "insurance", "internal control", "logistic", "loss provision", "manufacturing efficiency", "network downtime", "operational change", "operational constraint", "operational difficulty", "operational downgrade", "operational environment", "operational policy", "operational pressure", "operational rating", "operational reform",
    "operational risk", "operational strain", "operational stress", "operational tightness", "operational uncertainty", "process efficiency", "process improvement", "process optimization", "product quality", "production safety", "regulatory compliance", "resilience", "supply chain constraint", "supply chain disorder",
    "supply chain disruption", "supply chain management", "supply chain risk", "supply chain shortage"],
    ["legal risk", "legal risk", "legal uncertainty", "legal change", "legal reform", "legal policy", "legal environment", "corruption", "antitrust", "antitrust risk", "anti-monopoly", "monopoly", "monopoly risk", "antitrust law", "compliance", "regulator",
    "litigation", "lawsuit", "settlement", "legal proceed", "legal action", "legal dispute", "legal claim", "legal case", "legal process", "judicial proceed"],
    ["climate risk", "blizzard", "climate disaster", "climate risk", "drought", "dust storm", "earthquake", "environmental hazard", "environmental risk", "fire-storm", 
    "flood", "flooded", "flooding", "forest fire", "freeze", "frigid", "frost", "frostbite", "frostbitten", "frozen", "global warming", "heatstroke", "heatwave", "hotness", "hurricane", "hyperthermia",
    "hypothermia", "landslide", "natural disaster", "precipitation", "rainfall", "severe weather", "shivering", "snowfall", "snowstorm", "solar-flare", "storm", "thunder", "thunderstorm",
    "tornado", "torrid", "torridity", "tropical cyclone", "tsunami", "typhoon", "volcanic", "volcano", "weather condition", "wildfire", "windy", "winter storm", "wintriness"],
    ["cybersecurity risk", "cybersecurity", "cyberattack", "cybercrime", "cybersecurity risk", "cybersecurity threat", "cybersecurity vulnerability", "cybersecurity exposure", "cybersecurity incident", "cybersecurity breach", "cybersecurity attack", "computer attack", "computer breach", "computer break-in",
    "computer joint security", "computer security", "computer system security", "computer virus", "confidential data", "confidentiality of data", "corruption of data", "crimeware", "cyber attack", "cyber fraud", "cyber incident", "cyber insurance", "cyber risk",
    "cyber security", "cyber terrorist", "cyber threat", "cyber-attack", "cyber-fraud", "cyber-incident", "cyberinsurance", "cyber-risk", "cyber-security", "cyber-terrorist", "cyber-threat", "cyber-based attack", "cyberfraud",
    "cyberincident", "cyberrisk", "cyberterrorist", "cyberthreat", "cyberbullying", "ddos", "data-breach", "data breach", "data confidentiality", "data corruption", "data theft", "denial of service", "dark web", "encryption", "espionage", "hack", "hacker", "hacked",
    "hacking", "hacktivism", "phishing", "information security", "information technology attack", "information technology security", "infosec", "intrusion", "keylogger", "keystroke logging", "malware", "network break-in", "network joint security", "network security", "ransomware", "security breach",
    "security expenditure", "security incident", "security management", "security monitoring", "social engineering", "system security", "unauthorized access"],
    ["Urgent and Timely action", "urgent", "immediate", "quick", "fast", "fastest", "rapid", "rapidly", "speed", "speed up", "sped up", "catch up", "speeding up", "accelerate", "priority", "swift",
    "prompt", "promptly", "expedite", "expeditious", "time-critical", "time-sensitive", "red code", "short term", "short-term", "short run", "emphasize",
    "highlight", "highlighted", "significant", "important", "critical", "key", "must", "have to", "has to", "essential", "vital", "imperative", "mandatory",
    "prominent", "pronounced", "marked", "eminent", "noticeable", "striking", "conspicuous", "remarkable", "salient", "notable", "distinctive", "pivotal", "crucial", "decisive",
    "indispensable", "integral", "necessity", "requisite", "urgent", "compelling", "forceful", "persuasive", "strong", "assertive", "definitive", "unmistakable", "expressive",
    "intense", "resounding", "strident", "sure", "confident", "insistent", "positive", "surprise", "dog and pony show", "misunderstanding"]
]