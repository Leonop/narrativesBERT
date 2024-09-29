import os
from typing import Dict, List
import pandas as pd


# Directory locations

PROJECT_DIR = os.getcwd()
data_folder = os.path.join(PROJECT_DIR, "data")
model_folder = os.path.join(PROJECT_DIR, "model")
output_folder = os.path.join(PROJECT_DIR, "output")
output_fig_folder = os.path.join(output_folder, "fig")
data_filename = 'earnings_calls_20231017.csv'
stop_list = pd.read_csv(os.path.join(data_folder, "stoplist.csv"))['stopwords'].tolist()
MODEL_SCORES = os.path.join(output_folder, "model_scores.txt")
DATE_COLUMN = "transcriptcreationdate_utc"
num_topic_to_plot = 20 # top_N topics to plot

TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
NROWS = 20000000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_FILTER = 2025 # filter the data based on the year
# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
BATCH_SIZE = 1000

# create a list of parameters to search over using GridSearchCV
N_NEIGHBORS = [10] 
N_COMPONENTS = [5]
MIN_DIST = [0.5]
MIN_SAMPLES = [10]
MIN_CLUSTER_SIZE = [15] # in HDBSCAN: Minimum size of clusters
N_TOPICS = [110]
N_TOP_WORDS = [20]
METRIC = ['cosine']
EMBEDDING_MODELS = ['paraphrase-MiniLM-L6-v2'] #'all-MiniLM-L6-v2'

# Parameters for TFIDF Vectorizer
MAX_DF = [0.9] # float Max document frequency for words to include.
MIN_DF = [2] # Min document frequency for words to include.
# SAVE RESULTS 
SAVE_RESULTS_COLS = ["params", "score", "probability"]
SEED_WORDS: Dict[str, List[str]] = {
    # Financial Performance
    "revenue": ["revenue", "income_statement", "sales", "top-line", "total_revenue"],
    "growth": ["growth", "expansion", "increase", "rise", "escalation", "year_over_year", "yoy"],
    "profit": ["profit", "net_income", "bottom-line", "earnings", "net_profit"],
    "cost": ["cost", "expenses", "expenditure", "overhead", "costs"],
    "cash": ["cash", "cash_flow", "liquidity", "cash_position", "cash_balance"],
    "debt": ["debt", "liabilities", "borrowing", "indebtedness", "debt_burden"],
    "equity": ["equity", "shareholders", "stockholders", "ownership", "equity_holders"],
    "investment": ["investment", "investing", "capital_expenditure", "capex", "investment_spending"],
    "dividend": ["dividend", "dividend_payment", "dividend_yield", "dividend_payout", "dividend_policy"],
    "risk": ["risk", "uncertainty", "volatility", "risk_factors", "risk_management"],
    "financial_position": ["financial_position", "balance_sheet", "financial_health", "financial_stability", "financial_standing"],
    "liquidity": ["liquidity", "liquid_assets", "current_assets", "quick_ratio", "current_ratio"],
    "gross_profit_margin": ["gross_margin", "profit_ratio", "markup_percentage", "gross_profit_rate", "sales_margin"],
    "Operating_margin": ["operating_profit_margin", "EBIT_margin", "operating_income_margin", "profit_margin", "operational_efficiency"],
    "free_cash_flow": ["cash_balance", "cash_burn_rate", "cash_convertion_cycle", "cash_flow", "cash_generation", "cash_position"],
    "return_on_equity": ["return_on_equity", "equity_returns", "shareholder_return", "net_income_to_equity", "equity_performance", "profitability_ratio"],
    "return_on_assets": ["return_on_assets", "asset_returns", "asset_performance", "net_income_to_assets", "asset_profitability"],
    "return_on_investment": ["return_on_investment", "investment_returns", "investment_performance", "net_income_to_investment", "investment_profitability"],
    "productivity": ["automation", "capacity_utilization", "cost_cutting", "cost_efficiency", "cost_reduction", "cost_saving", "digital_transformation", "efficiency", "labor_cost", "labor_efficiency", "labor_layoff", "labor_productivity", "labour_cost", "labour_efficiency", "labour_layoff", "labour_productivity", "laid_off", "lay_off"],
    "asset_impairment": ["allowance", "write-off", "impairment_charge", "asset_impairment", "goodwill_impairment"],
    "Coprporate_tax": ["effective_tax_rate", "tax_liabilities", "tax_planning", "tax_credits", "deferred_taxes"],
    # Guidance and Outlook
    "next_quarter_guidance": ["short-term_forecast", "upcoming_quarter_outlook", "near-term_projections", "quarterly_expectations", "forward_guidance"],
    "full_year_outlook": ["annual_forecast", "yearly_projection", "long-term_guidance", "fiscal_year_outlook", "12-month_projection"],
    "Long_term_financial_targets": ["multi-year_goals", "strategic_financial_objectives", "extended_financial_outlook", "long-range_targets", "future_financial_aims"],
    "Industry_forecast": ["industry_forecast", "sector_outlook", "market_projections", "industry_trends", "vertical_predictions", "sector_expectations"],
    "Economic_forecast": ["economic_forecast", "macroeconomic_outlook", "economic_projections", "economic_trends", "economic_expectations"],
    # Market Position
    "market_share": ["market_share", "market_dominance", "market_leadership", "market_position", "business_footprint"],
    "competitive_landscape": ["competitive_risk", "competitive_environment", "industry_rivalry", "market_competition", "competitor_analysis", "competitive_environment", "industry_dynamics"],
    "Brand_strength": ["brand_strength", "brand_power", "brand_health", "brand_recognition", "brand_equity", "brand_value"],
    "Customer_acquisition": ["new_customer_growth", "client_onboarding", "customer_wins", "new_business_generation", "expanding_customer_base"],
    "Customer_retention_rates": ["client_loyalty", "churn_rate", "customer_stickiness", "repeat_business", "customer_longevity"],
    "Net_Promoter_Score_NPS": ["customer_satisfaction_index", "loyalty_metric", "referral_likelihood", "customer_advocacy", "satisfaction_score"],
    # Product and Service
    "New_product_launches": ["product_releases", "new_offerings", "product_introductions", "market_debuts", "new_solutions"],
    "Product_mix_changes": ["product_portfolio_shifts", "offering_diversification", "product_line_adjustments", "sales_mix"],
    "Service_quality": ["customer_satisfaction_measures", "service_performance_indicators", "quality_assurance_metrics", "service_level_achievements", "customer_experience_scores"],
    "research_and_development": ["R&D_spending", "innovation_funding", "product_development_costs", "research_expenditure", "technology_investments"],
    "innovation_pipeline": ["future_products", "development_roadmap", "upcoming_innovations", "product_incubation", "new_concept_funnel"],
    "product_roadmap": ["development_timeline", "product_strategy", "future_releases", "product_evolution_plan", "feature_roadmap"],
    # Operational Efficiency:
    "Cost_cutting_initiatives": ["expense_reduction", "efficiency_programs", "cost_optimization", "savings_measures", "budget_trimming"],
    "Operational_improvements": ["process_enhancements", "efficiency_gains", "operational_streamlining", "productivity_boosts", "performance_upgrades"],
    "Productivity_metrics": ["efficiency_measures", "output_indicators", "performance_ratios", "productivity_KPIs", "operational_effectiveness"],
    "Capacity_utilization": ["resource_usage", "operational_efficiency", "production_capacity", "facility_utilization", "asset_efficiency"],
    "Supply_chain_efficiency": ["logistics_performance", "supply_network_optimization", "procurement_effectiveness", "distribution_efficiency", "supply_chain_streamlining"],
    "Inventory_turnover": ["stock_rotation", "inventory_efficiency", "stock_velocity", "goods_turnover_rate", "inventory_churn"],
    # Capital Structure and Allocation:
    "Debt_levels": ["borrowings", "financial_leverage", "liabilities", "indebtedness", "loan_balances"],
    "Debt_to_equity_ratio": ["leverage_ratio", "capital_structure", "financial_leverage", "gearing_ratio", "debt_to_capital_ratio"],
    "Share_buyback_plans": ["stock_repurchase_program", "share_repurchases", "buyback_initiative", "stock_retirement", "equity_reduction"],
    "Dividend_policy": ["payout_policy", "shareholder_distributions", "dividend_strategy", "income_distribution_plan", "yield_policy"],
    "Capital_expenditure_plans": ["CapEx_projections", "investment_plans", "asset_acquisition_strategy", "infrastructure_spending", "capital_outlays"],
    "Working_capital_management": ["cash_flow_management", "liquidity_management", "short-term_asset_management", "operational_liquidity", "current_asset_efficiency"],
    # Growth Strategies:
    "Geographic_expansion": ["market_entry", "territorial_growth", "global_reach_expansion", "new_market_penetration", "regional_diversification"],
    "Merger_and_acquisition_activities": ["M&A_strategy", "corporate_takeovers", "business_combinations", "acquisition_plans", "consolidation_efforts"],
    "Market_penetration_strategies": ["market_share_growth", "customer_base_expansion", "sales_penetration_tactics", "market_intensification", "deepening_market_presence"],
    "Diversification_efforts": ["business_expansion", "new_venture_development", "portfolio_diversification", "risk_spreading", "new_market_entry"],
    "Partnerships_and_collaborations": ["strategic_alliances", "joint_ventures", "cooperative_agreements", "business_partnerships", "collaborative_initiatives"],
    # Sales and Marketing:
    "Sales_pipeline": ["sales_funnel", "prospect_pipeline", "revenue_pipeline", "deal_flow", "sales_forecast"],
    "Backlog_or_order_book_status": ["unfilled_orders", "work_in_progress", "future_revenue", "committed_sales", "order_queue"],
    "Customer_acquisition_costs": ["CAC", "cost_per_customer", "marketing_efficiency", "acquisition_spend", "customer_onboarding_costs"],
    "Lifetime_value_of_customers": ["LTV", "customer_worth", "long-term_customer_value", "client_profitability", "customer_equity"],
    "Marketing_effectiveness": ["ROI_on_marketing", "campaign_performance", "marketing_efficiency", "promotional_impact", "advertising_effectiveness"],
    "Sales_force_productivity": ["sales_efficiency", "rep_performance", "sales_team_effectiveness", "selling_productivity", "revenue_per_salesperson"],
    # Segment Performance:
    "Business_unit_breakdowns": ["divisional_performance", "segment_analysis", "unit-level_results", "departmental_breakdown", "operational_segment_review"],
    "Geographic_segment_performance": ["regional_results", "country-specific_performance", "geographical_breakdown", "territorial_analysis", "location-based_performance"],
    "Product_category_performance": ["product_line_results", "category-wise_analysis", "product_segment_breakdown", "offering_performance", "product_mix_analysis"],
    "Customer_segment_analysis": ["client_group_performance", "customer_cohort_analysis", "demographic_performance", "user_segment_breakdown", "target_market_results"],
    # Cost Management:
    "Raw_material_costs": ["input_costs", "material_expenses", "commodity_prices", "resource_costs", "supply_expenses"],
    "Labor_costs": ["workforce", "workforce_expenses", "employee", "employee_costs","payroll", "payroll_expenses", "human_resource", "wage_and_salary"],
    "Overhead_expenses": ["indirect_costs", "fixed_costs", "operating_expenses", "overhead_burden", "non-direct_expenses"],
    "Cost_of_goods_sold_COGS": ["production_costs", "direct_costs", "manufacturing_expenses", "cost_of_sales", "product_costs"],
    "Selling_general_and_administrative_expenses_SG&A": ["operating_expenses", "overhead_costs", "non-production_costs", "administrative_burden", "commercial_expenses"],
    # Risk Management:
    "Regulatory_challenges": ["compliance_issues", "legal_hurdles", "regulatory_environment", "policy_challenges", "governmental_constraints"],
    "Litigation_updates": ["legal_proceedings", "lawsuit_status", "court_case_developments", "legal_dispute_updates", "judicial_proceedings"],
    "Cybersecurity_measures": ["data_protection", "information_security", "cyber_defense", "digital_safeguards", "IT_security"],
    "Foreign_exchange_impact": ["currency_effects", "forex_exposure", "exchange_rate_influence", "monetary_conversion_impact", "currency_risk"],
    "Interest_rate_sensitivity": ["rate_exposure", "interest_risk", "borrowing_cost_sensitivity", "debt_expense_fluctuation", "interest_rate_impact"],
    # Human Capital:
    "Employee_headcount": ["workforce_size", "staff_numbers", "personnel_count", "employee_strength", "team_size"],
    "Employee_turnover_rate": ["staff_attrition", "churn_rate", "workforce_stability", "retention_challenges", "employee_departures"],
    "Talent_acquisition_and_retention_strategies": ["hiring_initiatives", "employee_retention_programs", "workforce_planning", "talent_management", "recruitment_strategies"],
    "Workforce_diversity_and_inclusion": ["diversity_metrics", "inclusivity_efforts", "equal_opportunity_initiatives", "workforce_representation", "cultural_diversity"],
    "Employee_engagement_metrics": ["staff_satisfaction", "workforce_morale", "employee_loyalty", "job_satisfaction", "team_engagement"],
    # Technology and Digital:
    "Digital_transformation_initiatives": ["digitalization_efforts", "tech_modernization", "digital_evolution", "IT_transformation", "technology_upgrade"],
    "IT_infrastructure_investments": ["tech_spending", "system_upgrades", "IT_capex", "technology_infrastructure", "computing_resources"],
    "E-commerce_performance": ["online_sales", "digital_revenue", "web_store_results", "internet_retail_performance", "online_marketplace_metrics"],
    "Data_analytics_capabilities": ["business_intelligence", "data-driven_insights", "analytics_infrastructure", "information_analysis", "predictive_modeling"],
    "Artificial_intelligence_and_machine_learning_applications": ["AI_integration", "ML_implementation", "intelligent_automation", "cognitive_computing", "smart_algorithms"],
    # Sustainability and ESG:
    "Environmental_initiatives": ["eco-friendly_programs", "green_initiatives", "sustainability_efforts", "environmental_stewardship", "ecological_projects"],
    "Social_responsibility_programs": ["community_initiatives", "social_impact", "corporate_citizenship", "philanthropic_efforts", "societal_contributions"],
    "Governance_practices": ["corporate_governance", "board_practices", "management_oversight", "ethical_leadership", "shareholder_rights"],
    "Carbon_footprint_reduction_efforts": ["emissions_reduction", "climate_impact_mitigation", "greenhouse_gas_reduction", "carbon_neutrality_efforts", "environmental_impact_reduction"],
    "Sustainable_sourcing": ["ethical_procurement", "responsible_sourcing", "supply_chain_sustainability", "green_purchasing", "eco-friendly_suppliers"],
    # Intellectual Property:
    "Patent_portfolio": ["patent", "IP_assets", "patent_holdings", "invention_rights", "proprietary_technology", "patented_innovations"],
    "Trademark_developments": ["Trademark", "brand_protection", "trademark_portfolio", "intellectual_property_rights", "brand_assets", "trademark_strategy"],
    "Licensing_agreements": ["IP_licensing", "technology_transfer", "patent_licensing", "trademark_licensing", "copyright_agreements"],
    "IP_litigation": ["patent_disputes", "trademark_infringement", "copyright_cases", "intellectual_property_lawsuits", "IP_legal_battles"],
    "Corporate_innovation": ["innovation", "r&d", "research_development", "patent", "breakthrough_technologies"],
    # Customer-centric Metrics:
    "Customer_satisfaction_scores": ["client_happiness_index", "satisfaction_ratings", "customer_feedback_metrics", "service_quality_scores", "consumer_contentment_measures"],
    "Churn_rate": ["customer_attrition", "client_loss_rate", "turnover_rate", "defection_rate", "customer_departure_frequency"],
    "Average_revenue_per_user": ["per-customer_revenue", "user_monetization", "client_value", "revenue_intensity", "customer_yield", "ARPU"],
    "Customer_lifetime_value": ["lifetime_customer_worth", "long-term_customer_value", "customer_profitability", "total_customer_value", "client_lifetime_worth"],
    "Marketing_effectiveness": ["ROI_on_marketing", "campaign_performance", "marketing_efficiency", "promotional_impact", "advertising_effectiveness"],
    "Sales_force_productivity": ["sales_efficiency", "rep_performance", "sales_team_effectiveness", "selling_productivity", "revenue_per_salesperson"],
    # Pricing Strategies:
    "Pricing_power": ["price_elasticity", "pricing_leverage", "value_capture_ability", "price_setting_ability", "margin_potential"],
    "Discount_policies": ["price_reduction_strategies", "promotional_pricing", "markdown_strategies", "price_concessions", "rebate_programs"],
    "Dynamic_pricing_initiatives": ["real-time_pricing", "adaptive_pricing", "flexible_pricing", "demand-based_pricing", "price_optimization"],
    "Bundle_pricing_strategies": ["package_deals", "product_bundling", "combined_offering_prices", "multi-product_discounts", "solution_pricing"],
    # Corporate Structure:
    "Organizational_changes": ["structural_shifts", "corporate_reorganization", "business_restructuring", "organizational_redesign", "company_realignment"],
    "Executive_leadership_transitions": ["C-suite_changes", "management_shuffle", "leadership_succession", "executive_appointments", "senior_management_changes"],
    "Board_composition": ["director_lineup", "board_structure", "governance_makeup", "board_demographics", "directorship_changes"],
    "Subsidiary_performance": ["division_results", "affiliate_performance", "business_unit_outcomes", "subsidiary_contributions", "controlled_entity_results"],
    # Industry-specific Metrics:
    "Sector_specific_KPIs": ["industry_benchmarks", "vertical-specific_metrics", "sector_performance_indicators", "industry_standards", "niche_measurements"],
    "Regulatory_compliance_metrics": ["compliance_scores", "regulatory_adherence_measures", "conformity_indicators", "rule-following_metrics", "policy_compliance_rates"],
    "Industry_benchmarking": ["peer_comparison", "competitive_benchmarking", "industry_standards_comparison", "market_positioning", "sector_performance_ranking"],
    # Macroeconomic Factors:
    "Economic_indicators_affecting_the_business": ["macro_trends", "economic_influences", "market_conditions", "financial_environment", "economic_climate"]
}
