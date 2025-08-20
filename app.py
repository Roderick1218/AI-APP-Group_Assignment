# app.py - AI-Powered Travel Policy Advisor (Google Gemini + Vector DB)
"""
=======================================================================
🏢 Enterprise Travel Policy Management System - Main Application
=======================================================================

Core Function Modules (5 Requirements Implementation):
1. Policy Query Interface - ChatGPT-style intelligent policy Q&A
2. Request Validation - AI intelligent validation of travel request compliance
3. Approval Workflow - Automated approval routing based on cost and department
4. Travel Planning - Intelligent planning integrated with external travel information sources
5. Calendar Integration - Automatic generation of travel schedules

Technical Architecture:
• Frontend: Streamlit Web Interface
• AI Engine: Google Gemini + ChromaDB Vector Database  
• Database: PostgreSQL/Prisma with SQLAlchemy ORM
• Calendar: ICS format automatic generation and download

Version: v2.0 - Enterprise Complete Solution
=======================================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine, text

# --- AI AND VECTOR DB IMPORTS ---
from ics import Calendar, Event
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb
import google.generativeai as genai
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
import requests
import json

# --- CONFIGURATION ---
load_dotenv()

# Google API Key (optional). If missing, app falls back to local search.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# PostgreSQL Database Configuration
USE_ONLINE_DATABASE = os.getenv("USE_ONLINE_DATABASE", "true").lower() == "true"
DATABASE_ONLINE_URL = os.getenv("DATABASE_ONLINE")

if USE_ONLINE_DATABASE and DATABASE_ONLINE_URL:
    DATABASE_URL = DATABASE_ONLINE_URL.replace("postgres://", "postgresql://", 1)
    print("Using online Prisma database...")
else:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "travel_db")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "0183813235")
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print("Using local PostgreSQL database...")

def get_database_url():
    if USE_ONLINE_DATABASE and DATABASE_ONLINE_URL:
        try:
            conn = psycopg2.connect(DATABASE_ONLINE_URL)
            conn.close()
            print("Successfully connected to online Prisma database")
            return DATABASE_ONLINE_URL.replace("postgres://", "postgresql://", 1)
        except Exception as e:
            print(f"Online connection failed: {e}")
            print("Falling back to local PostgreSQL...")
    
    try:
        local_url = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '0183813235')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'travel_db')}"
        conn = psycopg2.connect(local_url.replace("postgresql://", "postgres://", 1))
        conn.close()
        print("Successfully connected to local PostgreSQL database")
        return local_url
    except Exception as e:
        print(f"Database connection failed: {e}")
        # Return a default that will cause a clear error message
        return "postgresql://postgres:0183813235@localhost:5432/travel_db"

DATABASE_URL = get_database_url()
CALENDAR_FILE = "trip_event.ics"

# --- GET POLICIES FROM DATABASE ---
@st.cache_data
def load_policies():
    """Load policies from database and return as serializable dictionaries with enhanced content"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            policies_result = connection.execute(text("SELECT rule_name, description FROM travel_policies"))
            policies = policies_result.fetchall()
        
        policy_list = []
        for name, desc in policies:
            # Create enhanced content for better AI understanding
            enhanced_content = f"""
Policy Name: {name}
Description: {desc}
Full Policy Text: {name} - {desc}

Context Keywords: {' '.join(name.lower().split())} {' '.join(desc.lower().split())}
Policy Category: {'accommodation' if 'hotel' in name.lower() or 'accommodation' in name.lower() 
                 else 'transportation' if 'flight' in name.lower() or 'transport' in name.lower()
                 else 'expense' if 'meal' in name.lower() or 'allowance' in name.lower() or 'cost' in name.lower()
                 else 'approval' if 'approval' in name.lower() or 'manager' in name.lower()
                 else 'general'}
"""
            
            policy_dict = {
                'rule_name': name,
                'description': desc,
                'content': enhanced_content,
                'searchable_text': f"{name} {desc}".lower()
            }
            policy_list.append(policy_dict)
        
        return policy_list
    
    except Exception as e:
        print(f"Error loading policies: {e}")
        st.error(f"Database error: {e}")
        # Return some sample policies if database fails
        return [
            {
                'rule_name': 'Hotel Accommodation Standards',
                'description': 'Business travelers may book hotels up to $300/night in major cities, $200/night in other locations',
                'content': 'Hotel Accommodation Standards - Business travelers may book hotels up to $300/night in major cities, $200/night in other locations',
                'searchable_text': 'hotel accommodation standards business travelers book hotels $300 night major cities $200 locations'
            },
            {
                'rule_name': 'Flight Class Policy', 
                'description': 'Economy class required for domestic flights. Business class allowed for international flights over 6 hours',
                'content': 'Flight Class Policy - Economy class required for domestic flights. Business class allowed for international flights over 6 hours',
                'searchable_text': 'flight class policy economy domestic business international flights 6 hours'
            }
        ]

# --- AI CONFIGURATION AND VECTOR DATABASE ---
def initialize_ai_components():
    """Initialize Gemini AI and Chroma vector database"""
    try:
        # Initialize Gemini model when API key is available
        model = None
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db_policy")
        
        # Get or create collection for travel policies
        try:
            collection = chroma_client.get_collection(name="travel_policies")
        except:
            collection = chroma_client.create_collection(
                name="travel_policies",
                metadata={"description": "Travel policy embeddings for semantic search"}
            )
        
        return model, collection
    except Exception as e:
        st.error(f"AI initialization failed: {e}")
        return None, None

# --- ENHANCED AI REQUEST VALIDATION & OPTIMIZATION ---
def ai_validate_travel_request(destination, departure_date, return_date, estimated_cost, travel_class, purpose, employee_email, business_justification, policies, gemini_model=None):
    """
    Enhanced AI-powered validation system that comprehensively checks requests against company policies
    Returns: (is_valid: bool, errors: list[str], warnings: list[str], suggestions: list[str], compliance_score: float)
    """
    errors, warnings, suggestions = [], [], []
    compliance_score = 100.0
    
    # === BASIC VALIDATION ===
    # Date validation
    if return_date <= departure_date:
        errors.append("❌ Return date must be after departure date")
        compliance_score -= 15
    
    if departure_date <= datetime.now().date():
        errors.append("❌ Departure date cannot be in the past")
        compliance_score -= 10
    
    # Duration analysis
    duration = (return_date - departure_date).days
    if duration > 30:
        warnings.append("⚠️ Trip duration exceeds 30 days - requires special long-term travel approval")
        compliance_score -= 5
    elif duration < 1:
        errors.append("❌ Trip duration must be at least 1 day")
        compliance_score -= 20
    
    # === DESTINATION-BASED POLICY VALIDATION ===
    dest_lower = destination.lower()
    
    # International vs Domestic detection
    international_destinations = [
        'london', 'paris', 'tokyo', 'singapore', 'sydney', 'toronto', 'mumbai', 'beijing',
        'madrid', 'rome', 'berlin', 'amsterdam', 'zurich', 'hong kong', 'dubai', 'bangkok'
    ]
    is_international = any(dest in dest_lower for dest in international_destinations)
    
    if is_international:
        suggestions.append("🌍 International travel detected - ensure passport validity (6+ months)")
        suggestions.append("🛡️ International travel insurance is mandatory")
        suggestions.append("💉 Check vaccination requirements for destination")
        if duration < 3:
            warnings.append("⚠️ Short international trips may not be cost-effective")
            compliance_score -= 3
    
    # === COST VALIDATION BY DESTINATION TIER ===
    tier1_cities = ['london', 'tokyo', 'singapore', 'new york', 'san francisco', 'zurich', 'hong kong']
    tier2_cities = ['manchester', 'osaka', 'kuala lumpur', 'chicago', 'boston', 'madrid', 'sydney']
    
    if dest_lower in tier1_cities:
        tier = 1
        daily_budget_cap = 600  # Tier 1: Premium cities
        hotel_cap_per_night = 350
    elif dest_lower in tier2_cities:
        tier = 2
        daily_budget_cap = 450  # Tier 2: Major cities
        hotel_cap_per_night = 250
    else:
        tier = 3
        daily_budget_cap = 350  # Tier 3: Other locations
        hotel_cap_per_night = 200
    
    # Total cost validation
    expected_total_cap = daily_budget_cap * duration
    if estimated_cost > expected_total_cap * 1.3:  # 30% tolerance
        errors.append(f"❌ Estimated cost (${estimated_cost:,.0f}) significantly exceeds policy limit (~${expected_total_cap:,.0f}) for {destination}")
        compliance_score -= 20
    elif estimated_cost > expected_total_cap:
        warnings.append(f"⚠️ Cost (${estimated_cost:,.0f}) exceeds recommended budget (~${expected_total_cap:,.0f})")
        compliance_score -= 8
    
    # Hotel cost validation
    estimated_hotel_cost = estimated_cost * 0.5  # Assume 50% of total cost is accommodation
    estimated_hotel_per_night = estimated_hotel_cost / duration
    if estimated_hotel_per_night > hotel_cap_per_night * 1.2:
        errors.append(f"❌ Hotel cost (~${estimated_hotel_per_night:.0f}/night) exceeds policy limit (${hotel_cap_per_night}/night)")
        compliance_score -= 15
    elif estimated_hotel_per_night > hotel_cap_per_night:
        warnings.append(f"⚠️ Hotel cost may exceed recommended limit (${hotel_cap_per_night}/night)")
        compliance_score -= 5
    
    # === TRAVEL CLASS VALIDATION ===
    if travel_class in ['Business Class', 'First Class']:
        if not is_international:
            errors.append("❌ Business/First class only allowed for international flights")
            compliance_score -= 25
        elif duration < 6 and travel_class == 'Business Class':
            warnings.append("⚠️ Business class typically approved for flights 6+ hours")
            compliance_score -= 10
        elif travel_class == 'First Class' and estimated_cost < 5000:
            warnings.append("⚠️ First class requires executive approval for high-value trips")
            compliance_score -= 15
    
    # === PURPOSE-BASED VALIDATION ===
    purpose_lower = purpose.lower()
    
    if 'client' in purpose_lower or 'customer' in purpose_lower:
        if estimated_cost > 3000:
            suggestions.append("💼 High-value client meeting - ensure sales director approval")
        suggestions.append("🤝 Client meeting - document expected business outcomes")
    
    if 'conference' in purpose_lower or 'training' in purpose_lower:
        suggestions.append("📚 Training/Conference - HR notification required")
        if duration > 5:
            warnings.append("⚠️ Extended training trips require educational value justification")
            compliance_score -= 5
    
    if 'vendor' in purpose_lower or 'supplier' in purpose_lower:
        suggestions.append("🏭 Vendor meeting - document procurement business case")
    
    # === BUSINESS JUSTIFICATION ANALYSIS ===
    if len(business_justification.strip()) < 50:
        warnings.append("⚠️ Business justification appears brief - provide more detailed explanation")
        compliance_score -= 10
    
    justification_lower = business_justification.lower()
    business_keywords = ['revenue', 'sales', 'contract', 'client', 'opportunity', 'meeting', 'negotiation', 'partnership']
    if not any(keyword in justification_lower for keyword in business_keywords):
        warnings.append("⚠️ Business justification should clearly explain business value and objectives")
        compliance_score -= 8
    
    # === TIMING VALIDATION ===
    advance_notice = (departure_date - datetime.now().date()).days
    if advance_notice < 7:
        if advance_notice < 3:
            warnings.append("⚠️ Less than 3 days notice - emergency approval required")
            compliance_score -= 15
        else:
            warnings.append("⚠️ Less than 7 days notice - limited booking options and higher costs")
            compliance_score -= 8
    
    # === POLICY-SPECIFIC VALIDATION ===
    policy_violations = []
    policy_compliance = []
    
    for policy in (policies or [])[:20]:  # Check top 20 policies
        rule_name = policy.get('rule_name', '').lower()
        description = policy.get('description', '').lower()
        
        # Flight policy checks
        if 'flight' in rule_name or 'class' in rule_name:
            if 'business class' in description and travel_class == 'Business Class':
                if 'international' in description and is_international:
                    policy_compliance.append(f"✅ Business class approved for international travel")
                elif 'domestic' in description and not is_international:
                    policy_violations.append(f"❌ Business class not allowed for domestic flights per {policy['rule_name']}")
                    compliance_score -= 15
        
        # Hotel policy checks
        if 'hotel' in rule_name or 'accommodation' in rule_name:
            if '$300' in description and tier == 1:
                policy_compliance.append(f"✅ Tier 1 city hotel policy compliant")
            elif '$200' in description and tier in [2, 3]:
                policy_compliance.append(f"✅ Hotel policy compliant for destination tier")
        
        # Meal policy checks
        if 'meal' in rule_name or 'per diem' in rule_name:
            if '$75' in description or 'daily' in description:
                expected_meal_cost = 75 * duration
                if estimated_cost * 0.2 > expected_meal_cost * 1.3:  # Assume 20% for meals
                    warnings.append(f"⚠️ Meal expenses may exceed per diem allowance")
                    compliance_score -= 5
    
    # === AI-POWERED INTELLIGENT ANALYSIS ===
    if gemini_model:
        try:
            ai_analysis_prompt = f"""
Analyze this travel request for policy compliance and provide intelligent recommendations:

Request Details:
- Destination: {destination} (Tier {tier})
- Duration: {duration} days
- Cost: ${estimated_cost:,.2f}
- Travel Class: {travel_class}
- Purpose: {purpose}
- International: {'Yes' if is_international else 'No'}
- Business Justification: {business_justification[:200]}...

Current Issues Found:
- Errors: {len(errors)}
- Warnings: {len(warnings)}

Provide a brief analysis focusing on:
1. Any additional policy concerns not caught by automated rules
2. Cost optimization opportunities
3. Risk assessment for this trip
4. Approval likelihood based on business case

Keep response under 150 words.
"""
            
            ai_response = gemini_model.generate_content(ai_analysis_prompt)
            if ai_response and ai_response.text:
                suggestions.append(f"🤖 AI Analysis: {ai_response.text}")
        except Exception as e:
            pass  # AI analysis is optional
    
    # === GENERATE OPTIMIZATION SUGGESTIONS ===
    if estimated_cost > expected_total_cap * 0.8:  # If cost is high
        suggestions.append(f"💡 Consider booking {advance_notice + 7} days earlier for better rates")
        suggestions.append(f"🏨 Look for corporate partner hotels in {destination}")
        if tier == 1:
            suggestions.append("🚇 Use public transportation - excellent options in major cities")
    
    if travel_class in ['Business Class', 'First Class'] and duration < 8:
        suggestions.append("✈️ Consider Premium Economy for shorter international flights")
    
    # === FINAL COMPLIANCE SCORE CALCULATION ===
    compliance_score = max(0, min(100, compliance_score))  # Ensure 0-100 range
    
    # Determine overall validity
    is_valid = len(errors) == 0 and compliance_score >= 70
    
    return is_valid, errors, warnings, suggestions, compliance_score, policy_compliance, policy_violations

def generate_validation_report(is_valid, errors, warnings, suggestions, compliance_score, policy_compliance, policy_violations, destination, estimated_cost):
    """Generate a comprehensive validation report with AI insights"""
    
    # Determine compliance level
    if compliance_score >= 90:
        compliance_level = "🟢 EXCELLENT"
        compliance_color = "#28a745"
    elif compliance_score >= 75:
        compliance_level = "🟡 GOOD"
        compliance_color = "#ffc107"
    elif compliance_score >= 60:
        compliance_level = "🟠 NEEDS REVIEW"
        compliance_color = "#fd7e14"
    else:
        compliance_level = "🔴 HIGH RISK"
        compliance_color = "#dc3545"
    
    report = f"""
## 🔍 AI VALIDATION REPORT

### 📊 Compliance Overview
**Overall Status:** {compliance_level}  
**Compliance Score:** {compliance_score:.1f}/100  
**Request Status:** {"✅ APPROVED" if is_valid else "❌ REQUIRES ATTENTION"}

---

### 📋 Validation Results
"""
    
    if errors:
        report += f"""
#### ❌ CRITICAL ISSUES ({len(errors)})
"""
        for error in errors:
            report += f"- {error}\n"
    
    if warnings:
        report += f"""
#### ⚠️ WARNINGS ({len(warnings)})
"""
        for warning in warnings:
            report += f"- {warning}\n"
    
    if policy_compliance:
        report += f"""
#### ✅ POLICY COMPLIANCE ({len(policy_compliance)})
"""
        for compliance in policy_compliance:
            report += f"- {compliance}\n"
    
    if policy_violations:
        report += f"""
#### 🚫 POLICY VIOLATIONS ({len(policy_violations)})
"""
        for violation in policy_violations:
            report += f"- {violation}\n"
    
    if suggestions:
        report += f"""
#### 💡 RECOMMENDATIONS ({len(suggestions)})
"""
        for suggestion in suggestions:
            report += f"- {suggestion}\n"
    
    report += f"""
---

### 📈 Cost Analysis
- **Estimated Total:** ${estimated_cost:,.2f}
- **Destination:** {destination}
- **Cost Efficiency:** {"High" if compliance_score >= 80 else "Medium" if compliance_score >= 60 else "Low"}

### 🎯 Next Steps
"""
    
    if is_valid:
        report += """- ✅ Request meets policy requirements
- 📋 Submit for approval workflow
- 📅 Calendar integration available
- 💼 Proceed with booking after approval"""
    else:
        report += """- ❌ Address critical issues before submission
- 📝 Revise business justification if needed
- 💰 Consider cost optimization suggestions
- 🔄 Re-validate after corrections"""
    
    return report

def validate_travel_request(destination, departure_date, return_date, estimated_cost, policies):
    """Legacy validation function - maintained for backward compatibility"""
    errors, warnings = [] , []
    # Date sanity
    if return_date <= departure_date:
        errors.append("Return date must be after departure date.")
    # Duration
    duration = (return_date - departure_date).days
    if duration > 30:
        warnings.append("Trip duration exceeds 30 days; long-term travel requires special approval.")
    # Cost caps heuristic by tier (align with get_cost_estimates tiers)
    dest = destination.lower()
    tier1 = ['london', 'tokyo', 'singapore', 'new york', 'san francisco']
    tier2 = ['manchester', 'osaka', 'kuala lumpur', 'chicago', 'boston']
    tier = 1 if dest in tier1 else 2 if dest in tier2 else 3
    daily_caps = {1: 600, 2: 450, 3: 350}  # rough cap per day
    cap = daily_caps[tier] * max(1, duration)
    if estimated_cost > cap * 1.5:  # allow 50% margin
        warnings.append(f"Estimated cost (${estimated_cost:,.0f}) significantly exceeds typical cap (~${cap:,.0f}).")
    # Pull explicit policy hints
    for p in (policies or [])[:15]:
        textp = f"{p.get('rule_name','')} {p.get('description','')}".lower()
        if 'business class' in textp and duration < 6:
            warnings.append("Business class usually allowed for 6+ hour flights only.")
            break
    return len(errors) == 0, errors, warnings

def suggest_cost_optimizations(destination, travel_info, estimated_cost, duration_days):
    """Return simple cost optimization suggestions with potential savings."""
    suggestions = []
    costs = travel_info.get('estimated_costs', {})
    hotel = float(costs.get('accommodation', 0))
    meals = float(costs.get('meals', 0))
    transport = float(costs.get('local_transport', 0))
    # Hotel: suggest business district nearby or tier down
    if hotel > 0:
        suggestions.append({
            'area': 'Accommodation',
            'tip': 'Consider hotels slightly outside prime business districts or corporate rates.',
            'saving': round(hotel * 0.15, 2)
        })
    # Meals: per diem respect
    if meals > 0:
        suggestions.append({
            'area': 'Meals',
            'tip': 'Use company per-diem restaurants or hotel breakfast inclusions.',
            'saving': round(meals * 0.10, 2)
        })
    # Transport: public transit
    if transport > 0:
        suggestions.append({
            'area': 'Transport',
            'tip': 'Use public transit pass or ride-share pooling when feasible.',
            'saving': round(transport * 0.20, 2)
        })
    # Flight: off-peak
    suggestions.append({
        'area': 'Flights',
        'tip': 'Choose off-peak flight times or 1-stop within policy to reduce airfare.',
        'saving': round(estimated_cost * 0.08, 2)
    })
    total_saving = sum(s['saving'] for s in suggestions)
    return suggestions, total_saving

def log_policy_interaction(event_type, payload):
    """Persist lightweight learning logs for future policy improvements."""
    try:
        line = json.dumps({'ts': datetime.now().isoformat(), 'type': event_type, 'data': payload}, ensure_ascii=False)
        with open('policy_learning_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

def populate_vector_database(policies, collection):
    """Enhanced vector database population with better embeddings and metadata"""
    try:
        # Check if collection already has documents
        count = collection.count()
        if count > 0:
            return
            
        # Add policies to vector database with enhanced content
        documents = []
        metadatas = []
        ids = []
        
        for i, policy in enumerate(policies):
            # Create enhanced document text for better semantic search
            enhanced_doc = f"""
{policy['rule_name']}

{policy['description']}

Keywords: {policy.get('searchable_text', '')}

Common Questions: 
- What is the policy for {policy['rule_name'].lower()}?
- How much can I spend on {policy['rule_name'].lower()}?
- What are the rules for {policy['rule_name'].lower()}?
"""
            
            documents.append(enhanced_doc)
            metadatas.append({
                'rule_name': policy['rule_name'],
                'description': policy['description'],
                'category': policy.get('category', 'general')
            })
            ids.append(f"policy_{i}")
        
        # Also add common question variations for better matching
        common_questions = [
            {
                'text': "Hotel accommodation limits costs per night room booking standards",
                'meta': {'rule_name': 'Hotel Policy', 'category': 'accommodation'}
            },
            {
                'text': "Flight class business economy international domestic travel flying airplane tickets",
                'meta': {'rule_name': 'Flight Policy', 'category': 'transportation'}
            },
            {
                'text': "Meal allowance daily food expense restaurant dining costs per day",
                'meta': {'rule_name': 'Meal Policy', 'category': 'expense'}
            },
            {
                'text': "Approval manager director VP cost limits authorization required",
                'meta': {'rule_name': 'Approval Policy', 'category': 'approval'}
            }
        ]
        
        for j, q in enumerate(common_questions):
            documents.append(q['text'])
            metadatas.append(q['meta'])
            ids.append(f"common_q_{j}")
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        st.success(f"✅ Enhanced vector database populated with {len(policies)} policies + {len(common_questions)} question patterns")
        
    except Exception as e:
        st.error(f"Failed to populate vector database: {e}")

def ai_policy_search(query, policies, gemini_model, policy_collection):
    """Enhanced AI-powered policy search using Gemini with intelligent conversation"""
    try:
        # Fallback to local search if AI components not available
        if not gemini_model or not policy_collection:
            return smart_policy_search(query, policies)
            
        # Search vector database for relevant policies
        results = policy_collection.query(
            query_texts=[query],
            n_results=5  # Get more results for better context
        )
        
        if not results['documents'] or not results['documents'][0]:
            return smart_policy_search(query, policies)
        
        # Prepare enhanced context for Gemini
        relevant_policies = results['documents'][0]
        
        # Also include some general policies for context
        general_context = ""
        for policy in policies[:10]:  # Include top policies for broader context
            general_context += f"• {policy['rule_name']}: {policy['description']}\n"
        
        # Create comprehensive prompt for intelligent conversation
        prompt = f"""You are an intelligent travel policy assistant for a company. Your role is to provide helpful, conversational, and accurate responses about travel policies.

CONVERSATION STYLE:
- Be friendly and professional, like a helpful HR representative
- Use natural language and explain things clearly
- If you don't know something specific, suggest what the employee should do
- Provide practical examples when helpful
- Use emojis appropriately to make responses engaging

AVAILABLE TRAVEL POLICIES:
{chr(10).join([f"• {doc}" for doc in relevant_policies])}

ADDITIONAL COMPANY POLICIES FOR CONTEXT:
{general_context}

EMPLOYEE QUESTION: "{query}"

Please provide a helpful, conversational response based on the policies above. If the question relates to policies not covered above, you can:
1. Provide general guidance based on typical corporate travel policies
2. Suggest who they should contact for specific questions
3. Explain what information they might need to provide

Make your response practical and actionable."""

        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            return f"🤖 **AI Assistant:**\n\n{response.text}\n\n💡 *Need more specific details? Try asking about particular aspects like costs, approval process, or booking requirements.*"
        else:
            return smart_policy_search(query, policies)
            
    except Exception as e:
        st.warning(f"AI search temporarily unavailable, using local search: {e}")
        return smart_policy_search(query, policies)

def answer_policy_question(question):
    """Policy Q&A function using database search and AI assistant"""
    try:
        if not question or not question.strip():
            return "Question cannot be empty."
        
        # Load policies from database
        policies = load_policies()
        if not policies:
            return "No policies found in database."
        
        # Filter relevant policies based on question keywords
        q_lower = question.lower()
        relevant_policies = []
        
        for policy in policies:
            rule_name = policy.get('rule_name', '').lower()
            description = policy.get('description', '').lower()
            
            if (q_lower in rule_name or q_lower in description or 
                any(keyword in rule_name or keyword in description 
                    for keyword in q_lower.split())):
                relevant_policies.append(policy)
        
        # If no specific matches, use top policies
        if not relevant_policies:
            relevant_policies = policies[:5]
        
        # Initialize AI components
        gemini_model, policy_collection = initialize_ai_components()
        
        # Use AI search with filtered policies
        return ai_policy_search(question, relevant_policies, gemini_model, policy_collection)
        
    except Exception as e:
        return f"Service temporarily unavailable: {str(e)}"

# --- APPROVAL WORKFLOW SYSTEM ---
def get_approval_workflow(employee_data, estimated_cost, destination, purpose):
    """Determine approval workflow based on cost, department, and travel type"""
    employee_id, first_name, last_name, job_level, remaining_budget, department = employee_data
    
    workflow_steps = []
    approval_rules = []
    
    # Department-specific rules
    department_rules = {
        'Sales': {'threshold': 5000, 'requires_director': True},
        'Engineering': {'threshold': 3000, 'requires_director': False},
        'Marketing': {'threshold': 4000, 'requires_director': True},
        'Finance': {'threshold': 2000, 'requires_director': True},
        'HR': {'threshold': 2500, 'requires_director': False}
    }
    
    dept_rule = department_rules.get(department, {'threshold': 3000, 'requires_director': True})
    
    # International travel check
    international_destinations = ['london', 'paris', 'tokyo', 'singapore', 'sydney', 'toronto', 'mumbai', 'beijing']
    is_international = any(dest in destination.lower() for dest in international_destinations) or \
                      destination.lower() not in ['usa', 'united states', 'domestic']
    
    # Determine approval levels
    if estimated_cost <= 1000 and not is_international:
        # Level 1: Manager only
        workflow_steps.append({
            'level': 1,
            'approver_type': 'Manager',
            'department': department,
            'auto_approve': job_level in ['Director', 'VP', 'C-Level']
        })
        approval_rules.append("✅ Low-cost domestic travel - Manager approval only")
        
    elif estimated_cost <= dept_rule['threshold'] and not is_international:
        # Level 2: Manager + Department Head
        workflow_steps.extend([
            {'level': 1, 'approver_type': 'Manager', 'department': department},
            {'level': 2, 'approver_type': 'Director', 'department': department}
        ])
        approval_rules.append(f"⚠️ Medium-cost travel - Requires {department} department approval")
        
    elif is_international or estimated_cost > dept_rule['threshold']:
        # Level 3: Full approval chain
        workflow_steps.extend([
            {'level': 1, 'approver_type': 'Manager', 'department': department},
            {'level': 2, 'approver_type': 'Director', 'department': department},
            {'level': 3, 'approver_type': 'VP', 'department': 'Executive'}
        ])
        approval_rules.append("🔴 High-cost/International travel - Full approval chain required")
        
        # Additional requirements for international
        if is_international:
            approval_rules.extend([
                "📋 International travel checklist required",
                "🛂 Passport validity check (6+ months)",
                "💉 Vaccination requirements review",
                "🛡️ Travel insurance mandatory"
            ])
    
    # Special rules for certain purposes
    if 'conference' in purpose.lower() or 'training' in purpose.lower():
        approval_rules.append("📚 Training/Conference - HR notification required")
        
    if 'client' in purpose.lower() or 'customer' in purpose.lower():
        approval_rules.append("🤝 Client meeting - Sales director notification")
    
    return workflow_steps, approval_rules

def create_approval_workflow(request_id, workflow_steps, connection):
    """Create approval workflow records in database"""
    for step in workflow_steps:
        # Find appropriate approver (simplified - in real system would use proper org chart)
        approver_query = """
            SELECT employee_id, first_name, last_name, email 
            FROM employees 
            WHERE job_level = :approver_type 
            AND department = :department 
            LIMIT 1
        """
        
        try:
            approver_result = connection.execute(
                text(approver_query),
                {
                    'approver_type': step['approver_type'],
                    'department': step['department']
                }
            )
            approver = approver_result.fetchone()
            
            if approver:
                approver_id, approver_name, approver_lastname, approver_email = approver
                
                # Auto-approve if employee has sufficient authority
                status = 'approved' if step.get('auto_approve', False) else 'pending'
                comments = 'Auto-approved based on authority level' if status == 'approved' else 'Awaiting approval'
                
                connection.execute(
                    text("""
                        INSERT INTO approval_workflows 
                        (request_id, approval_level, approver_id, status, comments, created_at) 
                        VALUES (:req_id, :level, :approver_id, :status, :comments, NOW())
                    """),
                    {
                        'req_id': request_id,
                        'level': step['level'],
                        'approver_id': approver_id,
                        'status': status,
                        'comments': comments
                    }
                )
            else:
                # Create workflow without specific approver (to be assigned later)
                connection.execute(
                    text("""
                        INSERT INTO approval_workflows 
                        (request_id, approval_level, status, comments, created_at) 
                        VALUES (:req_id, :level, 'pending', 'Awaiting approver assignment', NOW())
                    """),
                    {
                        'req_id': request_id,
                        'level': step['level']
                    }
                )
        except Exception as e:
            st.warning(f"Could not assign approver for level {step['level']}: {e}")

# --- TRAVEL PLANNING SYSTEM ---
def get_travel_information(destination, departure_date, return_date):
    """Get comprehensive travel information for destination"""
    travel_info = {
        'destination_info': {},
        'weather_forecast': {},
        'travel_advisories': {},
        'currency_info': {},
        'time_zone': {},
        'estimated_costs': {}
    }
    
    try:
        # Destination information
        travel_info['destination_info'] = get_destination_info(destination)
        
        # Weather forecast
        travel_info['weather_forecast'] = get_weather_forecast(destination, departure_date)
        
        # Travel advisories and safety
        travel_info['travel_advisories'] = get_travel_advisories(destination)
        
        # Currency and exchange rates
        travel_info['currency_info'] = get_currency_info(destination)
        
        # Time zone information
        travel_info['time_zone'] = get_timezone_info(destination)
        
        # Cost estimates
        travel_info['estimated_costs'] = get_cost_estimates(destination, departure_date, return_date)
        
    except Exception as e:
        st.warning(f"Some travel information unavailable: {e}")
    
    return travel_info

def get_destination_info(destination):
    """Get basic destination information"""
    # Destination database lookup
    destination_db = {
        'london': {
            'country': 'United Kingdom',
            'language': 'English',
            'currency': 'GBP',
            'business_hours': '9:00 AM - 5:00 PM',
            'cultural_notes': 'Business attire expected. Punctuality highly valued.',
            'transportation': 'Excellent public transport. Oyster card recommended.',
            'business_districts': 'City of London, Canary Wharf, Westminster'
        },
        'tokyo': {
            'country': 'Japan',
            'language': 'Japanese',
            'currency': 'JPY',
            'business_hours': '9:00 AM - 6:00 PM',
            'cultural_notes': 'Business cards essential. Bow when greeting.',
            'transportation': 'JR Pass for tourists. Very punctual trains.',
            'business_districts': 'Marunouchi, Shibuya, Shinjuku'
        },
        'singapore': {
            'country': 'Singapore',
            'language': 'English, Mandarin, Malay',
            'currency': 'SGD',
            'business_hours': '9:00 AM - 6:00 PM',
            'cultural_notes': 'Multicultural environment. English widely spoken.',
            'transportation': 'MRT system excellent. Grab for taxis.',
            'business_districts': 'Central Business District, Marina Bay'
        },
        'new york': {
            'country': 'United States',
            'language': 'English',
            'currency': 'USD',
            'business_hours': '9:00 AM - 6:00 PM',
            'cultural_notes': 'Fast-paced business environment. Direct communication style.',
            'transportation': 'Subway, taxis, Uber/Lyft widely available.',
            'business_districts': 'Manhattan Financial District, Midtown'
        }
    }
    
    return destination_db.get(destination.lower(), {
        'country': 'Information not available',
        'language': 'Check local requirements',
        'currency': 'Local currency',
        'business_hours': 'Standard business hours',
        'cultural_notes': 'Research local business customs',
        'transportation': 'Check local transport options',
        'business_districts': 'Research main business areas'
    })

def get_weather_forecast(destination, departure_date):
    """Get weather forecast for travel dates"""
    # Simplified weather data (in real app would use weather API)
    weather_data = {
        'temperature_range': '15-25°C',
        'conditions': 'Partly cloudy',
        'precipitation': '20% chance of rain',
        'recommendations': [
            'Pack light jacket for evenings',
            'Umbrella recommended',
            'Business attire suitable'
        ]
    }
    return weather_data

def get_travel_advisories(destination):
    """Get travel advisories and safety information"""
    advisories = {
        'safety_level': 'Low Risk',
        'health_requirements': 'No special vaccinations required',
        'documentation': 'Passport required, valid for 6+ months',
        'covid_restrictions': 'Check latest COVID-19 travel requirements',
        'emergency_contacts': {
            'local_emergency': '112 (EU), 911 (US), 000 (AU)',
            'embassy': 'Contact nearest embassy/consulate'
        }
    }
    return advisories

def get_currency_info(destination):
    """Get currency and exchange rate information"""
    currency_info = {
        'local_currency': 'Local currency',
        'exchange_rate': 'Check current rates',
        'payment_methods': 'Credit cards widely accepted',
        'cash_recommendations': 'Carry some local cash for small vendors',
        'banking': 'International ATMs available'
    }
    return currency_info

def get_timezone_info(destination):
    """Get timezone information"""
    timezone_info = {
        'local_time': 'Check world clock',
        'time_difference': 'Calculate based on your location',
        'business_hours_local': 'Standard business hours apply',
        'jet_lag_tips': [
            'Adjust sleep schedule before travel',
            'Stay hydrated during flight',
            'Get sunlight upon arrival'
        ]
    }
    return timezone_info

def get_cost_estimates(destination, departure_date, return_date):
    """Get estimated costs for the destination"""
    duration = (return_date - departure_date).days
    
    # Cost estimates based on destination tier
    cost_tiers = {
        'tier1': {'hotel': 300, 'meals': 150, 'transport': 100},  # Major cities
        'tier2': {'hotel': 200, 'meals': 120, 'transport': 80},   # Secondary cities
        'tier3': {'hotel': 150, 'meals': 100, 'transport': 60}    # Smaller cities
    }
    
    # Simplified tier assignment
    if destination.lower() in ['london', 'tokyo', 'singapore', 'new york', 'san francisco']:
        tier = 'tier1'
    elif destination.lower() in ['manchester', 'osaka', 'kuala lumpur', 'chicago', 'boston']:
        tier = 'tier2'
    else:
        tier = 'tier3'
    
    costs = cost_tiers[tier]
    
    estimates = {
        'accommodation': costs['hotel'] * duration,
        'meals': costs['meals'] * duration,
        'local_transport': costs['transport'] * duration,
        'total_estimate': (costs['hotel'] + costs['meals'] + costs['transport']) * duration,
        'daily_breakdown': costs,
        'cost_tier': tier.upper(),
        'duration_days': duration
    }
    
    return estimates

# --- CALENDAR INTEGRATION SYSTEM ---
def create_comprehensive_travel_calendar(request_data, travel_info):
    """Create comprehensive travel calendar with all events"""
    try:
        calendar = Calendar()
        
        destination = request_data['destination']
        departure_date = request_data['departure_date']
        return_date = request_data['return_date']
        purpose = request_data['purpose']
        
        # Main travel event
        main_event = Event()
        main_event.name = f"Business Travel: {destination}"
        main_event.description = f"""
Purpose: {purpose}

Travel Information:
- Destination: {destination}
- Country: {travel_info['destination_info'].get('country', 'N/A')}
- Currency: {travel_info['destination_info'].get('currency', 'N/A')}
- Language: {travel_info['destination_info'].get('language', 'N/A')}
- Business Hours: {travel_info['destination_info'].get('business_hours', 'N/A')}

Estimated Costs:
- Accommodation: ${travel_info['estimated_costs'].get('accommodation', 0):.2f}
- Meals: ${travel_info['estimated_costs'].get('meals', 0):.2f}
- Local Transport: ${travel_info['estimated_costs'].get('local_transport', 0):.2f}
- Total Estimate: ${travel_info['estimated_costs'].get('total_estimate', 0):.2f}

Weather: {travel_info['weather_forecast'].get('conditions', 'Check forecast')}
Temperature: {travel_info['weather_forecast'].get('temperature_range', 'N/A')}

Cultural Notes: {travel_info['destination_info'].get('cultural_notes', 'N/A')}
Transportation: {travel_info['destination_info'].get('transportation', 'N/A')}
        """
        main_event.begin = departure_date
        main_event.end = return_date
        main_event.location = destination
        calendar.events.add(main_event)
        
        # Pre-travel preparation events
        prep_date = departure_date - timedelta(days=7)
        prep_event = Event()
        prep_event.name = f"Travel Prep: {destination}"
        prep_event.description = f"""
Travel Preparation Checklist:
□ Confirm flight tickets
□ Book accommodation
□ Check passport validity (6+ months)
□ Review travel insurance
□ Check vaccination requirements
□ Currency exchange
□ Pack appropriate clothing
□ Download offline maps
□ Notify bank of travel
□ Set up international phone plan

Emergency Contacts:
- Local Emergency: {travel_info['travel_advisories']['emergency_contacts'].get('local_emergency', 'N/A')}
- Embassy: {travel_info['travel_advisories']['emergency_contacts'].get('embassy', 'N/A')}
        """
        prep_event.begin = prep_date
        prep_event.end = prep_date + timedelta(hours=2)
        calendar.events.add(prep_event)
        
        # Post-travel follow-up
        followup_date = return_date + timedelta(days=1)
        followup_event = Event()
        followup_event.name = f"Travel Follow-up: {destination}"
        followup_event.description = f"""
Post-Travel Tasks:
□ Submit expense report
□ Upload receipts
□ Write trip report
□ Follow up with contacts made
□ Update CRM with new leads
□ Schedule follow-up meetings
□ Share trip insights with team

Expense Submission Deadline: {(return_date + timedelta(days=30)).strftime('%Y-%m-%d')}
        """
        followup_event.begin = followup_date.replace(hour=9)
        followup_event.end = followup_date.replace(hour=10)
        calendar.events.add(followup_event)
        
        # Save calendar file
        calendar_filename = f"travel_{destination.lower().replace(' ', '_')}_{departure_date.strftime('%Y%m%d')}.ics"
        
        with open(calendar_filename, "w") as f:
            f.writelines(calendar.serialize_iter())
        
        return calendar_filename, len(calendar.events)
        
    except Exception as e:
        return None, str(e)

def generate_travel_itinerary(request_data, travel_info):
    """Generate detailed travel itinerary"""
    itinerary = f"""
# 🗓️ COMPREHENSIVE TRAVEL ITINERARY

## 📋 Trip Overview
- **Destination:** {request_data['destination']}
- **Purpose:** {request_data['purpose']}
- **Duration:** {request_data['departure_date']} to {request_data['return_date']}
- **Total Days:** {(request_data['return_date'] - request_data['departure_date']).days}

## 🌍 Destination Information
- **Country:** {travel_info['destination_info'].get('country', 'N/A')}
- **Language:** {travel_info['destination_info'].get('language', 'N/A')}
- **Currency:** {travel_info['destination_info'].get('currency', 'N/A')}
- **Business Hours:** {travel_info['destination_info'].get('business_hours', 'N/A')}
- **Time Zone:** {travel_info['time_zone'].get('local_time', 'Check local time')}

## 💰 Budget Breakdown
- **Accommodation:** ${travel_info['estimated_costs'].get('accommodation', 0):.2f}
- **Meals:** ${travel_info['estimated_costs'].get('meals', 0):.2f}
- **Local Transport:** ${travel_info['estimated_costs'].get('local_transport', 0):.2f}
- **TOTAL ESTIMATE:** ${travel_info['estimated_costs'].get('total_estimate', 0):.2f}

## 🌤️ Weather Forecast
- **Conditions:** {travel_info['weather_forecast'].get('conditions', 'Check forecast')}
- **Temperature:** {travel_info['weather_forecast'].get('temperature_range', 'N/A')}
- **Precipitation:** {travel_info['weather_forecast'].get('precipitation', 'N/A')}

## 🛡️ Travel Advisories
- **Safety Level:** {travel_info['travel_advisories'].get('safety_level', 'Check current status')}
- **Health Requirements:** {travel_info['travel_advisories'].get('health_requirements', 'N/A')}
- **Documentation:** {travel_info['travel_advisories'].get('documentation', 'N/A')}

## 🏢 Business Information
- **Business Districts:** {travel_info['destination_info'].get('business_districts', 'N/A')}
- **Cultural Notes:** {travel_info['destination_info'].get('cultural_notes', 'N/A')}
- **Transportation:** {travel_info['destination_info'].get('transportation', 'N/A')}

## 📞 Emergency Contacts
- **Local Emergency:** {travel_info['travel_advisories']['emergency_contacts'].get('local_emergency', 'N/A')}
- **Embassy:** {travel_info['travel_advisories']['emergency_contacts'].get('embassy', 'Contact nearest embassy')}

## ✅ Pre-Travel Checklist
□ Confirm flights and accommodation
□ Check passport validity (6+ months required)
□ Review and purchase travel insurance
□ Check vaccination requirements
□ Arrange currency exchange
□ Download offline maps and translation apps
□ Notify bank and credit card companies
□ Set up international phone/data plan
□ Pack weather-appropriate business attire
□ Prepare business cards and meeting materials

## 📱 Recommended Apps
- Google Translate (offline mode)
- XE Currency
- Local transportation apps
- Weather app
- Company expense tracking app
    """
    
    return itinerary

# Initialize AI components
gemini_model, policy_collection = initialize_ai_components()
def smart_policy_search(query, policies):
    """Smart local search that matches questions to relevant policies"""
    
    question_lower = query.lower()
    relevant_policies = []
    
    # Policy keyword mapping
    policy_keywords = {
        'flight': ['flight', 'plane', 'airplane', 'air travel', 'flying', 'business class', 'economy', 'airline'],
        'hotel': ['hotel', 'accommodation', 'lodging', 'stay', 'room', 'night', 'per night'],
        'meal': ['meal', 'food', 'dining', 'restaurant', 'breakfast', 'lunch', 'dinner', 'per diem', 'allowance'],
        'approval': ['approval', 'approve', 'manager', 'supervisor', 'permission', 'international', 'authorize']
    }
    
    # Find which categories are mentioned
    mentioned_categories = []
    for category, keywords in policy_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            mentioned_categories.append(category)
    
    # Match policies to categories
    for policy in policies:
        policy_content_lower = policy['content'].lower()
        policy_name_lower = policy['rule_name'].lower()
        
        # Check if this policy matches any mentioned categories
        for category in mentioned_categories:
            if category in policy_content_lower or category in policy_name_lower:
                if policy not in relevant_policies:
                    relevant_policies.append(policy)
                break
    
    # Intelligent fallback for different question types
    if not relevant_policies:
        # Cost/budget questions
        if any(word in question_lower for word in ['cost', 'price', 'money', 'budget', 'expensive', 'cheap', 'limit', 'max', 'maximum']):
            for policy in policies:
                if any(word in policy['content'].lower() for word in ['cost', 'per night', 'per diem', 'business class', 'maximum']):
                    relevant_policies.append(policy)
        
        # Rule/requirement questions
        elif any(word in question_lower for word in ['rule', 'requirement', 'must', 'need', 'required', 'allow', 'permitted', 'can i', 'should i']):
            for policy in policies:
                if any(word in policy['content'].lower() for word in ['approval', 'require', 'must', 'international']):
                    relevant_policies.append(policy)
        
        # General travel questions
        elif any(word in question_lower for word in ['travel', 'trip', 'business trip', 'policy', 'policies']):
            relevant_policies = policies[:2]  # Show first 2
    
    # If still no matches, show most relevant
    if not relevant_policies:
        relevant_policies = policies[:1]
    
    # Format response
    if len(relevant_policies) == 1:
        policy = relevant_policies[0]
        response = f"**📋 {policy['rule_name']}:**\n\n{policy['content']}\n\n"
        response += "💡 **Need more info?** Ask about specific topics like 'hotel costs' or 'flight rules'."
    
    else:
        response = f"**📋 Found {len(relevant_policies)} Relevant Policies:**\n\n"
        for i, policy in enumerate(relevant_policies[:3], 1):
            response += f"**{i}. {policy['rule_name']}:**\n{policy['content']}\n\n"
        
        if len(relevant_policies) > 3:
            response += f"💡 **And {len(relevant_policies) - 3} more policies...** Try a more specific question."
        else:
            response += "💡 **Need something specific?** Try asking about 'meal allowance' or 'approval process'."
    
    return response

# --- CALENDAR FUNCTION ---
def create_calendar_event(destination: str, departure_date: str, return_date: str, purpose: str) -> str:
    """Create a calendar event for the travel request."""
    try:
        calendar = Calendar()
        event = Event()
        
        event.name = f"Business Travel to {destination}"
        event.description = f"Purpose: {purpose}"
        event.begin = datetime.strptime(departure_date, "%Y-%m-%d")
        event.end = datetime.strptime(return_date, "%Y-%m-%d")
        event.location = destination
        
        calendar.events.add(event)
        
        with open(CALENDAR_FILE, "w") as f:
            f.writelines(calendar.serialize_iter())
        
        return f"✅ Calendar event created successfully! File saved as '{CALENDAR_FILE}'"
    except Exception as e:
        return f"❌ Error creating calendar event: {str(e)}"

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Enterprise Travel Policy Manager",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏢 Enterprise Travel Policy Management System")
st.markdown("**AI-Powered Travel Policy Assistant & Request Management**")

# Initialize session state (ChatGPT style chat)
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your AI Travel Policy Assistant. I can help you with:\n\n• Flight booking policies and costs\n• Hotel accommodation rules\n• Meal allowances and per diem\n• International travel requirements\n• Approval workflows\n• Budget and expense guidelines\n\nFeel free to ask me anything about company travel policies!",
            "timestamp": datetime.now().strftime("%H:%M")
        }
    ]

if "travel_result" not in st.session_state:
    st.session_state.travel_result = None
if "travel_history" not in st.session_state:
    st.session_state.travel_history = []

# =======================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# =======================================================================

with st.sidebar:
    st.markdown("## 🚀 System Dashboard")
    
    # System Status
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI Status", "🟢 Online")
    with col2:
        st.metric("Database", "🟢 Connected")
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    total_requests = len(st.session_state.travel_history)
    approved_requests = len([r for r in st.session_state.travel_history if "APPROVED" in r.get("status", "")])
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Requests", total_requests)
    with col4:
        st.metric("Approved", approved_requests)
    
    # Travel History
    if st.session_state.travel_history:
        st.markdown("### 📋 Recent Requests")
        for i, request in enumerate(st.session_state.travel_history[-3:]):  # Show last 3
            with st.expander(f"Request {request.get('request_id', f'#{i+1}')}"):
                st.write(f"**Destination:** {request.get('destination', 'N/A')}")
                st.write(f"**Cost:** ${request.get('cost', 0):,.0f}")
                st.write(f"**Status:** {request.get('status', 'Unknown')}")
                st.write(f"**Date:** {request.get('timestamp', 'N/A')}")
    
    st.markdown("---")
    
    # Policy Quick Reference
    st.markdown("### 📚 Policy Quick Reference")
    
    policy_info = {
        "✈️ Flight Policies": [
            "Economy class for trips < 6 hours",
            "Business class for long-haul flights",
            "Book 21+ days in advance"
        ],
        "🏨 Hotel Policies": [
            "Standard business hotels",
            "Max $250/night domestic",
            "Max $400/night international"
        ],
        "💰 Budget Limits": [
            "Standard employee: $15K/year",
            "Manager: $20K/year",
            "Executive: $30K/year"
        ]
    }
    
    for category, items in policy_info.items():
        with st.expander(category):
            for item in items:
                st.write(f"• {item}")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep welcome message
        st.rerun()
    
    if st.button("📊 Download Policy Guide", use_container_width=True):
        st.info("📖 Policy guide download would start here")
    
    if st.button("📞 Contact Support", use_container_width=True):
        st.info("📧 Email: travel-support@company.com\n📱 Phone: +1-800-TRAVEL")
    
    # Footer
    st.markdown("---")
    st.markdown("**🤖 AI Travel Assistant v2.0**")
    st.markdown("*Powered by Google Gemini*")

# Initialize AI components
gemini_model, policy_collection = initialize_ai_components()

# Load policies and populate vector database
try:
    policies = load_policies()
    if policies and policy_collection:
        populate_vector_database(policies, policy_collection)
        st.success(f"✅ Loaded {len(policies)} travel policies")
    else:
        st.warning("⚠️ Using fallback policies - database connection issue")
        policies = []
except Exception as e:
    st.error(f"❌ Error loading policies: {e}")
    policies = []

# Main tabs
tab1, tab2 = st.tabs(["🤖 Policy Q&A Chat", "📋 Submit Travel Request"])

# =======================================================================
# TAB 1: CHATGPT-STYLE POLICY Q&A INTERFACE (Requirement 1 Implementation)
# =======================================================================

with tab1:
    st.header("🤖 Travel Policy AI Assistant")
    
    # ChatGPT-style chat interface styling
    st.markdown("""
    <style>
    /* Global font styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #fafafa;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0 8px 50px;
        max-width: 80%;
        float: right;
        clear: both;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background: white;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 50px 8px 0;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
        font-size: 14px;
        line-height: 1.4;
    }
    
    /* Message time styling */
    .message-time {
        font-size: 0.8em;
        opacity: 0.7;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat Message Display
    st.markdown("### 💬 Chat")
    
    # Create scrollable chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You</strong><br>
                    {message["content"]}
                    <div class="message-time">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 AI Assistant</strong><br>
                    {message["content"]}
                    <div class="message-time">{message["timestamp"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)
    
    # Smart Quick Question Buttons
    st.markdown("### 🚀 Smart Quick Questions")
    
    # First Row - Basic Policy Questions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏨 Hotel Policy", key="hotel_quick_chat"):
            user_msg = "What level of hotel can I stay in? What are the nightly cost limits?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col2:
        if st.button("✈️ Flight Policy", key="flight_quick_chat"):
            user_msg = "When can I book business class? What are the cabin class regulations for domestic and international flights?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col3:
        if st.button("🍽️ Meal Standards", key="meal_quick_chat"):
            user_msg = "What are the daily meal allowance standards? What expenses can be reimbursed?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col4:
        if st.button("🌍 International Travel", key="international_quick_chat"):
            user_msg = "What special approvals are needed for international travel? What additional requirements are there?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    # Second Row - Practical Scenario Questions
    st.markdown("#### 💼 Common Scenario Questions")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("📋 Approval Process", key="approval_quick_chat"):
            user_msg = "Who needs to approve my travel request? What is the approval process?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col6:
        if st.button("💰 Cost Limits", key="cost_quick_chat"):
            user_msg = "What are the cost limits for different types of business travel? What happens if I exceed them?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col7:
        if st.button("⏰ Booking Time", key="booking_quick_chat"):
            user_msg = "How far in advance do I need to book? How are urgent travel requests handled?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col8:
        if st.button("📄 Reimbursement Documents", key="reimbursement_quick_chat"):
            user_msg = "What documents are needed for reimbursement? What are the reimbursement process and time requirements?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 AI thinking..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    # Chat Input Interface
    st.markdown("### ✍️ Ask Your Question")
    
    # Create chat input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message...", 
                placeholder="e.g., What's the policy for business class flights?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("📤 Send", use_container_width=True)
        
        with col3:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    # Handle chat input
    if send_button and user_input.strip():
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Generate AI reply
        with st.spinner("🤖 AI is thinking..."):
            response = answer_policy_question(user_input)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        st.rerun()
    
    # Handle clear chat
    if clear_button:
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # Keep welcome message
        st.rerun()
    
    # Chat Statistics
    if len(st.session_state.chat_messages) > 1:
        with st.expander(f"📊 Chat Statistics ({len(st.session_state.chat_messages)-1} messages)"):
            user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions Asked", len(user_messages))
            with col2:
                st.metric("AI Responses", len(assistant_messages)-1)  # Exclude welcome message
            with col3:
                if user_messages:
                    first_msg_time = user_messages[0]["timestamp"]
                    st.metric("Session Started", first_msg_time)

# =======================================================================
# TAB 2: ENHANCED TRAVEL REQUEST SUBMISSION (Requirements 2,3,4,5 Implementation)  
# =======================================================================

# Note: Removed duplicate tab1 interface to avoid confusion
# ChatGPT-style interface is now the only policy query method

with tab2:
    st.header("Submit Travel Request")
    
    # Initialize session state for travel request results
    if "travel_result" not in st.session_state:
        st.session_state.travel_result = None
    if "travel_history" not in st.session_state:
        st.session_state.travel_history = []
    
    # Fixed Travel Result Display Area
    st.markdown("### 📋 Travel Request Status")
    travel_result_container = st.container()
    
    with travel_result_container:
        if st.session_state.travel_result:
            # Show current travel request result
            st.markdown("""
            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;">
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.travel_result)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💼 **Submit your travel request below.** The result will appear here and stay fixed for easy reference.")
    
    st.markdown("---")
    
    # Travel Request Form (Simplified & Enhanced for Core Validation)
    st.markdown("### ✈️ Simplified Travel Request System")
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 20px;">
    <strong>🎯 Smart AI Validation System:</strong><br>
    ✅ <strong>Real-time Policy Check:</strong> Instant validation against company policies<br>
    ✅ <strong>Budget Analysis:</strong> Automatic cost breakdown and eligibility check<br>
    ✅ <strong>Smart Suggestions:</strong> AI recommendations for compliance and savings<br>
    ✅ <strong>Transparent Results:</strong> Clear display of costs, policies, and approval status
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("travel_request"):
        # Core Travel Information (Simplified)
        st.markdown("#### 🎯 Essential Travel Details")
        col1, col2 = st.columns(2)
        
        with col1:
            destination = st.text_input("🌍 Destination", placeholder="e.g., New York, London, Tokyo", help="Enter city or country name")
            departure_date = st.date_input(
                "📅 Departure Date", 
                value=datetime.now().date() + timedelta(days=30),
                min_value=datetime.now().date() + timedelta(days=1)
            )
            estimated_cost = st.number_input("💰 Estimated Total Cost ($)", min_value=0.0, step=100.0, value=3000.0, help="Total estimated cost including flights, hotels, meals, and transportation")
            
        with col2:
            return_date = st.date_input(
                "📅 Return Date",
                value=datetime.now().date() + timedelta(days=35),
                min_value=datetime.now().date() + timedelta(days=2)
            )
            purpose = st.selectbox("🎯 Travel Purpose", [
                "Client Meeting", "Conference/Training", "Sales Meeting", 
                "Site Visit", "Team Meeting", "Vendor Meeting", "Other"
            ], help="Business purpose affects approval requirements")
            travel_class = st.selectbox("✈️ Travel Class", [
                "Economy", "Premium Economy", "Business Class", "First Class"
            ], help="Class will be validated against company policy")
        
        # Essential Contact & Accommodation
        st.markdown("#### � Contact & Accommodation")
        col3, col4 = st.columns(2)
        
        with col3:
            employee_email = st.text_input("📧 Employee Email", placeholder="your.email@company.com", help="Required for approval notifications")
            
        with col4:
            hotel_preference = st.selectbox("🏨 Hotel Preference", [
                "Standard Business Hotel", "Economy/Budget", "Luxury (requires approval)", "Company Preferred Partners"
            ], help="Hotel type affects cost and approval requirements")
        
        # AI Validation Preview
        if destination and departure_date and return_date and estimated_cost > 0:
            st.markdown("#### 🤖 Real-time AI Validation Preview")
            duration = (return_date - departure_date).days
            
            # Create preview columns
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.markdown("**📅 Trip Overview**")
                if duration > 0:
                    st.success(f"Duration: {duration} days")
                    if duration > 14:
                        st.warning("⚠️ Extended trip: VP approval needed")
                    elif duration > 7:
                        st.info("📋 Medium trip: Director approval")
                    else:
                        st.success("✅ Standard trip: Manager approval")
                else:
                    st.error("❌ Invalid date range")
            
            with col6:
                st.markdown("**💰 Cost Analysis**")
                daily_cost = estimated_cost / max(duration, 1)
                st.info(f"Daily rate: ${daily_cost:.0f}")
                
                if estimated_cost > 5000:
                    st.error(f"⚠️ HIGH COST: ${estimated_cost:,.0f}")
                    st.caption("VP approval required")
                elif estimated_cost > 3000:
                    st.warning(f"💰 MEDIUM: ${estimated_cost:,.0f}")
                    st.caption("Director approval required")
                else:
                    st.success(f"✅ STANDARD: ${estimated_cost:,.0f}")
                    st.caption("Manager approval sufficient")
            
            with col7:
                st.markdown("**🌍 Destination Check**")
                # Enhanced destination analysis
                international_destinations = {
                    'london': 'Tier 1 (High cost)',
                    'paris': 'Tier 1 (High cost)', 
                    'tokyo': 'Tier 1 (High cost)',
                    'singapore': 'Tier 1 (High cost)',
                    'new york': 'Tier 2 (Medium cost)',
                    'chicago': 'Tier 2 (Medium cost)',
                    'toronto': 'Tier 2 (Medium cost)',
                    'mexico city': 'Tier 3 (Lower cost)'
                }
                
                dest_tier = "Domestic"
                for city, tier in international_destinations.items():
                    if city in destination.lower():
                        dest_tier = tier
                        break
                
                if "Tier 1" in dest_tier:
                    st.warning(f"🌍 {dest_tier}")
                    st.caption("High-cost destination")
                elif "Tier 2" in dest_tier:
                    st.info(f"🌍 {dest_tier}")
                    st.caption("Medium-cost destination")
                elif "Tier 3" in dest_tier:
                    st.success(f"🌍 {dest_tier}")
                    st.caption("Cost-effective destination")
                else:
                    st.success("🏠 Domestic Travel")
                    st.caption("Standard processing")
        
        # Action Buttons
        st.markdown("---")
        col8, col9 = st.columns([3, 1])
        
        with col8:
            submitted = st.form_submit_button(
                "🚀 Submit Travel Request & Get AI Analysis", 
                use_container_width=True,
                help="Get complete AI validation, cost breakdown, policy check, and eligibility determination"
            )
            
        with col9:
            preview_button = st.form_submit_button(
                "👁️ Quick Preview",
                help="Preview AI validation without submitting"
            )
    # Enhanced Form Processing (Requirements 2-5 Implementation)
    if preview_button:
        if all([destination, departure_date, return_date, estimated_cost, purpose, employee_email]):
            if return_date <= departure_date:
                st.error("❌ Return date must be after departure date")
            else:
                with st.spinner("🔍 AI Preview Analysis..."):
                    # Load policies and AI components
                    policies_cache = load_policies()
                    gemini_model, _ = initialize_ai_components()
                    
                    # Run enhanced AI validation with simplified fields
                    is_valid, errors, warnings, suggestions, compliance_score, policy_compliance, policy_violations = ai_validate_travel_request(
                        destination=destination,
                        departure_date=departure_date,
                        return_date=return_date,
                        estimated_cost=estimated_cost,
                        travel_class=travel_class,
                        purpose=purpose,
                        employee_email=employee_email,
                        business_justification=f"Travel purpose: {purpose}. Basic request validation.",
                        policies=policies_cache,
                        gemini_model=gemini_model
                    )
                    
                    # Generate comprehensive validation report
                    validation_report = generate_validation_report(
                        is_valid, errors, warnings, suggestions, compliance_score, 
                        policy_compliance, policy_violations, destination, estimated_cost
                    )
                    
                    preview_content = f"""
### 🔍 ENHANCED AI VALIDATION PREVIEW

{validation_report}

### 📊 Preview Summary
- **Validation Status:** {'✅ READY TO SUBMIT' if is_valid else '❌ REQUIRES ATTENTION'}
- **Compliance Score:** {compliance_score:.1f}/100
- **Issues Found:** {len(errors)} errors, {len(warnings)} warnings
- **Policy Checks:** {len(policy_compliance)} compliant, {len(policy_violations)} violations

### 💡 Preview Mode Notice
This is a preview analysis. Submit the form to get the complete validation with approval workflow routing.
"""
                    
                    # Display the preview
                    if is_valid:
                        st.success(f"✅ **Validation Passed** - Compliance Score: {compliance_score:.1f}/100")
                    else:
                        st.error(f"❌ **Validation Issues Found** - Compliance Score: {compliance_score:.1f}/100")
                    
                    st.markdown(preview_content)
        else:
            st.error("❌ Please fill in all required fields for preview")
    
    if submitted:
        # Enhanced validation for all required fields
        required_fields = [destination, departure_date, return_date, estimated_cost, purpose, employee_email]
        if all(required_fields):
            if return_date <= departure_date:
                st.error("❌ Return date must be after departure date!")
                st.stop()
            
            # Check if departure date is in the past
            if departure_date <= datetime.now().date():
                st.error("❌ Departure date must be in the future!")
                st.stop()
            
            with st.spinner("🤖 Processing Simplified Travel Request with Enhanced AI Validation..."):
                try:
                    # === STEP 1: ENHANCED AI VALIDATION (Requirement 2) ===
                    st.write("🔍 **Step 1:** Running comprehensive AI validation...")
                    
                    # Load policies and AI components
                    policies_cache = load_policies()
                    gemini_model, _ = initialize_ai_components()
                    
                    # Run enhanced AI validation with simplified inputs
                    is_valid, errors, warnings, suggestions, compliance_score, policy_compliance, policy_violations = ai_validate_travel_request(
                        destination=destination,
                        departure_date=departure_date,
                        return_date=return_date,
                        estimated_cost=estimated_cost,
                        travel_class=travel_class,
                        purpose=purpose,
                        employee_email=employee_email,
                        business_justification=f"Purpose: {purpose}. Hotel preference: {hotel_preference}. Travel class: {travel_class}.",
                        policies=policies_cache,
                        gemini_model=gemini_model
                    )
                    
                    # Display validation results in real-time
                    if is_valid:
                        st.success(f"✅ AI Validation Passed - Compliance Score: {compliance_score:.1f}/100")
                    else:
                        st.warning(f"⚠️ Validation Issues Found - Compliance Score: {compliance_score:.1f}/100")
                    
                    # Show critical errors if any
                    if errors:
                        st.error("**Critical Issues:**")
                        for error in errors:
                            st.error(f"• {error}")
                    
                    # === STEP 2: COMPREHENSIVE ANALYSIS & COST BREAKDOWN ===
                    st.write("� **Step 2:** Generating detailed cost breakdown and eligibility analysis...")
                    
                    # Calculate enhanced cost breakdown
                    duration = (return_date - departure_date).days
                    cost_breakdown = {
                        'flight_cost': estimated_cost * 0.6,  # 60% for flights
                        'hotel_cost': estimated_cost * 0.25,  # 25% for hotels  
                        'meals_cost': estimated_cost * 0.10,  # 10% for meals
                        'transport_cost': estimated_cost * 0.05  # 5% for local transport
                    }
                    
                    # Determine destination tier and limits
                    international_destinations = {
                        'london': {'tier': 'Tier 1', 'daily_limit': 400, 'visa': 'Required', 'currency': 'GBP', 'flight_cost': 800},
                        'paris': {'tier': 'Tier 1', 'daily_limit': 380, 'visa': 'Required', 'currency': 'EUR', 'flight_cost': 750},
                        'tokyo': {'tier': 'Tier 1', 'daily_limit': 420, 'visa': 'Required', 'currency': 'JPY', 'flight_cost': 1200},
                        'singapore': {'tier': 'Tier 1', 'daily_limit': 350, 'visa': 'Required', 'currency': 'SGD', 'flight_cost': 1000},
                        'new york': {'tier': 'Tier 2', 'daily_limit': 300, 'visa': 'Not Required', 'currency': 'USD', 'flight_cost': 400},
                        'toronto': {'tier': 'Tier 2', 'daily_limit': 280, 'visa': 'Required', 'currency': 'CAD', 'flight_cost': 350},
                        'mexico city': {'tier': 'Tier 3', 'daily_limit': 200, 'visa': 'Not Required', 'currency': 'MXN', 'flight_cost': 300}
                    }
                    
                    dest_info = None
                    for city, info in international_destinations.items():
                        if city in destination.lower():
                            dest_info = info
                            break
                    
                    if not dest_info:
                        dest_info = {'tier': 'Domestic', 'daily_limit': 250, 'visa': 'Not Required', 'currency': 'USD', 'flight_cost': 200}
                    
                    # Calculate detailed costs with realistic estimates
                    realistic_costs = {
                        'flight_cost': dest_info['flight_cost'],
                        'hotel_cost': dest_info['daily_limit'] * 0.6 * duration,  # 60% of daily limit for hotels
                        'meals_cost': dest_info['daily_limit'] * 0.3 * duration,  # 30% for meals
                        'transport_cost': dest_info['daily_limit'] * 0.1 * duration  # 10% for transport
                    }
                    realistic_total = sum(realistic_costs.values())
                    
                    # Employee budget simulation (in real system, this would come from database)
                    # For demo purposes, we'll use a reasonable budget based on email domain
                    if 'manager' in employee_email.lower() or 'director' in employee_email.lower():
                        employee_budget = 20000  # Managers get higher budget
                    elif 'vp' in employee_email.lower() or 'executive' in employee_email.lower():
                        employee_budget = 30000  # Executives get highest budget
                    else:
                        employee_budget = 15000  # Standard employee budget
                    
                    # Determine approval status based on actual budget and policies
                    approval_status = "✅ APPROVED"
                    status_details = []
                    
                    # Check against employee budget first
                    if estimated_cost > employee_budget:
                        approval_status = "❌ EXCEEDS EMPLOYEE BUDGET"
                        status_details.append(f"Cost ${estimated_cost:,} exceeds your annual budget ${employee_budget:,}")
                    # Check for reasonableness compared to market rates
                    elif estimated_cost > realistic_total * 2.0:  # More generous threshold
                        approval_status = "⚠️ COST REVIEW REQUIRED"
                        status_details.append("Cost significantly higher than market estimates - requires justification")
                    # Standard approval thresholds
                    elif estimated_cost > 5000:
                        approval_status = "🔍 VP APPROVAL REQUIRED"
                        status_details.append("High cost requires executive approval")
                    elif estimated_cost > 3000:
                        approval_status = "⚠️ DIRECTOR APPROVAL REQUIRED"
                        status_details.append("Medium cost requires director approval")
                    # Check travel class appropriateness
                    elif travel_class in ["Business Class", "First Class"] and duration < 6:
                        approval_status = "⚠️ CLASS DOWNGRADE RECOMMENDED"
                        status_details.append("Premium class not justified for short trips")
                    
                    # Override if AI compliance score is too low
                    if compliance_score < 50:
                        approval_status = "❌ POLICY VIOLATIONS"
                        status_details.append("Multiple policy violations detected")
                    elif compliance_score < 70 and approval_status == "✅ APPROVED":
                        approval_status = "⚠️ CONDITIONAL APPROVAL"
                        status_details.append("Minor policy issues - requires manager review")
                    
                    # Budget comparison logic fix
                    budget_status = "✅ Within Budget"
                    if estimated_cost <= employee_budget:
                        if estimated_cost <= realistic_total:
                            budget_status = "✅ Excellent - Under market rate"
                        elif estimated_cost <= realistic_total * 1.2:
                            budget_status = "✅ Good - Reasonable cost"
                        else:
                            budget_status = "⚠️ Higher than typical - justify if needed"
                    else:
                        budget_status = "❌ Exceeds your annual budget"
                    
                    # Generate the comprehensive result
                    result_content = f"""
# 🎯 AI Travel Request Analysis & Eligibility Report

## {approval_status}
**Overall Compliance Score: {compliance_score:.1f}/100** | **Trip: {destination} ({duration} days)**

---

## 💰 Detailed Cost Analysis & Budget Check

### � Your Request vs. Company Guidelines
| Category | Your Estimate | Realistic Cost | Company Limit | Status |
|----------|---------------|----------------|---------------|---------|
| ✈️ **Flights** | ${cost_breakdown['flight_cost']:,.0f} | ${realistic_costs['flight_cost']:,.0f} | Varies by destination | {'✅ Reasonable' if cost_breakdown['flight_cost'] <= realistic_costs['flight_cost'] * 1.2 else '⚠️ High'} |
| 🏨 **Hotels** | ${cost_breakdown['hotel_cost']:,.0f} | ${realistic_costs['hotel_cost']:,.0f} | ${dest_info['daily_limit'] * 0.6:.0f}/night | {'✅ Within Limit' if cost_breakdown['hotel_cost'] <= realistic_costs['hotel_cost'] * 1.1 else '❌ Exceeds Limit'} |
| 🍽️ **Meals** | ${cost_breakdown['meals_cost']:,.0f} | ${realistic_costs['meals_cost']:,.0f} | ${dest_info['daily_limit'] * 0.3:.0f}/day | {'✅ Within Limit' if cost_breakdown['meals_cost'] <= realistic_costs['meals_cost'] * 1.1 else '❌ Exceeds Limit'} |
| 🚗 **Transport** | ${cost_breakdown['transport_cost']:,.0f} | ${realistic_costs['transport_cost']:,.0f} | ${dest_info['daily_limit'] * 0.1:.0f}/day | {'✅ Within Limit' if cost_breakdown['transport_cost'] <= realistic_costs['transport_cost'] * 1.1 else '❌ Exceeds Limit'} |

### � **Total Cost Comparison:**
### 💵 **Budget Analysis:**
- **Your Estimate:** ${estimated_cost:,.0f}
- **Your Annual Budget:** ${employee_budget:,.0f}
- **Remaining Budget:** ${employee_budget - estimated_cost:,.0f}
- **Market Comparison:** ${realistic_total:,.0f} (typical cost)
- **Budget Status:** {budget_status}

### 📊 **Cost Efficiency:**
- **vs Your Budget:** {((estimated_cost / employee_budget) * 100):.1f}% of annual limit
- **vs Market Rate:** {((estimated_cost / realistic_total) * 100):.1f}% of typical cost
- **Savings Opportunity:** ${max(0, estimated_cost - realistic_total):,.0f} potential savings available

---

## 🌍 Destination Information: {destination}

### 📍 **Location Details:**
- **Classification:** {dest_info['tier']} destination
- **Daily Expense Limit:** ${dest_info['daily_limit']}/day
- **Visa Requirement:** {dest_info['visa']}
- **Currency:** {dest_info['currency']}
- **Estimated Flight Cost:** ${dest_info['flight_cost']}

### 🏨 **Accommodation Guidelines:**
- **Recommended Hotels:** {hotel_preference}
- **Max Hotel Rate:** ${dest_info['daily_limit'] * 0.6:.0f}/night
- **Your Hotel Budget:** ${cost_breakdown['hotel_cost']/duration:.0f}/night
- **Status:** {'✅ Within policy' if cost_breakdown['hotel_cost']/duration <= dest_info['daily_limit'] * 0.6 else '❌ Exceeds policy'}

---

## 📋 Policy Compliance Analysis

### ✅ **Company Policy Check:**
"""
                    
                    # Add specific policy checks
                    policy_issues = []
                    policy_compliant = []
                    
                    # Flight class policy
                    if travel_class == "Economy":
                        policy_compliant.append("✅ Flight class: Economy selected (compliant)")
                    elif travel_class == "Business Class" and duration >= 6:
                        policy_compliant.append("✅ Flight class: Business justified for long trip")
                    else:
                        policy_issues.append("❌ Flight class: Premium class not justified for trip duration")
                    
                    # Cost policy
                    daily_cost = estimated_cost / duration
                    if daily_cost <= dest_info['daily_limit']:
                        policy_compliant.append("✅ Daily expenses: Within company limits")
                    else:
                        policy_issues.append(f"❌ Daily expenses: ${daily_cost:.0f}/day exceeds ${dest_info['daily_limit']}/day limit")
                    
                    # Duration policy
                    if duration <= 14:
                        policy_compliant.append("✅ Trip duration: Within standard limits")
                    else:
                        policy_issues.append("❌ Trip duration: Extended travel requires special approval")
                    
                    # Add policy results to content
                    if policy_compliant:
                        for item in policy_compliant:
                            result_content += f"{item}\n"
                    
                    if policy_issues:
                        result_content += "\n**❌ Policy Violations Found:**\n"
                        for item in policy_issues:
                            result_content += f"{item}\n"
                    
                    # Add AI validation results
                    if errors:
                        result_content += "\n**🚨 Critical Issues:**\n"
                        for error in errors[:3]:
                            result_content += f"• {error}\n"
                    
                    if warnings:
                        result_content += "\n**⚠️ Warnings:**\n"
                        for warning in warnings[:3]:
                            result_content += f"• {warning}\n"
                    
                    # Add eligibility determination
                    result_content += f"""

---

## 🎯 ELIGIBILITY DETERMINATION

### 📊 **Final Status: {approval_status}**
"""
                    
                    for detail in status_details:
                        result_content += f"• {detail}\n"
                    
                    if approval_status == "✅ APPROVED":
                        result_content += f"""
**✅ YOUR REQUEST IS APPROVED FOR SUBMISSION**
- Budget status: {budget_status}
- Estimated cost: ${estimated_cost:,} (within your ${employee_budget:,} annual limit)
- All policies met - standard approval process applies
- Processing time: 2-3 business days
"""
                    elif "BUDGET" in approval_status:
                        result_content += f"""
**❌ BUDGET ISSUE DETECTED**
- Your estimate: ${estimated_cost:,}
- Your annual limit: ${employee_budget:,} 
- Shortfall: ${estimated_cost - employee_budget:,}
- Action required: Reduce cost or request budget increase
"""
                    elif "REQUIRED" in approval_status:
                        result_content += f"""
**⚠️ HIGHER APPROVAL REQUIRED**
- Budget status: {budget_status}
- Cost: ${estimated_cost:,} (within your ${employee_budget:,} limit)
- Requires executive review due to amount
- Processing time: 5-7 business days
"""
                    elif "REVIEW" in approval_status:
                        result_content += f"""
**🔍 COST REVIEW REQUIRED**
- Budget status: {budget_status}
- Your cost: ${estimated_cost:,} vs market rate: ${realistic_total:,}
- Justification needed for higher-than-typical cost
- Manager review required
"""
                    else:
                        result_content += """
**❌ REQUEST NEEDS REVISION**
- Policy violations must be addressed
- Cost adjustments required
- Resubmit after corrections
"""
                    
                    # Add smart suggestions
                    if suggestions:
                        result_content += """

---

## 💡 AI Smart Recommendations

### 🎯 **Cost Optimization Suggestions:**
"""
                        for i, suggestion in enumerate(suggestions[:3], 1):
                            if isinstance(suggestion, dict):
                                result_content += f"{i}. **{suggestion.get('area', 'General')}:** {suggestion.get('tip', suggestion)}\n"
                            else:
                                result_content += f"{i}. {suggestion}\n"
                    
                    # Add estimated savings
                    if estimated_cost > realistic_total:
                        potential_savings = estimated_cost - realistic_total
                        result_content += f"""

### 💰 **Potential Savings: ${potential_savings:,.0f}**
- Switch to economy class: Save ~${(estimated_cost * 0.3 if travel_class != 'Economy' else 0):,.0f}
- Choose standard hotels: Save ~${max(0, cost_breakdown['hotel_cost'] - realistic_costs['hotel_cost']):,.0f}
- Optimize meal expenses: Save ~${max(0, cost_breakdown['meals_cost'] - realistic_costs['meals_cost']):,.0f}
"""
                    
                    # Add next steps
                    result_content += f"""

---

## 📞 Next Steps & Required Actions

### 🔄 **Immediate Actions Required:**
1. **Budget Status:** {budget_status} - Your ${estimated_cost:,} request vs ${employee_budget:,} annual limit
2. **Policy Review:** {'Address policy violations' if policy_issues else 'All policies compliant ✅'}
3. **Cost Optimization:** {'Consider cost reduction opportunities above' if estimated_cost > realistic_total * 1.2 else 'Cost appears reasonable'}
4. **Approval Process:** {'VP approval required (5-7 days)' if estimated_cost > 5000 else 'Standard approval (2-3 days)'}

### 📋 **Budget Summary:**
- **Annual Travel Budget:** ${employee_budget:,}
- **This Request:** ${estimated_cost:,}
- **Budget Utilization:** {((estimated_cost / employee_budget) * 100):.1f}%
- **Remaining After Trip:** ${employee_budget - estimated_cost:,}

### 📊 **Request Details:**
- **Request ID:** TRQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}
- **Employee:** {employee_email}
- **Compliance Score:** {compliance_score:.1f}/100
- **Final Status:** {approval_status}
- **Processing Time:** {'5-7 business days' if estimated_cost > 5000 else '2-3 business days'}

### 📱 **Support Contacts:**
- **Policy Questions:** hr@company.com
- **Travel Booking:** travel@company.com  
- **Budget Issues:** finance@company.com
- **Emergency Travel:** +1-800-EMERGENCY

### 🎯 **Pro Tips:**
- Book flights 3-4 weeks in advance for better rates
- Use company preferred hotels for automatic approval
- Keep all receipts for expense reporting
- Consider travel insurance for international trips
"""
                    
                    # Update session state with the comprehensive result
                    st.session_state.travel_result = result_content
                    st.session_state.travel_history.append({
                        "request_id": f"TRQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "destination": destination,
                        "cost": estimated_cost,
                        "travel_class": travel_class,
                        "status": approval_status,
                        "compliance_score": compliance_score,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": result_content
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    error_content = f"""
## ❌ System Processing Error

### � Error Details
- **Error Type:** {type(e).__name__}
- **Details:** {str(e)}
- **Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### 💼 Request Information
- **Destination:** {destination}
- **Travel Class:** {travel_class}
- **Purpose:** {purpose}
- **Estimated Cost:** ${estimated_cost:,.2f}

### � Support Actions
1. Check all form fields are properly filled
2. Try submitting again in a few minutes
3. Contact IT support if problem persists
4. For urgent requests, contact manager directly

### � Troubleshooting
- Verify internet connection
- Check if all required fields are completed
- Try refreshing the page
- Contact support: tech-support@company.com
"""
                    st.session_state.travel_result = error_content
                    st.rerun()
                    
        else:
            st.error("❌ Please fill in all required fields: Destination, Dates, Cost, Purpose, Email, and Hotel Preference")


# End of application
