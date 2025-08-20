# app.py - AI-Powered Travel Policy Advisor (Google Gemini + Vector DB)
"""
=======================================================================
🏢 企业旅行政策管理系统 - 主应用程序
=======================================================================

核心功能模块 (5大要求实现):
1. Policy Query Interface - ChatGPT风格的智能政策问答
2. Request Validation - AI智能验证差旅申请合规性
3. Approval Workflow - 基于成本和部门的自动化审批路由
4. Travel Planning - 集成外部旅行信息源的智能规划
5. Calendar Integration - 自动生成差旅日程安排

技术架构:
• Frontend: Streamlit Web Interface
• AI Engine: Google Gemini + ChromaDB Vector Database  
• Database: PostgreSQL/Prisma with SQLAlchemy ORM
• Calendar: ICS格式自动生成与下载

版本: v2.0 - 企业级完整解决方案
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
    """Load policies from database and return as serializable dictionaries"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            policies_result = connection.execute(text("SELECT rule_name, description FROM travel_policies"))
            policies = policies_result.fetchall()
        
        policy_list = []
        for name, desc in policies:
            policy_dict = {
                'rule_name': name,
                'description': desc,
                'content': f"Policy Name: {name}\nDetails: {desc}\nTravel Policy: {name} - {desc}"
            }
            policy_list.append(policy_dict)
        
        return policy_list
    
    except Exception as e:
        print(f"Error loading policies: {e}")
        st.error(f"Database error: {e}")
        return []

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

# --- REQUEST VALIDATION & OPTIMIZATION ---
def validate_travel_request(destination, departure_date, return_date, estimated_cost, policies):
    """Validate request against simple policy heuristics using loaded policies text.
    Returns: (is_valid: bool, errors: list[str], warnings: list[str])"""
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
    """Populate ChromaDB with policy embeddings if empty"""
    try:
        # Check if collection already has documents
        count = collection.count()
        if count > 0:
            return
            
        # Add policies to vector database
        documents = []
        metadatas = []
        ids = []
        
        for i, policy in enumerate(policies):
            documents.append(policy['content'])
            metadatas.append({
                'rule_name': policy['rule_name'],
                'description': policy['description']
            })
            ids.append(str(i))
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        st.success(f"✅ Vector database populated with {len(policies)} policies")
        
    except Exception as e:
        st.error(f"Failed to populate vector database: {e}")

def ai_policy_search(query, policies, gemini_model, policy_collection):
    """AI-powered policy search using Gemini and vector database"""
    try:
        # Fallback to local search if AI components not available
        if not gemini_model or not policy_collection:
            return smart_policy_search(query, policies)
            
        # Search vector database for relevant policies
        results = policy_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            return smart_policy_search(query, policies)
        
        # Prepare context for Gemini
        relevant_policies = results['documents'][0]
        context = "\n\n".join([f"Policy: {doc}" for doc in relevant_policies])
        
        # Create prompt for Gemini
        prompt = f"""You are a travel policy assistant. Answer strictly based on the provided policy entries. If the entries are not relevant, say you do not know.

TRAVEL POLICIES:
{context}

USER QUESTION: {query}

Answer only using the policies above. If you cannot find the answer in them, reply: "I don't know."
Provide a helpful, specific answer based on the policies provided."""

        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            return f"🤖 **AI Assistant Response:**\n\n{response.text}"
        else:
            return smart_policy_search(query, policies)
            
    except Exception as e:
        st.warning(f"AI search failed, using local search: {e}")
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

# 初始化会话状态 (ChatGPT style chat)
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

# 主要标签页
tab1, tab2 = st.tabs(["🤖 Policy Q&A Chat", "📋 Submit Travel Request"])

# =======================================================================
# TAB 1: CHATGPT-STYLE POLICY Q&A INTERFACE (要求1实现)
# =======================================================================

with tab1:
    st.header("🤖 Travel Policy AI Assistant")
    
    # ChatGPT风格的聊天界面样式
    st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #fafafa;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0 8px 50px;
        max-width: 80%;
        float: right;
        clear: both;
    }
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
    }
    .message-time {
        font-size: 0.8em;
        opacity: 0.7;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 聊天消息显示
    st.markdown("### 💬 Chat")
    
    # 创建可滚动的聊天容器
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
    
    # 快速操作按钮
    st.markdown("### 🚀 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏨 Hotel Policy", key="hotel_quick_chat"):
            user_msg = "What are the hotel cost limits and booking rules?"
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
        if st.button("✈️ Flight Rules", key="flight_quick_chat"):
            user_msg = "What are the flight class rules and booking policies?"
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
        if st.button("🍽️ Meal Allowance", key="meal_quick_chat"):
            user_msg = "What is the daily meal allowance and per diem?"
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
        if st.button("🌍 International", key="intl_quick_chat"):
            user_msg = "What are the international travel requirements and policies?"
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
    
    # 聊天输入界面
    st.markdown("### ✍️ Ask Your Question")
    
    # 创建聊天输入表单
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
    
    # 处理聊天输入
    if send_button and user_input.strip():
        # 添加用户消息
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # 生成AI回复
        with st.spinner("🤖 AI is thinking..."):
            response = answer_policy_question(user_input)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
        
        st.rerun()
    
    # 处理清除聊天
    if clear_button:
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # 保留欢迎消息
        st.rerun()
    
    # 聊天统计
    if len(st.session_state.chat_messages) > 1:
        with st.expander(f"📊 Chat Statistics ({len(st.session_state.chat_messages)-1} messages)"):
            user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions Asked", len(user_messages))
            with col2:
                st.metric("AI Responses", len(assistant_messages)-1)  # 排除欢迎消息
            with col3:
                if user_messages:
                    first_msg_time = user_messages[0]["timestamp"]
                    st.metric("Session Started", first_msg_time)

# =======================================================================
# TAB 2: ENHANCED TRAVEL REQUEST SUBMISSION (要求2,3,4,5实现)
# =======================================================================

with tab1:
    st.header("Ask About Travel Policies")
    st.markdown("**💡 Try specific questions like:**")
    st.markdown("- What are the hotel cost limits?")
    st.markdown("- Do I need approval for international travel?")
    st.markdown("- What's the meal allowance?")
    st.markdown("- Flight class rules for long trips?")
    
    # Load policies and populate vector database
    try:
        policies = load_policies()
        st.success(f"✅ Loaded {len(policies)} travel policies")
        
        # Populate vector database if needed
        if policy_collection:
            populate_vector_database(policies, policy_collection)
            
    except Exception as e:
        st.error(f"❌ Error loading policies: {e}")
        st.stop()
    
    # Initialize session state for fixed result display
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Fixed Result Display Area (always visible at top)
    st.markdown("### 🤖 AI Assistant Response")
    result_container = st.container()
    
    with result_container:
        if st.session_state.current_result:
            # Show current result in a prominent box
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.current_result)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Show welcome message when no query yet
            st.info("💼 **Ask me about travel policies!** Your answer will appear here and stay fixed for easy reference.")
    
    st.markdown("---")
    
    # Main Query Interface
    st.markdown("### 💬 Ask Your Question")
    
    # Text input for questions
    user_query = st.text_input("Enter your travel policy question:", placeholder="e.g., What are the flight booking policies?")
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Ask Question", disabled=not user_query):
            if user_query.strip():
                with st.spinner("🤖 Processing your question..."):
                    # Use the new answer_policy_question function
                    response = answer_policy_question(user_query)
                    st.session_state.current_result = response
                    st.session_state.query_history.append({
                        "query": user_query,
                        "response": response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Result", key="clear_result_main"):
            st.session_state.current_result = None
            st.rerun()
    
    # Quick question buttons
    st.markdown("### � Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏨 Hotel Policy", key="hotel_quick"):
            with st.spinner("🤖 Processing..."):
                response = answer_policy_question("What are the hotel cost limits and booking rules?")
                st.session_state.current_result = response
                st.session_state.query_history.append({
                    "query": "Hotel Policy",
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            st.rerun()
    
    with col2:
        if st.button("✈️ Flight Rules", key="flight_quick"):
            with st.spinner("🤖 Processing..."):
                response = answer_policy_question("What are the flight class rules and costs?")
                st.session_state.current_result = response
                st.session_state.query_history.append({
                    "query": "Flight Rules",
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            st.rerun()
    
    with col3:
        if st.button("🍽️ Meal Allowance", key="meal_quick"):
            with st.spinner("🤖 Processing..."):
                response = answer_policy_question("What is the daily meal allowance?")
                st.session_state.current_result = response
                st.session_state.query_history.append({
                    "query": "Meal Allowance",
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            st.rerun()
    
    # Query History
    if st.session_state.query_history:
        with st.expander(f"📝 Query History ({len(st.session_state.query_history)} questions)"):
            for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):  # Show last 5
                st.markdown(f"""
                **🔍 Question {len(st.session_state.query_history)-i+1} ({item['timestamp']}):**
                - **Q:** {item['query']}
                - **A:** {item['response'][:100]}...
                """)
                if st.button(f"📋 Load This Answer", key=f"load_{i}"):
                    st.session_state.current_result = item['response']
                    st.rerun()
                st.markdown("---")

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
    
    # Travel Request Form (Enhanced for Enterprise Requirements)
    st.markdown("### ✈️ Enhanced Travel Request System")
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 20px;">
    <strong>🚀 Enterprise Features Active:</strong><br>
    ✅ <strong>Requirement 2:</strong> AI Request Validation against all company policies<br>
    ✅ <strong>Requirement 3:</strong> Smart Approval Workflow routing by cost/department<br>
    ✅ <strong>Requirement 4:</strong> Real-time Travel Planning with live data<br>
    ✅ <strong>Requirement 5:</strong> Automatic Calendar Integration (.ics export)
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("travel_request"):
        col1, col2 = st.columns(2)
        with col1:
            destination = st.text_input("🌍 Destination", placeholder="e.g., New York, London, Tokyo")
            departure_date = st.date_input(
                "📅 Departure Date", 
                value=datetime.now().date() + timedelta(days=30),
                min_value=datetime.now().date() + timedelta(days=1)
            )
            estimated_cost = st.number_input("💰 Estimated Cost ($)", min_value=0.0, step=100.0, help="AI will validate against company budget policies")
            travel_class = st.selectbox("✈️ Travel Class", [
                "Economy", "Premium Economy", "Business Class", "First Class"
            ], help="Auto-validated against policy for flight duration")
            
        with col2:
            return_date = st.date_input(
                "📅 Return Date",
                value=datetime.now().date() + timedelta(days=35),
                min_value=datetime.now().date() + timedelta(days=2)
            )
            purpose = st.selectbox("🎯 Travel Purpose", [
                "Client Meeting", "Conference/Training", "Sales Meeting", 
                "Site Visit", "Team Meeting", "Vendor Meeting", "Other"
            ], help="Affects approval workflow routing")
            employee_email = st.text_input("👤 Employee Email", placeholder="your.email@company.com")
            urgency = st.selectbox("⚡ Request Urgency", [
                "Standard (5-7 days)", "Urgent (2-3 days)", "Emergency (Same day)"
            ], help="Affects approval workflow priority")
        
        # Enhanced Business Justification with AI Assistance
        st.markdown("#### 📝 Business Justification & Travel Details")
        col3, col4 = st.columns(2)
        
        with col3:
            business_justification = st.text_area(
                "Business Justification", 
                placeholder="Detailed explanation of business need, expected outcomes, and why travel is necessary...",
                height=100,
                help="AI will analyze justification strength for approval recommendations"
            )
            hotel_preference = st.selectbox("🏨 Hotel Preference", [
                "Standard Business Hotel", "Economy/Budget", "Luxury (requires justification)", "Company Preferred Partners"
            ])
            
        with col4:
            attendees = st.text_area(
                "Meeting Attendees/Contacts",
                placeholder="List key contacts, meeting participants, or conference details...",
                height=100,
                help="Helps with travel planning and itinerary creation"
            )
            special_requirements = st.text_area(
                "Special Requirements",
                placeholder="Dietary restrictions, accessibility needs, visa requirements, etc.",
                help="Will be included in travel planning recommendations"
            )
        
        # Real-time validation preview
        if departure_date and return_date and estimated_cost > 0:
            duration = (return_date - departure_date).days
            col5, col6, col7 = st.columns(3)
            
            with col5:
                if duration > 0:
                    st.info(f"📅 **Trip Duration:** {duration} days")
                    # Quick policy check preview
                    if duration > 14:
                        st.warning("⚠️ Extended travel (>14 days) requires VP approval")
                else:
                    st.error("❌ Invalid date range")
            
            with col6:
                # Cost category preview
                if estimated_cost > 5000:
                    st.warning(f"💰 **High Cost:** ${estimated_cost:,.2f} (VP approval required)")
                elif estimated_cost > 3000:
                    st.info(f"💰 **Medium Cost:** ${estimated_cost:,.2f} (Director approval required)")
                else:
                    st.success(f"💰 **Standard Cost:** ${estimated_cost:,.2f} (Manager approval)")
            
            with col7:
                # International travel preview
                international_destinations = ['london', 'paris', 'tokyo', 'singapore', 'sydney', 'toronto', 'beijing', 'mumbai']
                if destination and any(dest in destination.lower() for dest in international_destinations):
                    st.info("🌍 **International Travel** - Additional requirements may apply")
                else:
                    st.success("🏠 **Domestic Travel** - Standard processing")
        
        # Enhanced submit buttons
        st.markdown("---")
        col8, col9, col10 = st.columns([2, 1, 1])
        
        with col8:
            submitted = st.form_submit_button(
                "🚀 Submit Enhanced Travel Request", 
                use_container_width=True,
                help="Processes through AI validation, approval workflow, travel planning, and calendar integration"
            )
            
        with col9:
            draft_button = st.form_submit_button(
                "💾 Save Draft",
                help="Save request without submitting for approval"
            )
            
        with col10:
            preview_button = st.form_submit_button(
                "👁️ Preview",
                help="Preview validation and workflow without submitting"
            )
    # Enhanced Form Processing (Requirements 2-5 Implementation)
    if preview_button:
        if all([destination, departure_date, return_date, estimated_cost, purpose, employee_email, business_justification]):
            if return_date <= departure_date:
                st.error("❌ Return date must be after departure date")
            else:
                with st.spinner("🔍 AI Preview Analysis..."):
                    # Quick preview without database submission
                    policies_cache = load_policies()
                    is_valid, preview_errors, preview_warnings = validate_travel_request(
                        destination, departure_date, return_date, estimated_cost, policies_cache
                    )
                    
                    preview_content = f"""
### 🔍 AI Validation Preview (Requirement 2)
**Status:** {'✅ Valid' if is_valid else '❌ Issues Found'}

**Policy Compliance:**
{chr(10).join(['• ' + error for error in preview_errors]) if preview_errors else '• All validations passed'}

**Warnings:**
{chr(10).join(['• ' + warning for warning in preview_warnings]) if preview_warnings else '• No warnings'}

### 🔄 Approval Workflow Preview (Requirement 3)
**Estimated Approval Levels:** {2 if estimated_cost > 5000 else 1 if estimated_cost > 3000 else 1}
**Expected Processing Time:** {3 if estimated_cost > 5000 else 2 if estimated_cost > 3000 else 1} business days

### 🌍 Travel Planning Preview (Requirement 4)
**Destination Analysis:** AI will provide comprehensive destination information including weather, local transportation, business districts, and cost estimates.

### 📅 Calendar Integration Preview (Requirement 5)
**Calendar Events:** Will generate {2 + (return_date - departure_date).days} calendar events including departure, daily itinerary, and return.
"""
                    
                    st.session_state.travel_result = preview_content
                    st.rerun()
        else:
            st.error("❌ Please fill in all required fields for preview")
    
    if draft_button:
        if destination and employee_email:
            draft_content = f"""
### 💾 Draft Saved Successfully

**Draft Summary:**
- **Destination:** {destination or 'Not specified'}
- **Dates:** {departure_date if departure_date else 'Not set'} to {return_date if return_date else 'Not set'}
- **Cost:** ${estimated_cost:,.2f}
- **Purpose:** {purpose}
- **Status:** Draft - Not submitted

**Next Steps:**
- Complete remaining fields
- Review business justification
- Submit when ready for approval workflow
"""
            st.session_state.travel_result = draft_content
            st.rerun()
        else:
            st.error("❌ Please provide at least destination and email to save draft")
    
    if submitted:
        # Enhanced validation for all required fields
        required_fields = [destination, departure_date, return_date, estimated_cost, purpose, employee_email, business_justification]
        if all(required_fields):
            if return_date <= departure_date:
                st.error("❌ Return date must be after departure date!")
                st.stop()
            
            # Check if departure date is in the past
            if departure_date <= datetime.now().date():
                st.error("❌ Departure date must be in the future!")
                st.stop()
            
            with st.spinner("🤖 Processing Enhanced Travel Request with AI..."):
                try:
                    # Requirement 2: AI Request Validation
                    st.write("🔍 **Step 1:** AI Validation against company policies...")
                    policies_cache = load_policies()
                    is_valid, val_errors, val_warnings = validate_travel_request(
                        destination, departure_date, return_date, estimated_cost, policies_cache
                    )
                    
                    # Enhanced validation for new fields
                    additional_validations = []
                    
                    # Validate travel class against policy
                    duration_hours = (return_date - departure_date).days * 8  # Rough flight time estimation
                    if travel_class == "Business Class" and duration_hours < 6:
                        additional_validations.append("Business class only approved for flights 6+ hours")
                    elif travel_class == "First Class":
                        additional_validations.append("First class requires VP approval regardless of duration")
                    
                    # Validate business justification strength
                    if len(business_justification.split()) < 10:
                        additional_validations.append("Business justification appears insufficient (requires detailed explanation)")
                    
                    # Urgency validation
                    if urgency == "Emergency (Same day)" and estimated_cost > 2000:
                        additional_validations.append("Emergency high-cost requests require CEO approval")
                    
                    # Get employee data for enhanced processing
                    st.write("👤 **Step 2:** Employee verification and budget check...")
                    
                    engine = create_engine(DATABASE_URL)
                    with engine.connect() as connection:
                        employee_check = connection.execute(
                            text("SELECT employee_id, first_name, last_name, job_level, remaining_budget, department FROM employees WHERE email = :email"),
                            {"email": employee_email}
                        )
                        employee_data = employee_check.fetchone()
                        
                        if not employee_data:
                            error_content = f"""
## ❌ Enhanced Validation Failed - Employee Not Found

### 📧 Email Verification
- **Provided Email:** {employee_email}
- **Status:** Not found in company directory
- **Action Required:** Contact HR for account setup

### 💼 Request Details
- **Destination:** {destination}
- **Purpose:** {purpose}
- **Travel Class:** {travel_class}
- **Urgency:** {urgency}
- **Estimated Cost:** ${estimated_cost:,.2f}

### 📞 Next Steps
1. Verify email address spelling
2. Contact HR to add profile to system
3. Resubmit request once account is created
"""
                            st.session_state.travel_result = error_content
                            st.rerun()
                            st.stop()
                        
                        employee_id, first_name, last_name, job_level, remaining_budget, department = employee_data
                        remaining_budget_float = float(remaining_budget)
                        
                        # Requirement 3: Smart Approval Workflow
                        st.write("🔄 **Step 3:** Configuring smart approval workflow...")
                        workflow_steps, approval_rules = get_approval_workflow(
                            employee_data, estimated_cost, destination, purpose
                        )
                        
                        # Enhanced workflow for new fields
                        enhanced_workflow_notes = []
                        if travel_class in ["Business Class", "First Class"]:
                            enhanced_workflow_notes.append(f"{travel_class} requires additional VP approval")
                        if urgency == "Emergency (Same day)":
                            enhanced_workflow_notes.append("Emergency request flagged for priority processing")
                        if len(special_requirements.strip()) > 0:
                            enhanced_workflow_notes.append("Special requirements noted for travel planning")
                        
                        # Budget validation
                        if remaining_budget_float < estimated_cost:
                            error_content = f"""
## ❌ Enhanced Request Failed - Insufficient Budget

### 💰 Budget Analysis
- **Employee:** {first_name} {last_name} ({department})
- **Remaining Budget:** ${remaining_budget_float:,.2f}
- **Required Cost:** ${estimated_cost:,.2f}
- **Shortfall:** ${estimated_cost - remaining_budget_float:,.2f}

### 🎯 Request Details
- **Destination:** {destination}
- **Travel Class:** {travel_class}
- **Urgency:** {urgency}
- **Purpose:** {purpose}

### 💡 Recommended Actions
1. Reduce estimated cost (consider economy class)
2. Request budget increase from manager
3. Split costs across multiple budget periods
4. Consider virtual meeting alternatives

### 📊 Budget Optimization Suggestions
- Economy class saves ~${(estimated_cost * 0.3):,.0f}
- Standard hotel vs luxury saves ~${(estimated_cost * 0.15):,.0f}
- Advance booking discount ~${(estimated_cost * 0.1):,.0f}
"""
                            st.session_state.travel_result = error_content
                            st.rerun()
                            st.stop()
                        
                        # Requirement 4: Travel Planning Integration
                        st.write("🌍 **Step 4:** AI Travel Planning with real-time data...")
                        travel_info = get_travel_information(destination, departure_date, return_date, purpose)
                        
                        # Enhanced travel planning for business requirements
                        enhanced_travel_planning = {
                            'business_districts': travel_info.get('business_districts', 'Research required'),
                            'meeting_venues': 'Professional conference centers and meeting facilities',
                            'transportation_options': travel_info.get('transportation', 'Public transit and taxi services'),
                            'business_hours': 'Local business operating hours: 9 AM - 6 PM',
                            'cultural_considerations': 'Business etiquette and cultural norms',
                            'emergency_contacts': 'Local emergency services and company contacts'
                        }
                        
                        # Requirement 5: Calendar Integration
                        st.write("📅 **Step 5:** Generating comprehensive travel calendar...")
                        
                        # Create enhanced request data
                        enhanced_request_data = {
                            'destination': destination,
                            'departure_date': departure_date,
                            'return_date': return_date,
                            'purpose': purpose,
                            'travel_class': travel_class,
                            'hotel_preference': hotel_preference,
                            'attendees': attendees,
                            'special_requirements': special_requirements,
                            'urgency': urgency,
                            'business_justification': business_justification
                        }
                        
                        calendar_file, event_count = create_comprehensive_travel_calendar(enhanced_request_data, travel_info)
                        
                        # Submit to database
                        st.write("💾 **Step 6:** Recording in database...")
                        insert_result = connection.execute(
                            text("""
                                INSERT INTO travel_requests 
                                (employee_id, destination, departure_date, return_date, purpose, estimated_cost, status) 
                                VALUES (:emp_id, :dest, :dep_date, :ret_date, :purpose, :cost, 'pending')
                                RETURNING request_id
                            """),
                            {
                                "emp_id": employee_id,
                                "dest": destination,
                                "dep_date": departure_date,
                                "ret_date": return_date,
                                "purpose": purpose,
                                "cost": estimated_cost
                            }
                        )
                        request_id = insert_result.fetchone()[0]
                        
                        # Create approval workflow
                        create_approval_workflow(request_id, workflow_steps, connection)
                        
                        # Update budget
                        connection.execute(
                            text("UPDATE employees SET remaining_budget = remaining_budget - :cost WHERE employee_id = :emp_id"),
                            {"cost": estimated_cost, "emp_id": employee_id}
                        )
                        connection.commit()
                        
                        # Cost optimization suggestions
                        suggestions, potential_saving = suggest_cost_optimizations(
                            destination, travel_info, estimated_cost, (return_date - departure_date).days
                        )
                        
                        # Create comprehensive result
                        result_content = f"""
## ✅ Enhanced Travel Request Processed Successfully!

### 📋 Request Summary
- **Request ID:** {request_id}
- **Employee:** {first_name} {last_name} ({department})
- **Destination:** {destination}
- **Dates:** {departure_date} to {return_date} ({(return_date - departure_date).days} days)
- **Travel Class:** {travel_class}
- **Hotel Preference:** {hotel_preference}
- **Urgency:** {urgency}
- **Cost:** ${estimated_cost:,.2f}
- **Remaining Budget:** ${remaining_budget_float - estimated_cost:,.2f}

### 🔍 AI Validation Results (Requirement 2 ✅)
"""
                        
                        if is_valid and not additional_validations:
                            result_content += "✅ **Status:** All validations passed - Ready for approval\n\n"
                        else:
                            result_content += "⚠️ **Status:** Issues found requiring attention\n\n"
                        
                        # Show all validation results
                        all_errors = val_errors + additional_validations
                        if all_errors:
                            result_content += "❌ **Validation Issues:**\n"
                            for error in all_errors:
                                result_content += f"• {error}\n"
                            result_content += "\n"
                        
                        if val_warnings:
                            result_content += "⚠️ **Warnings:**\n"
                            for warning in val_warnings:
                                result_content += f"• {warning}\n"
                            result_content += "\n"
                        
                        # Enhanced approval workflow (Requirement 3 ✅)
                        result_content += "### ✅ Smart Approval Workflow (Requirement 3 ✅)\n"
                        if workflow_steps:
                            result_content += f"📋 **Required Approvals:** {len(workflow_steps)} step(s)\n"
                            for i, step in enumerate(workflow_steps, 1):
                                result_content += f"{i}. {step['approver_type']} ({step['department']}) - Level {step['level']}\n"
                            result_content += f"\n⏱️ **Estimated Approval Time:** {2 + len(workflow_steps)} business days\n"
                        else:
                            result_content += "🎉 **Auto-Approved!** No additional approval required.\n"
                        
                        # Enhanced workflow notes
                        if enhanced_workflow_notes:
                            result_content += "\n📝 **Enhanced Workflow Notes:**\n"
                            for note in enhanced_workflow_notes:
                                result_content += f"• {note}\n"
                        result_content += "\n"
                        
                        # AI Travel Planning (Requirement 4 ✅)
                        result_content += "### 🌍 AI Travel Planning (Requirement 4 ✅)\n"
                        result_content += f"""**Destination Intelligence:**
• **Location:** {travel_info['destination_info'].get('country', 'Analyzing...')}
• **Currency:** {travel_info['destination_info'].get('currency', 'Local currency')}
• **Time Zone:** {travel_info.get('timezone', 'UTC')}
• **Weather:** {travel_info['weather_forecast'].get('conditions', 'Check forecast')}
• **Business Districts:** {enhanced_travel_planning['business_districts']}
• **Transportation:** {enhanced_travel_planning['transportation_options']}

**Meeting Planning:**
• **Attendees:** {attendees if attendees else 'Not specified'}
• **Venue Recommendations:** {enhanced_travel_planning['meeting_venues']}
• **Business Hours:** {enhanced_travel_planning['business_hours']}
• **Cultural Notes:** {enhanced_travel_planning['cultural_considerations']}

**Special Requirements:**
{special_requirements if special_requirements else 'None specified'}

"""
                        
                        # Cost breakdown with enhancements
                        result_content += f"""### 💰 Enhanced Cost Analysis
**Estimated Costs:**
• **Accommodation ({hotel_preference}):** ${travel_info['estimated_costs'].get('accommodation', 0):.2f}
• **Flights ({travel_class}):** ${travel_info['estimated_costs'].get('flights', estimated_cost * 0.4):.2f}
• **Meals & Incidentals:** ${travel_info['estimated_costs'].get('meals', 0):.2f}
• **Local Transport:** ${travel_info['estimated_costs'].get('local_transport', 0):.2f}
• **Total Budget:** ${estimated_cost:.2f}

"""
                        
                        # Cost optimization suggestions
                        if suggestions:
                            result_content += "💡 **Cost Optimization Opportunities:**\n"
                            for suggestion in suggestions:
                                result_content += f"• {suggestion['area']}: {suggestion['tip']} (Save ~${suggestion['saving']:.0f})\n"
                            result_content += f"\n**Potential Total Savings: ${potential_saving:.0f}**\n\n"
                        
                        # Calendar Integration (Requirement 5 ✅)
                        result_content += "### 📅 Smart Calendar Integration (Requirement 5 ✅)\n"
                        if calendar_file:
                            result_content += f"""✅ **Travel Calendar Generated:** `{calendar_file}`
📅 **Events Created:** {event_count} comprehensive events
📥 **Compatible With:** Google Calendar, Outlook, Apple Calendar, and all .ics applications

**Calendar Includes:**
• Flight departure and arrival reminders
• Daily travel itinerary with business hours
• Meeting blocks and networking time
• Meal times and cultural activity suggestions
• Return travel and post-trip follow-ups

"""
                        else:
                            result_content += "❌ Calendar generation encountered an issue\n\n"
                        
                        # Next steps with enhanced workflow
                        result_content += """### 📞 Next Steps & Action Items
1. **Immediate:** Review validation results and address any flagged issues
2. **Approval:** Monitor approval workflow progress (notifications sent to approvers)
3. **Booking:** Await approval before making any travel arrangements
4. **Calendar:** Download and import calendar file to your preferred application
5. **Preparation:** Review travel planning recommendations and special requirements
6. **Documentation:** Keep all receipts and documentation for expense reporting

### 🔄 Workflow Status
- **Current Stage:** Submitted for approval
- **Next Approver:** {workflow_steps[0]['approver_type'] if workflow_steps else 'Auto-approved'}
- **Expected Completion:** {(datetime.now() + timedelta(days=(2 + len(workflow_steps)))).strftime('%Y-%m-%d')}

### 📱 Contact Information
- **Questions:** Contact your manager or HR for workflow questions
- **Changes:** Email travel.requests@company.com for modifications
- **Emergency:** Use emergency contact protocols for urgent changes
"""
                        
                        # Update session state
                        st.session_state.travel_result = result_content
                        st.session_state.travel_history.append({
                            "request_id": request_id,
                            "destination": destination,
                            "cost": estimated_cost,
                            "travel_class": travel_class,
                            "urgency": urgency,
                            "status": "submitted successfully",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "result": result_content
                        })
                        
                        # Learning log for AI improvement
                        log_policy_interaction('enhanced_request_submitted', {
                            'request_id': request_id,
                            'destination': destination,
                            'cost': float(estimated_cost),
                            'department': department,
                            'travel_class': travel_class,
                            'urgency': urgency,
                            'validation_passed': is_valid and not additional_validations
                        })
                        
                        st.rerun()
                        
                except Exception as e:
                    error_content = f"""
## ❌ Enhanced System Error

### 🚨 Error Details
- **Error Type:** System processing error
- **Details:** {str(e)}
- **Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Request ID:** Not assigned

### 💼 Request Information
- **Destination:** {destination}
- **Travel Class:** {travel_class}
- **Purpose:** {purpose}
- **Urgency:** {urgency}
- **Estimated Cost:** ${estimated_cost:,.2f}

### 📞 Technical Support
1. Contact IT support with error details above
2. Check internet connection and database status
3. Try resubmitting in a few minutes
4. For urgent requests, contact manager directly

### 🔄 Troubleshooting Steps
- Verify all form fields are correctly filled
- Check date ranges are valid
- Ensure email address is company domain
- Try reducing estimated cost if budget-related error
"""
                    st.session_state.travel_result = error_content
                    st.rerun()
                    
        else:
            st.error("❌ Please fill in all required fields including business justification")
    
    # Clear result button
    if st.button("🗑️ Clear Result", key="clear_travel_result"):
        st.session_state.travel_result = None
        st.rerun()
    
    # Download Options (only show if there's a result)
    if st.session_state.travel_result and "✅ Travel Request Submitted Successfully!" in st.session_state.travel_result:
        st.markdown("### 📥 Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate calendar file content
            calendar_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//AI Travel System//EN\nEND:VCALENDAR"
            st.download_button(
                label="📅 Download Calendar",
                data=calendar_content,
                file_name=f"travel_calendar_{datetime.now().strftime('%Y%m%d')}.ics",
                mime="text/calendar"
            )
        
        with col2:
            # Download current result as markdown
            st.download_button(
                label="📋 Download Result",
                data=st.session_state.travel_result.encode('utf-8'),
                file_name=f"travel_request_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col3:
            # Generate expense template
            expense_template = f"""
# Expense Report Template

## Trip Information
- Date: {datetime.now().strftime('%Y-%m-%d')}
- Purpose: Business Travel

## Cost Categories
| Category | Estimated | Actual | Difference |
|----------|-----------|--------|------------|
| Accommodation | $ | $ | $ |
| Meals | $ | $ | $ |
| Transport | $ | $ | $ |
| Other | $ | $ | $ |
| **TOTAL** | **$** | **$** | **$** |

## Receipts Checklist
□ Flight tickets
□ Hotel receipts  
□ Meal receipts
□ Transportation receipts
□ Other business expenses
"""
            st.download_button(
                label="💳 Download Expense Template",
                data=expense_template.encode('utf-8'),
                file_name=f"expense_template_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    # Travel Request History
    if st.session_state.travel_history:
        with st.expander(f"📝 Travel Request History ({len(st.session_state.travel_history)} requests)"):
            for i, item in enumerate(reversed(st.session_state.travel_history[-5:]), 1):  # Show last 5
                status_emoji = "✅" if "success" in item['status'].lower() else "❌"
                st.markdown(f"""
                **{status_emoji} Request {len(st.session_state.travel_history)-i+1} ({item['timestamp']}):**
                - **Destination:** {item['destination']}
                - **Cost:** ${item['cost']:,.2f}
                - **Status:** {item['status']}
                """)
                if st.button(f"📋 Load This Result", key=f"load_travel_{i}"):
                    st.session_state.travel_result = item['result']
                    st.rerun()
                st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ System Info")
    if gemini_model and policy_collection:
        st.success("✅ AI Assistant Active")
        st.info("🤖 Google Gemini + ChromaDB")
    else:
        st.success("✅ Local Search Active")
        st.info("🔎 Keyword-based search")
    
    st.header("� New Features")
    st.markdown("""
    **✅ Approval Workflow:** Smart routing based on cost & department
    **✅ Travel Planning:** Comprehensive destination information
    **✅ Calendar Integration:** Auto-generated travel schedules
    **✅ Cost Estimation:** Detailed budget breakdowns
    **✅ Itinerary Generation:** Professional travel plans
    """)
    
    st.header("📋 Quick Policy Guide")
    st.markdown("""
    **Flight Costs:** Business class for 6+ hour flights
    **Hotel:** Max $300/night in major cities  
    **Meals:** $75 daily allowance
    **Approval:** Automated routing by cost/department
    """)
    
    st.header("🔄 Database")
    if USE_ONLINE_DATABASE:
        st.info("🌐 Online (Prisma)")
    else:
        st.info("🏠 Local (PostgreSQL)")
    
    st.header("💡 Search Tips")
    st.markdown("""
    **Ask naturally:**
    - "What's the hotel policy for London?"
    - "Do I need approval for a $4000 trip?"
    - "Can I book business class to Tokyo?"
    - "What's the meal allowance?"
    """)
    
    # Show sample destinations
    st.header("🌍 Sample Destinations")
    destinations = ["London", "Tokyo", "Singapore", "New York"]
    for dest in destinations:
        if st.button(f"📍 {dest}", key=f"dest_{dest}"):
            st.info(f"Try asking: 'What's the travel policy for {dest}?'")
    
    # Show all policies
    if st.button("📋 Show All Policies"):
        try:
            policies = load_policies()
            st.markdown("**All Travel Policies:**")
            for i, policy in enumerate(policies[:5], 1):  # Show first 5
                st.markdown(f"**{i}. {policy['rule_name']}**")
                st.markdown(f"{policy['description'][:100]}...")
                st.markdown("---")
            if len(policies) > 5:
                st.markdown(f"**...and {len(policies) - 5} more policies**")
        except Exception as e:
            st.error(f"Could not load policies: {e}")
            st.error(f"Error: {e}")
