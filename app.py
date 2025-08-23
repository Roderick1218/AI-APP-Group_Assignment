# app.py - AI-Powered Travel Policy Advisor (Google Gemini + Vector DB)
"""
=======================================================================
🏢 Malaysian Enterprise Travel Policy Management System
=======================================================================

Core Function Modules (5 Requirements Implementation):
1. Policy Query Interface - ChatGPT-style intelligent policy Q&A
2. Submit Travel Request → AI validates against policies
3. Routes for Approval → Creates calendar events → Sends confirmations  
4. Travel Planning - Intelligent planning integrated with external travel information sources
5. Calendar Integration - Automatic generation of travel schedules

Technical Architecture:
• Frontend: Streamlit Web Interface
• AI Engine: Google Gemini + ChromaDB Vector Database  
• Database: PostgreSQL/Prisma with SQLAlchemy ORM
• Calendar: ICS format automatic generation and download
• Currency: Malaysian Ringgit (MYR)
• Base Country: Malaysia

Version: v2.0 - Malaysian Enterprise Complete Solution
=======================================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine, text
import re
import json
import requests

# --- AI AND VECTOR DB IMPORTS ---
from ics import Calendar, Event, DisplayAlarm
import chromadb
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()

# Google API Key (optional). If missing, app falls back to local search.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CURRENCY CONVERSION UTILITIES ---

def ensure_rm_currency(text):
    """Ensure all currency displays use RM instead of dollar signs"""
    if not text or not isinstance(text, str):
        return text
    
    # Replace dollar signs with RM using regex patterns
    text = re.sub(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', r'RM\1', text)
    text = re.sub(r'\$(\d+(?:\.\d{2})?)', r'RM\1', text)
    
    return text

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
                'description': 'Business travelers may book hotels up to RM400/night for international travel, RM250/night for domestic Malaysia',
                'content': 'Hotel Accommodation Standards - Business travelers may book hotels up to RM400/night for international travel, RM250/night for domestic Malaysia',
                'searchable_text': 'hotel accommodation standards business travelers book hotels RM400 night international RM250 domestic malaysia'
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

# --- SIMPLIFIED AI REQUEST VALIDATION ---
def ai_validate_travel_request(destination, departure_date, return_date, estimated_cost, travel_class, purpose, employee_email, business_justification, policies, gemini_model=None):
    """
    Simplified travel request validation system
    Submit travel request → AI validates against policies → Routes for approval → Database insertion
    """
    import psycopg2
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        # 1. Connect to database and get employee information
        conn = psycopg2.connect(os.getenv('DATABASE_ONLINE'))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT employee_id, first_name, last_name, department, job_level, 
                   annual_travel_budget, remaining_budget, manager_id
            FROM employees 
            WHERE email = %s
        """, (employee_email,))
        
        employee_data = cursor.fetchone()
        if not employee_data:
            return f"""
# ❌ Employee Validation Failed

**Error:** Employee not found in database
**Email:** {employee_email}

Please contact HR to verify your employee record.
"""
        
        employee_id, first_name, last_name, department, job_level, annual_budget, remaining_budget, manager_id = employee_data
        
        # Convert Decimal types to float to avoid type errors
        annual_budget = float(annual_budget) if annual_budget else 0.0
        remaining_budget = float(remaining_budget) if remaining_budget else 0.0
        
        # 2. Basic validation
        errors = []
        warnings = []
        
        # Date validation
        if return_date <= departure_date:
            errors.append("Return date must be after departure date")
        
            errors.append("Departure date cannot be in the past")
        
        duration = (return_date - departure_date).days
        
        # Budget check
        budget_ok = remaining_budget >= estimated_cost
        
        # Policy check
        if estimated_cost > 15000:
            errors.append("Cost exceeds RM15,000 company limit")
        
        if not budget_ok:
            errors.append(f"Insufficient budget: RM{remaining_budget:.2f} remaining")
        
        # International travel check
        domestic_cities = ['kuala lumpur', 'kl', 'johor bahru', 'penang', 'sabah', 'sarawak']
        is_international = destination.lower() not in domestic_cities and 'malaysia' not in destination.lower()
        
        if is_international and estimated_cost > 8000:
            warnings.append("International travel requires additional approvals")
        
        # 3. Determine approval level with department-specific rules
        # Department-specific rules (MYR amounts)
        department_rules = {
            'Sales': {'threshold': 12000, 'requires_director': True},
            'Information Technology': {'threshold': 8000, 'requires_director': False},
            'Engineering': {'threshold': 8000, 'requires_director': False},
            'Marketing': {'threshold': 10000, 'requires_director': True},
            'Finance': {'threshold': 6000, 'requires_director': True},
            'Human Resources': {'threshold': 6000, 'requires_director': False}
        }
        
        dept_rule = department_rules.get(department, {'threshold': 8000, 'requires_director': True})
        
        # Apply department-specific thresholds
        if estimated_cost <= 3000:
            approval_level = 0  # Auto approval for small amounts only for senior levels
            # But check if user is senior enough for auto approval
            if job_level not in ['Director', 'VP', 'C-Level']:
                approval_level = 1  # Manager approval for non-senior staff
        elif estimated_cost <= dept_rule['threshold']:
            approval_level = 1  # Manager approval within department limit
        elif estimated_cost <= 15000:
            approval_level = 2  # Director approval
        else:
            approval_level = 3  # Senior management approval
        
        # 4. Calculate compliance score
        compliance_score = 100
        if errors:
            compliance_score -= len(errors) * 25
        if warnings:
            compliance_score -= len(warnings) * 10
        
        compliance_score = max(compliance_score, 0)
        
        # 5. Determine final status based on approval level and compliance
        if errors:
            final_status = "❌ REJECTED"
            status_code = "rejected"
        else:
            # Check if request needs manual approval based on approval level
            if approval_level == 0:
                # Auto approval only for senior staff with small amounts
                if job_level in ['Director', 'VP', 'C-Level'] and estimated_cost <= 3000:
                    final_status = "✅ APPROVED"
                    status_code = "approved"
                else:
                    final_status = "⏳ PENDING APPROVAL"
                    status_code = "pending"
            elif approval_level >= 1:
                # Requires manual approval
                approval_types = {
                    1: "Manager",
                    2: "Director", 
                    3: "Senior Management"
                }
                final_status = f"⏳ PENDING {approval_types.get(approval_level, 'APPROVAL')}"
                status_code = "pending"
            else:
                final_status = "✅ APPROVED" if compliance_score >= 80 else "⚠️ NEEDS REVIEW"
                status_code = "approved" if compliance_score >= 80 else "pending"
        
        # 6. Insert travel request into database
        request_id = None
        if status_code in ['approved', 'pending']:
            cursor.execute("""
                INSERT INTO travel_requests 
                (employee_id, destination, departure_date, return_date, purpose, 
                 estimated_cost, status, approval_level_required, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING request_id
            """, (employee_id, destination, departure_date, return_date, purpose, 
                  estimated_cost, status_code, approval_level))
            
            request_id = cursor.fetchone()[0]
            
            # Create approval workflow
            if approval_level > 0:
                # Check if manager_id exists before inserting
                if manager_id:
                    cursor.execute("""
                        INSERT INTO approval_workflows 
                        (request_id, approver_id, approval_level, status, comments, created_at)
                        VALUES (%s, %s, %s, 'pending', %s, NOW())
                    """, (request_id, manager_id, approval_level, 'Awaiting approval'))
                else:
                    # Insert without approver_id if manager_id is null
                    cursor.execute("""
                        INSERT INTO approval_workflows 
                        (request_id, approval_level, status, comments, created_at)
                        VALUES (%s, %s, 'pending', %s, NOW())
                    """, (request_id, approval_level, 'Awaiting approver assignment'))
            
            # Update remaining budget (if auto-approved)
            if status_code == 'approved':
                cursor.execute("""
                    UPDATE employees 
                    SET remaining_budget = remaining_budget - %s 
                    WHERE employee_id = %s
                """, (estimated_cost, employee_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # 7. Generate simplified report
        next_steps = []
        if status_code == 'approved':
            next_steps.append("✅ Request approved - proceed with booking")
        elif status_code == 'pending':
            if approval_level == 1:
                next_steps.append("⏳ Waiting for manager approval")
            elif approval_level == 2:
                next_steps.append("⏳ Waiting for director approval")
            else:
                next_steps.append("⏳ Waiting for senior management approval")
        else:
            next_steps.append("❌ Request rejected - contact HR for assistance")
        
        if is_international and status_code != 'rejected':
            next_steps.append("🌍 International travel - check passport validity")
            next_steps.append("�️ International travel insurance required")
        
        report = f"""
# 🚀 Travel Request Validation Result

## {final_status}
**Compliance Score: {compliance_score}/100** | **Request ID: {request_id or 'N/A'}**

---

## 👤 Employee Information
- **Name:** {first_name} {last_name}
- **Department:** {department}
- **Job Level:** {job_level}

## 💰 Budget Analysis
- **Annual Budget:** RM{annual_budget:,.2f}
- **Remaining Budget:** RM{remaining_budget:,.2f}
- **Request Amount:** RM{estimated_cost:,.2f}
- **Budget Status:** {'✅ Within Budget' if budget_ok else '❌ Exceeds Budget'}
- **Budget Utilization:** {((annual_budget - remaining_budget + estimated_cost) / annual_budget * 100):.1f}%

## 📋 Validation Results"""
        
        if errors:
            report += f"""
### ❌ Issues Found ({len(errors)})
"""
            for error in errors:
                report += f"- {error}\n"
        
        if warnings:
            report += f"""
### ⚠️ Warnings ({len(warnings)})
"""
            for warning in warnings:
                report += f"- {warning}\n"
        
        report += f"""
### 🎯 Next Steps
"""
        for step in next_steps:
            report += f"- {step}\n"
        
        report += f"""
---
### 📋 Trip Details
- **Destination:** {destination}
- **Duration:** {duration} days
- **Purpose:** {purpose}
- **Approval Level Required:** {approval_level}
"""
        
        return report
        
    except Exception as e:
        return f"""
# ❌ System Error

**Error:** {str(e)}
**Employee:** {employee_email}

Please try again or contact system administrator.
"""

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
        # Fallback to smart local search if AI components not available
        if not gemini_model or not policy_collection:
            return smart_policy_search(query, policies)
        
        # Try to search vector database for semantic relevance
        try:
            results = policy_collection.query(
                query_texts=[query],
                n_results=8  # Get more results for comprehensive context
            )
            
            if results['documents'] and results['documents'][0]:
                # Combine vector search results with filtered policies
                vector_policies = results['documents'][0]
                relevant_policy_text = "\n".join([f"• {doc}" for doc in vector_policies])
            else:
                relevant_policy_text = ""
        except:
            relevant_policy_text = ""
        
        # Prepare comprehensive context from filtered policies
        filtered_context = ""
        for i, policy in enumerate(policies[:10], 1):  # Top 10 filtered policies
            filtered_context += f"{i}. **{policy['rule_name']}**: {policy['description']}\n\n"
        
        # Create advanced prompt for detailed, conversational response
        prompt = f"""You are an expert Travel Policy Assistant for a Malaysian enterprise company. Your role is to provide comprehensive, helpful, and actionable guidance about travel policies.

IMPORTANT CURRENCY GUIDELINES:
• ALWAYS use Malaysian Ringgit (RM) for ALL monetary amounts
• NEVER use dollar signs ($) - convert any USD amounts to RM
• Example: Instead of $200, use RM300; Instead of $3000, use RM12000
• All cost limits and budgets should be in RM currency

RESPONSE GUIDELINES:
• Be conversational and professional, like an experienced HR specialist
• Provide specific details from the policies when available
• Use clear formatting with bullet points and sections when helpful
• Include practical examples and scenarios with RM amounts
• If information is incomplete, guide the employee on next steps
• Use appropriate emojis to make responses engaging but professional
• Always end with helpful suggestions for follow-up questions

EMPLOYEE QUESTION: "{query}"

RELEVANT COMPANY TRAVEL POLICIES:
{filtered_context}

ADDITIONAL SEMANTIC SEARCH RESULTS:
{relevant_policy_text}

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Relevant policy details with specific limits/requirements (in RM)
3. Practical examples or scenarios (using RM amounts)
4. Any important exceptions or special cases
5. Next steps or who to contact for additional help
6. Suggested follow-up questions

Please provide a comprehensive, well-structured response that fully addresses the employee's question based on the company policies above. Remember to use only RM currency throughout your response."""

        # Generate detailed response with Gemini
        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            # Format the AI response with proper structure
            ai_response = response.text.strip()
            
            # 🇲🇾 Ensure all currency displays use RM
            ai_response = ensure_rm_currency(ai_response)
            
            # Add helpful footer
            footer = f"\n\n---\n💡 **Need more help?**\n• Ask about specific cost limits\n• Inquire about approval processes\n• Request examples for your situation\n• Contact HR at hr@company.com for policy clarifications"
            
            return f"🤖 **AI Travel Policy Assistant:**\n\n{ai_response}{footer}"
        else:
            return smart_policy_search(query, policies)
            
    except Exception as e:
        print(f"AI search error: {e}")
        # Always fall back to local search if AI fails
        return smart_policy_search(query, policies)

def answer_policy_question(question):
    """Enhanced Policy Q&A function using intelligent database search and AI assistant"""
    try:
        if not question or not question.strip():
            return "❓ Please ask me a question about travel policies. For example: 'What are the hotel cost limits?' or 'When can I book business class?'"
        
        # Load policies from database
        policies = load_policies()
        if not policies:
            return "❌ No policies found in database. Please contact your HR department."
        
        # Enhanced keyword-based policy filtering
        q_lower = question.lower().strip()
        relevant_policies = []
        
        # Advanced keyword mapping for better policy matching
        keyword_categories = {
            'flight': ['flight', 'fly', 'airplane', 'plane', 'air', 'airline', 'business class', 'economy', 'first class', 'cabin', 'seat', 'upgrade'],
            'hotel': ['hotel', 'accommodation', 'room', 'stay', 'lodging', 'night', 'nightly', 'resort', 'hostel', 'motel'],
            'meal': ['meal', 'food', 'dining', 'restaurant', 'eat', 'breakfast', 'lunch', 'dinner', 'allowance', 'per diem', 'diem'],
            'approval': ['approval', 'approve', 'manager', 'supervisor', 'permission', 'authorization', 'sign off', 'director', 'vp'],
            'international': ['international', 'overseas', 'abroad', 'foreign', 'visa', 'passport', 'customs', 'immigration'],
            'transportation': ['transport', 'taxi', 'uber', 'lyft', 'car', 'rental', 'train', 'bus', 'ground'],
            'cost': ['cost', 'price', 'money', 'budget', 'expense', 'limit', 'maximum', 'minimum', 'cheap', 'expensive', 'fee'],
            'emergency': ['emergency', 'urgent', 'crisis', 'immediate', 'asap', 'rush', 'last minute'],
            'conference': ['conference', 'training', 'seminar', 'workshop', 'meeting', 'event', 'summit']
        }
        
        # Find relevant policies using enhanced matching
        question_keywords = q_lower.split()
        matched_categories = set()
        
        # Identify which categories the question relates to
        for category, keywords in keyword_categories.items():
            if any(keyword in q_lower for keyword in keywords):
                matched_categories.add(category)
        
        # Score policies based on relevance
        policy_scores = []
        for policy in policies:
            score = 0
            policy_text = f"{policy.get('rule_name', '')} {policy.get('description', '')}".lower()
            
            # Category matching
            for category in matched_categories:
                if any(keyword in policy_text for keyword in keyword_categories[category]):
                    score += 10
            
            # Direct keyword matching
            for keyword in question_keywords:
                if len(keyword) > 2:  # Ignore very short words
                    if keyword in policy_text:
                        score += 5
            
            # Exact phrase matching
            if q_lower in policy_text:
                score += 15
            
            if score > 0:
                policy_scores.append((policy, score))
        
        # Sort by relevance score and take top matches
        policy_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_policies = [policy for policy, score in policy_scores[:8]]  # Top 8 most relevant
        
        # If no specific matches found, provide intelligent fallback
        if not relevant_policies:
            # Try partial matching for common question patterns
            fallback_patterns = {
                'what': policies[:3],  # General what questions
                'how': policies[:3],   # How-to questions
                'can i': policies[:2], # Permission questions
                'cost': [p for p in policies if any(word in p.get('description', '').lower() 
                        for word in ['cost', 'limit', 'allowance', 'budget'])],
                'approve': [p for p in policies if 'approval' in p.get('description', '').lower()]
            }
            
            for pattern, fallback_policies in fallback_patterns.items():
                if pattern in q_lower and fallback_policies:
                    relevant_policies = fallback_policies[:3]
                    break
            
            if not relevant_policies:
                relevant_policies = policies[:5]  # Ultimate fallback
        
        # Initialize AI components
        gemini_model, policy_collection = initialize_ai_components()
        
        # Use enhanced AI search with filtered policies
        return ai_policy_search(question, relevant_policies, gemini_model, policy_collection)
        
    except Exception as e:
        return f"🔧 Service temporarily unavailable: {str(e)}\n\n💡 **Tip:** Try rephrasing your question or contact HR for assistance."

# --- APPROVAL WORKFLOW SYSTEM ---
# --- TRAVEL PLANNING SYSTEM ---
def get_travel_information(destination, departure_date, return_date):
    """Get comprehensive travel information for destination"""
    travel_info = {
        'destination_info': {},
        'travel_advisories': {},
        'currency_info': {},
        'time_zone': {},
        'estimated_costs': {}
    }
    
    try:
        # Destination information
        travel_info['destination_info'] = get_destination_info(destination)
        
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

def generate_ai_travel_plan(gemini_model, destination, departure_date, return_date, estimated_cost, policies, purpose, travel_class, hotel_preference):
    """Generates an intelligent travel plan using the Gemini AI model, referencing answer_policy_question's logic and enforcing RM currency. Now includes hotel_preference for hotel comparison."""
    if not gemini_model:
        return "Travel Planning AI is currently unavailable."

    # Build a summary of the most relevant policies for context
    policy_summary = ""
    for policy in (policies or [])[:10]:
        policy_summary += f"- {policy.get('rule_name', '')}: {policy.get('description', '')}\n"

    # Compose a comprehensive prompt, inspired by answer_policy_question, now with hotel_preference
    prompt = f"""
You are an expert Corporate Travel Planner for a Malaysian enterprise. Your job is to generate a detailed, policy-compliant travel plan for the following request, strictly using Malaysian Ringgit (RM/MYR) for all monetary values. Do not use dollar signs ($) or any other currency.

---
EMPLOYEE TRAVEL REQUEST:
- Destination: {destination}
- Departure Date: {departure_date.strftime('%Y-%m-%d') if hasattr(departure_date, 'strftime') else departure_date}
- Return Date: {return_date.strftime('%Y-%m-%d') if hasattr(return_date, 'strftime') else return_date}
- Estimated Cost: RM {estimated_cost:,.2f}
- Purpose: {purpose}
- Travel Class: {travel_class}
- Hotel Preference: {hotel_preference}

---
COMPANY TRAVEL POLICIES (summarized):
{policy_summary}

---
RESPONSE GUIDELINES:
1. All prices and cost estimates must be in Malaysian Ringgit (RM/MYR) only.
2. Provide realistic, recent market-based price ranges for flights and hotels (simulate searching AirAsia, Malaysia Airlines, Agoda, Booking.com, etc.).
3. All flight options must match the requested travel class: "{travel_class}". If not possible, explain why and suggest the closest compliant alternatives.
4. For hotels, compare and recommend options based on the user's hotel preference: "{hotel_preference}". If no exact match, show the closest alternatives and explain.
5. Clearly state if any options violate company policy or budget, and explain the reason.
6. For each flight and hotel, show a comparison table with price range, booking website, and a note about the price basis (e.g., "Based on fares in the last 6 months").
7. Summarize travel advisory and visa requirements for Malaysians visiting {destination}, referencing the [Malaysian MFA](https://www.kln.gov.my/) and the official {destination} embassy site.
8. Output must be in clean, readable Markdown. Do not add extra explanations or commentary outside the plan.

---
RESPONSE STRUCTURE:
1. *Suggested Flights*: Table with airline, stops, class, price range (RM), booking website, and price basis.
2. *Suggested Hotels*: Table with hotel name, price per night (RM), booking website, reason/note (including how it matches the user's hotel preference), and price basis.
3. *Travel Advisory & Visa*: Bullet points summarizing requirements for Malaysians.
4. *User Input Recap*: List all input parameters for transparency.
5. *Key Company Policies*: List the most relevant policies.

---
Important: All suggestions must strictly comply with the user's input and company policies, and use only RM currency. If no matching options are available, explain clearly and recommend the next closest compliant alternatives.

Output only the complete Markdown plan as described above.
"""

    try:
        response = gemini_model.generate_content(prompt)
        # Ensure all currency displays use RM (reuse ensure_rm_currency)
        if hasattr(response, 'text'):
            ai_response = response.text.strip()
        else:
            ai_response = str(response)
        # Enforce RM currency formatting (in case AI slips)
        ai_response = ensure_rm_currency(ai_response)
        return ai_response
    except Exception as e:
        return f"Could not generate AI travel plan: {str(e)}"

def get_dashboard_stats():
    """Get real-time dashboard statistics from the travel_requests table"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            # Query to get total requests
            total_result = connection.execute(text("SELECT COUNT(*) as total FROM travel_requests"))
            total_requests = total_result.fetchone()[0] if total_result else 0
            
            # Query to get approved requests (handle both upper and lower case)
            approved_result = connection.execute(text("SELECT COUNT(*) as approved FROM travel_requests WHERE LOWER(status) = 'approved'"))
            approved_requests = approved_result.fetchone()[0] if approved_result else 0
            
            # Query to get pending requests (handle both upper and lower case)
            pending_result = connection.execute(text("SELECT COUNT(*) as pending FROM travel_requests WHERE LOWER(status) = 'pending'"))
            pending_requests = pending_result.fetchone()[0] if pending_result else 0
            
            # Query to get recent requests (last 5) - simplified without employee name for now
            recent_query = text("""
            SELECT request_id, destination, estimated_cost, status, created_at
            FROM travel_requests
            ORDER BY created_at DESC
            LIMIT 5
            """)
            recent_result = connection.execute(recent_query)
            recent_requests = [dict(row._mapping) for row in recent_result]
            
            return {
                'total_requests': total_requests,
                'approved_requests': approved_requests,
                'pending_requests': pending_requests,
                'recent_requests': recent_requests or []
            }
        
    except Exception as e:
        print(f"Dashboard stats error: {e}")
        return {
            'total_requests': 0,
            'approved_requests': 0,
            'pending_requests': 0,
            'recent_requests': []
        }

# --- CALENDAR INTEGRATION SYSTEM ---

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

# --- ENHANCED CALENDAR INTEGRATION SYSTEM ---
def create_comprehensive_travel_calendar(destination, departure_date, return_date, purpose, travel_class, hotel_preference, estimated_cost, employee_email):
    """
    Create a comprehensive travel calendar with detailed itinerary
    Compatible with Google Calendar, Outlook, Apple Calendar, and all .ics supporting apps
    """
    try:
        calendar = Calendar()
        calendar.creator = "Enterprise Travel Management System"
        calendar.prodid = "-//Enterprise Travel//Travel Calendar//EN"
        
        # Convert string dates to datetime objects
        if isinstance(departure_date, str):
            dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
        else:
            dep_date = datetime.combine(departure_date, datetime.min.time())
            
        if isinstance(return_date, str):
            ret_date = datetime.strptime(return_date, "%Y-%m-%d") 
        else:
            ret_date = datetime.combine(return_date, datetime.min.time())
        
        duration = (ret_date.date() - dep_date.date()).days
        
        # === 1. MAIN TRAVEL EVENT ===
        main_event = Event()
        main_event.name = f"🏢 Business Travel: {destination}"
        main_event.begin = dep_date.replace(hour=6, minute=0)  # Early morning start
        main_event.end = ret_date.replace(hour=22, minute=0)   # Evening return
        main_event.location = destination
        main_event.description = f"""
📋 BUSINESS TRAVEL DETAILS
═══════════════════════════
🎯 Purpose: {purpose}
✈️ Travel Class: {travel_class}
🏨 Hotel: {hotel_preference}
💰 Budget: RM{estimated_cost:,.2f}
👤 Employee: {employee_email}
📞 Emergency: +1-800-TRAVEL-HELP

🔔 IMPORTANT REMINDERS:
• Passport & travel documents
• Expense tracking app
• Company credit card
• Travel insurance confirmation
• Emergency contact information

💼 BUSINESS OBJECTIVES:
• Review and confirm meeting schedules
• Prepare presentation materials
• Update travel expense reports daily
• Submit final reports within 5 days of return
"""
        
        # Add reminders for main event
        main_event.alarms = [
            DisplayAlarm(trigger=timedelta(days=-7)),
            DisplayAlarm(trigger=timedelta(days=-1)),
            DisplayAlarm(trigger=timedelta(hours=-2))
        ]
        
        calendar.events.add(main_event)
        
        # === 2. DEPARTURE DAY EVENTS ===
        
        # Pre-flight preparation
        prep_event = Event()
        prep_event.name = f"✈️ Departure Preparation - {destination}"
        prep_event.begin = dep_date.replace(hour=4, minute=0)
        prep_event.end = dep_date.replace(hour=6, minute=0)
        prep_event.description = f"""
🧳 DEPARTURE CHECKLIST:
• Check-in online (24h before)
• Pack business attire for {purpose}
• Travel documents & ID
• Phone chargers & adapters
• Business cards & materials
• Company laptop & charger

📱 Final confirmations:
• Flight status check
• Ground transportation
• Hotel reservation
• First day meeting schedule
"""
        prep_event.alarms = [DisplayAlarm(trigger=timedelta(hours=-1))]
        calendar.events.add(prep_event)
        
        # Flight departure (estimated)
        flight_dep = Event()
        flight_dep.name = f"🛫 Flight Departure to {destination}"
        flight_dep.begin = dep_date.replace(hour=8, minute=0)
        flight_dep.end = dep_date.replace(hour=12, minute=0)  # Assume 4-hour flight
        flight_dep.location = "Airport Departure Terminal"
        flight_dep.description = f"""
✈️ FLIGHT INFORMATION:
• Travel Class: {travel_class}
• Estimated Duration: 4 hours
• Destination: {destination}

🎯 In-flight productivity:
• Review meeting materials
• Prepare presentation notes
• Update project status
• Plan daily schedule

📱 Upon Arrival:
• Confirm hotel check-in
• Update arrival status to team
• Set up local transportation
"""
        flight_dep.alarms = [DisplayAlarm(trigger=timedelta(hours=-3))]
        calendar.events.add(flight_dep)
        
        # Hotel check-in
        checkin_event = Event()
        checkin_event.name = f"🏨 Hotel Check-in: {hotel_preference}"
        checkin_event.begin = dep_date.replace(hour=15, minute=0)
        checkin_event.end = dep_date.replace(hour=16, minute=0)
        checkin_event.location = f"{hotel_preference}, {destination}"
        checkin_event.description = f"""
🏨 HOTEL INFORMATION:
• Type: {hotel_preference}
• Location: {destination}
• Check-in: 3:00 PM
• Check-out: {ret_date.strftime('%Y-%m-%d')} 11:00 AM

📋 Check-in Tasks:
• Confirm room amenities
• WiFi setup for business
• Review hotel services
• Plan transportation to meetings

💼 Hotel Business Center:
• Printing services available
• Meeting room bookings
• Concierge assistance
"""
        calendar.events.add(checkin_event)
        
        # === 3. DAILY BUSINESS ACTIVITIES ===
        
        current_date = dep_date.date()
        for day_num in range(duration):
            if day_num == 0:  # Arrival day
                continue
                
            current_day = current_date + timedelta(days=day_num)
            current_datetime = datetime.combine(current_day, datetime.min.time())
            
            # Morning business block
            morning_meeting = Event()
            morning_meeting.name = f"💼 Business Activities - Day {day_num + 1}"
            morning_meeting.begin = current_datetime.replace(hour=9, minute=0)
            morning_meeting.end = current_datetime.replace(hour=12, minute=0)
            morning_meeting.location = destination
            morning_meeting.description = f"""
📅 Day {day_num + 1} - Business Schedule

🎯 PURPOSE: {purpose}

⏰ MORNING BLOCK (9 AM - 12 PM):
• Primary business meetings
• {purpose} activities
• Network building
• Progress review

📋 Daily Tasks:
• Update expense tracking
• Send status update to manager
• Review next day schedule
• Follow up on action items

💼 Evening Tasks:
• Document meeting outcomes
• Update project status
• Prepare tomorrow's materials
"""
            calendar.events.add(morning_meeting)
            
            # Afternoon business block
            afternoon_meeting = Event()
            afternoon_meeting.name = f"🤝 Afternoon Sessions - Day {day_num + 1}"
            afternoon_meeting.begin = current_datetime.replace(hour=14, minute=0)
            afternoon_meeting.end = current_datetime.replace(hour=17, minute=0)
            afternoon_meeting.location = destination
            afternoon_meeting.description = f"""
🕐 AFTERNOON BLOCK (2 PM - 5 PM):
• Follow-up meetings
• {purpose} continuation
• Documentation time
• Strategic discussions

🎯 Key Focus Areas:
• Meeting objectives completion
• Relationship building
• Information gathering
• Next steps planning

📝 End-of-Day Review:
• Meeting notes compilation
• Action items tracking
• Expense recording
• Schedule confirmation for next day
"""
            calendar.events.add(afternoon_meeting)
            
            # Daily expense reminder
            expense_reminder = Event()
            expense_reminder.name = f"💰 Daily Expense Tracking - Day {day_num + 1}"
            expense_reminder.begin = current_datetime.replace(hour=20, minute=0)
            expense_reminder.end = current_datetime.replace(hour=20, minute=30)
            expense_reminder.description = f"""
💰 DAILY EXPENSE TRACKING:

📊 Record today's expenses:
• Meals: breakfast, lunch, dinner
• Transportation: taxi, public transit
• Business entertainment
• Incidental expenses

📱 Expense App Tasks:
• Photo receipts immediately
• Categorize all expenses
• Add business purpose notes
• Submit for pre-approval if needed

💡 Budget Status:
• Total Budget: RM{estimated_cost:,.2f}
• Track daily spending
• Flag any budget concerns
"""
            expense_reminder.alarms = [DisplayAlarm(trigger=timedelta(minutes=-30))]
            calendar.events.add(expense_reminder)
        
        # === 4. DEPARTURE DAY EVENTS ===
        
        # Hotel checkout
        checkout_event = Event()
        checkout_event.name = f"🏨 Hotel Check-out & Final Preparations"
        checkout_event.begin = ret_date.replace(hour=10, minute=0)
        checkout_event.end = ret_date.replace(hour=11, minute=0)
        checkout_event.location = f"{hotel_preference}, {destination}"
        checkout_event.description = f"""
🏨 CHECK-OUT PROCESS:
• Settle hotel bill
• Collect all receipts
• Confirm no items left behind
• Arrange airport transportation

📋 Final Business Tasks:
• Send thank you emails
• Confirm follow-up actions
• Update travel status
• Submit expense receipts

💼 Travel Preparation:
• Pack all materials
• Charge devices for flight
• Check flight status
• Confirm departure gate
"""
        checkout_event.alarms = [DisplayAlarm(trigger=timedelta(minutes=-30))]
        calendar.events.add(checkout_event)
        
        # Return flight
        return_flight = Event()
        return_flight.name = f"🛬 Return Flight from {destination}"
        return_flight.begin = ret_date.replace(hour=14, minute=0)
        return_flight.end = ret_date.replace(hour=18, minute=0)
        return_flight.location = f"{destination} Airport"
        return_flight.description = f"""
✈️ RETURN FLIGHT INFORMATION:
• Departure: {destination}
• Travel Class: {travel_class}
• Estimated Duration: 4 hours

🎯 In-flight Tasks:
• Trip summary notes
• Action items review
• Expense report draft
• Follow-up planning

📱 Post-arrival Tasks:
• Confirm safe arrival
• Submit travel completion report
• Upload expense receipts
• Schedule follow-up meetings
"""
        return_flight.alarms = [DisplayAlarm(trigger=timedelta(hours=-3))]
        calendar.events.add(return_flight)
        
        # === 5. POST-TRAVEL FOLLOW-UP ===
        
        # Next day follow-up
        followup_date = ret_date.date() + timedelta(days=1)
        followup_datetime = datetime.combine(followup_date, datetime.min.time())
        
        followup_event = Event()
        followup_event.name = f"📋 Post-Travel Follow-up: {destination}"
        followup_event.begin = followup_datetime.replace(hour=9, minute=0)
        followup_event.end = followup_datetime.replace(hour=10, minute=0)
        followup_event.description = f"""
📋 POST-TRAVEL TASKS:

💼 Business Follow-up:
• Send thank you emails to contacts
• Share meeting outcomes with team
• Update project status
• Schedule follow-up meetings

💰 Administrative Tasks:
• Submit final expense report
• Upload all receipts
• Complete travel satisfaction survey
• Update travel profile if needed

📊 Reporting Requirements:
• Trip summary report
• Business outcomes achieved
• Lessons learned
• Recommendations for future trips

⏰ Deadline: Complete within 5 business days
"""
        followup_event.alarms = [DisplayAlarm(trigger=timedelta(hours=-1))]
        calendar.events.add(followup_event)
        
        # Final expense submission deadline
        expense_deadline = followup_date + timedelta(days=4)
        expense_deadline_datetime = datetime.combine(expense_deadline, datetime.min.time())
        
        expense_deadline_event = Event()
        expense_deadline_event.name = f"💰 Expense Report Deadline - {destination} Trip"
        expense_deadline_event.begin = expense_deadline_datetime.replace(hour=17, minute=0)
        expense_deadline_event.end = expense_deadline_datetime.replace(hour=17, minute=30)
        expense_deadline_event.description = f"""
💰 EXPENSE REPORT FINAL SUBMISSION

📊 Final Tasks:
• Review all expense entries
• Ensure all receipts uploaded
• Verify total amounts
• Submit for manager approval

💼 Trip Summary Required:
• Business objectives achieved
• Total expenses: RM{estimated_cost:,.2f} budgeted
• ROI and business value
• Future recommendations

⚠️ IMPORTANT: Expense reports must be submitted within 5 business days of return
"""
        expense_deadline_event.alarms = [
            DisplayAlarm(trigger=timedelta(days=-1)),
            DisplayAlarm(trigger=timedelta(hours=-2))
        ]
        calendar.events.add(expense_deadline_event)
        
        # === 6. SAVE CALENDAR FILE ===
        calendar_filename = f"travel_itinerary_{destination.replace(' ', '_')}_{dep_date.strftime('%Y%m%d')}.ics"
        calendar_path = f"./data/{calendar_filename}"
        
        # Ensure data directory exists
        import os
        os.makedirs("./data", exist_ok=True)
        
        with open(calendar_path, "w", encoding='utf-8') as f:
            f.write(str(calendar))
        
        # Count total events created
        event_count = len(calendar.events)
        
        return calendar_filename, event_count, calendar_path
        
    except Exception as e:
        print(f"Calendar creation error: {e}")
        return None, 0, None

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
            "content": "👋 **Welcome to your AI Travel Policy Assistant!**\n\nI'm here to help you navigate company travel policies with intelligent, detailed responses. I can assist you with:\n\n**🏨 Accommodation Policies:**\n• Hotel cost limits and standards\n• Extended stay arrangements\n• Booking requirements and corporate rates\n\n**✈️ Flight & Transportation:**\n• Flight class eligibility (economy vs business)\n• International travel requirements\n• Ground transportation rules\n\n**💰 Expense Management:**\n• Daily meal allowances and per diem\n• Cost thresholds and budget limits\n• Expense reporting procedures\n\n**📋 Approval Workflows:**\n• Who needs to approve your travel\n• Required documentation\n• Emergency travel procedures\n\n**🌍 International Travel:**\n• Visa and passport requirements\n• Health and safety protocols\n• Special approval processes\n\n💬 **Just ask me anything!** Try questions like:\n• \"What's the hotel limit for New York?\"\n• \"When can I book business class?\"\n• \"Who approves international travel?\"\n• \"What's the meal allowance policy?\"\n\nI'll search our policy database and provide detailed, actionable guidance! 🚀",
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
    
    # Get real-time dashboard statistics
    dashboard_stats = get_dashboard_stats()
    
    # Quick Stats from Database - Only show Total Requests, Approved, and Pending
    st.markdown("### 📈 Quick Stats")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Requests", dashboard_stats['total_requests'])
    with col4:
        st.metric("Approved", dashboard_stats['approved_requests'])
    
    # Show Pending in full width
    st.metric("Pending", dashboard_stats['pending_requests'])
    
    # Recent Requests from Database (simplified)
    if dashboard_stats['recent_requests']:
        st.markdown("### 📋 Recent Requests")
        for request in dashboard_stats['recent_requests']:
            with st.expander(f"Request #{request.get('request_id', 'N/A')}"):
                st.write(f"**Destination:** {request.get('destination', 'N/A')}")
                st.write(f"**Cost:** RM{float(request.get('estimated_cost', 0)):,.0f}")
                st.write(f"**Status:** {request.get('status', 'Unknown')}")
                created_at = request.get('created_at')
                if created_at:
                    st.write(f"**Date:** {created_at.strftime('%Y-%m-%d %H:%M') if hasattr(created_at, 'strftime') else str(created_at)}")
    
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
            "Max RM250/night domestic",
            "Max RM400/night international"
        ],
        "💰 Budget Limits": [
            "Standard employee: RM15K/year",
            "Manager: RM20K/year",
            "Executive: RM30K/year"
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

# TAB 1: CHATGPT-STYLE POLICY Q&A INTERFACE
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
    st.markdown("### 🚀 Popular Policy Questions")
    st.markdown("*Click any button below for instant answers to common travel policy questions:*")
    
    # First Row - Essential Policy Questions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏨 Hotel Cost Limits", key="hotel_quick_chat"):
            user_msg = "What are the hotel cost limits? How much can I spend per night on accommodation in different cities?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching hotel policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col2:
        if st.button("✈️ Flight Class Rules", key="flight_quick_chat"):
            user_msg = "When am I eligible for business class? What are the flight class policies for domestic vs international travel?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching flight policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col3:
        if st.button("🍽️ Meal Allowances", key="meal_quick_chat"):
            user_msg = "What are the daily meal allowance limits? How does per diem work for domestic vs international travel?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching meal policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col4:
        if st.button("🌍 International Travel", key="international_quick_chat"):
            user_msg = "What special requirements and approvals are needed for international business travel? What documents do I need?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching international policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    # Second Row - Practical Scenario Questions
    st.markdown("#### 💼 Approval & Process Questions")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("📋 Who Approves Travel?", key="approval_quick_chat"):
            user_msg = "Who needs to approve my travel request? What's the approval workflow for different cost levels?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching approval policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col6:
        if st.button("💰 Cost Thresholds", key="cost_quick_chat"):
            user_msg = "What are the budget limits and cost thresholds for business travel? When do I need special approval?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching cost policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col7:
        if st.button("⏰ Booking Requirements", key="booking_quick_chat"):
            user_msg = "How far in advance should I book travel? What are the policies for last-minute or emergency travel?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching booking policies..."):
                response = answer_policy_question(user_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            st.rerun()
    
    with col8:
        if st.button("🚗 Transportation Policy", key="transport_quick_chat"):
            user_msg = "What are the policies for ground transportation? Can I use rideshare, taxis, or rental cars?"
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": user_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            with st.spinner("🤖 Searching transport policies..."):
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
    
    # === ENHANCED HELP SECTION ===
    with st.expander("💡 **How to Get the Best Answers from Your AI Assistant**"):
        st.markdown("""
        **🎯 For Best Results, Try These Question Types:**
        
        **✅ Specific Policy Questions:**
        • "What's the hotel limit for London?"
        • "Can I book business class to Tokyo?"
        • "What documents do I need for China travel?"
        
        **✅ Scenario-Based Questions:**
        • "I'm traveling to NYC for 3 days, what's my meal budget?"
        • "My client meeting got moved to next week, can I change my flight?"
        • "I need to attend an emergency meeting in Dubai, what approvals do I need?"
        
        **✅ Process Questions:**
        • "Who approves travel over RM15000?"
        • "How do I submit an expense report?"
        • "What's the emergency travel procedure?"
        
        **📝 Tips for Better Answers:**
        • Be specific about destinations, dates, and purpose
        • Mention your job level or department if relevant
        • Ask follow-up questions for clarification
        • Use natural language - no need for formal phrasing
        
        **🚀 Advanced Features:**
        • The AI searches through 15+ comprehensive company policies
        • Vector database provides semantic search for related topics
        • Responses include practical examples and next steps
        • All answers are based on your company's actual travel policies
        """)
    
    # Chat Input Interface
    st.markdown("### ✍️ Ask Your Question")
    st.markdown("*Type any travel policy question below. The AI will search our policy database and provide detailed guidance.*")
    
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
    
    # Enhanced Chat Statistics and Analytics
    if len(st.session_state.chat_messages) > 1:
        with st.expander(f"📊 Session Analytics ({len(st.session_state.chat_messages)-1} messages exchanged)"):
            user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Questions Asked", len(user_messages))
            with col2:
                st.metric("AI Responses", len(assistant_messages)-1)  # Exclude welcome message
            with col3:
                if user_messages:
                    first_msg_time = user_messages[0]["timestamp"]
                    st.metric("Session Started", first_msg_time)
            with col4:
                # Calculate average response quality indicator
                avg_response_length = sum(len(msg["content"]) for msg in assistant_messages[1:]) / max(len(assistant_messages)-1, 1)
                quality_score = "High" if avg_response_length > 500 else "Standard"
                st.metric("Response Detail", quality_score)
            
            # Policy topic analysis
            if len(user_messages) > 0:
                st.markdown("**📈 Topics Discussed This Session:**")
                topic_keywords = {
                    '🏨 Hotel': ['hotel', 'accommodation', 'room', 'stay'],
                    '✈️ Flight': ['flight', 'plane', 'business class', 'economy'],
                    '🍽️ Meals': ['meal', 'food', 'allowance', 'per diem'],
                    '💰 Costs': ['cost', 'budget', 'limit', 'money', 'price'],
                    '📋 Approval': ['approval', 'approve', 'manager', 'director'],
                    '🌍 International': ['international', 'overseas', 'visa', 'passport']
                }
                
                topics_discussed = []
                all_user_text = ' '.join([msg["content"].lower() for msg in user_messages])
                
                for topic, keywords in topic_keywords.items():
                    if any(keyword in all_user_text for keyword in keywords):
                        topics_discussed.append(topic)
                
                if topics_discussed:
                    st.write("• " + " • ".join(topics_discussed))
                else:
                    st.write("• General travel policy inquiries")
    
    # Suggested follow-up questions based on chat history
    if len(st.session_state.chat_messages) > 3:  # After some conversation
        with st.expander("🔮 **Suggested Follow-up Questions**"):
            follow_up_suggestions = [
                "What documentation do I need for my trip?",
                "Are there any special requirements for my destination?",
                "How do I submit expenses after my trip?",
                "What's the cancellation policy if plans change?",
                "Can I extend my trip for personal reasons?",
                "What's covered by travel insurance?",
                "How do I handle currency exchange?",
                "What if I need to travel on weekends or holidays?"
            ]
            
            st.markdown("**💡 Consider asking about:**")
            for i, suggestion in enumerate(follow_up_suggestions[:4]):  # Show 4 suggestions
                if st.button(f"❓ {suggestion}", key=f"follow_up_{i}"):
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": suggestion,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    with st.spinner("🤖 Generating detailed response..."):
                        response = answer_policy_question(suggestion)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                    st.rerun()
# TAB 2: ENHANCED TRAVEL REQUEST SUBMISSION
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
    <strong>🎯 Streamlined Travel Request Workflow:</strong><br>
    ✅ <strong>Submit Travel Request:</strong> Enter travel details<br>
    ✅ <strong>AI Validates Against Policies:</strong> Instant policy compliance check<br>
    ✅ <strong>Routes for Approval:</strong> Automatic approval routing<br>
    ✅ <strong>Creates Calendar Events:</strong> Generate travel itinerary<br>
    ✅ <strong>Sends Confirmations:</strong> Complete request processing
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
            estimated_cost = st.number_input("💰 Estimated Total Cost (RM)", min_value=0.0, step=100.0, value=3000.0, help="Total estimated cost including flights, hotels, meals, and transportation")
            
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
                st.info(f"Daily rate: RM{daily_cost:.0f}")
                
                if estimated_cost > 5000:
                    st.error(f"⚠️ HIGH COST: RM{estimated_cost:,.0f}")
                    st.caption("VP approval required")
                elif estimated_cost > 3000:
                    st.warning(f"💰 MEDIUM: RM{estimated_cost:,.0f}")
                    st.caption("Director approval required")
                else:
                    st.success(f"✅ STANDARD: RM{estimated_cost:,.0f}")
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
                use_container_width=True,
                help="Get a quick AI analysis preview without submitting the full request"
            )

    # Calendar generation section outside the form
    st.markdown("### 📅 **Travel Calendar Generation**")
    col_cal1, col_cal2 = st.columns([1, 1])

    with col_cal1:
        if st.button("Generate Travel Calendar", type="secondary", use_container_width=True, key="cal_gen"):
            # Check if minimum required fields are filled
            if destination and departure_date and return_date and employee_email:
                try:
                    result = create_comprehensive_travel_calendar(
                        destination=destination,
                        departure_date=departure_date,
                        return_date=return_date,
                        purpose=purpose or "Business Travel",
                        travel_class=travel_class,
                        hotel_preference=hotel_preference or "3-star",
                        estimated_cost=estimated_cost or 0,
                        employee_email=employee_email
                    )
                    
                    if result and result[0]:  # Check if function returned valid results
                        calendar_filename, event_count, calendar_path = result
                        
                        # Read the calendar content from the file
                        with open(calendar_path, 'r', encoding='utf-8') as f:
                            calendar_content = f.read()
                        
                        st.success(f"✅ Travel calendar generated successfully! ({event_count} events created)")
                        st.session_state['calendar_content'] = calendar_content
                        st.session_state['calendar_filename'] = calendar_filename
                    else:
                        st.error("Failed to generate calendar")
                except Exception as e:
                    st.error(f"Calendar generation error: {str(e)}")
            else:
                st.warning("Please fill in Destination, Departure Date, Return Date, and Employee Email to generate calendar")

    with col_cal2:
        # Show download button if calendar was generated
        if 'calendar_content' in st.session_state and 'calendar_filename' in st.session_state:
            st.download_button(
                label="📥 Download Calendar (.ics)",
                data=st.session_state['calendar_content'],
                file_name=st.session_state['calendar_filename'],
                mime="text/calendar",
                use_container_width=True,
                help="Download your travel calendar to import into Google Calendar, Outlook, or Apple Calendar",
                key="cal_download"
            )

    if 'calendar_content' in st.session_state:
        st.info("💡 **Import Instructions:** After downloading, open the .ics file or import it directly into your calendar app:\n"
               "• **Google Calendar:** Settings → Import & Export → Import\n"
               "• **Outlook:** File → Import\n" 
               "• **Apple Calendar:** File → Import")

    # Preview section
    if preview_button:
        if all([destination, departure_date, return_date, estimated_cost, purpose, employee_email]):
            if return_date <= departure_date:
                st.error("❌ Return date must be after departure date")
            else:
                with st.spinner("🔍 AI Preview Analysis..."):
                    # Load policies and AI components
                    policies_cache = load_policies()
                    gemini_model, _ = initialize_ai_components()
                    
                    # Run simplified AI validation
                    validation_report = ai_validate_travel_request(
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
                    
                    preview_content = f"""
### 🔍 SIMPLIFIED AI VALIDATION PREVIEW

{validation_report}

###  Preview Mode Notice
This is a preview analysis. Submit the form to get the complete workflow:
**Submit travel request → AI validates → Routes for approval → Creates calendar → Sends confirmations**
"""
                    
                    # Display the simplified preview
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
                    
                    # Run simplified AI validation
                    validation_result = ai_validate_travel_request(
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
                    
                    # Display simplified validation results
                    st.markdown(validation_result)
                    
                    # Check if request was rejected OR pending - if so, stop here
                    if "❌ REJECTED" in validation_result or "⏳ PENDING" in validation_result:
                        if "❌ REJECTED" in validation_result:
                            st.error("🚫 **Request processing stopped due to rejection.**")
                            st.info("📝 Please review the issues above and submit a corrected request.")
                        else:
                            st.warning("⏳ **Request is pending approval - processing paused.**")
                            st.info("📋 Your request has been submitted for approval. You will be notified when approved.")
                            st.info("💡 **Next Steps:** Wait for manager/director approval before proceeding with travel planning.")
                        st.stop()  # Stop execution here for both rejected and pending requests
                    
                    # === AI TRAVEL PLANNING ===
                    st.write("🤖 **Generating AI Travel Plan...**")
                    
                    # Generate AI travel plan
                    ai_travel_plan = generate_ai_travel_plan(
                        gemini_model=gemini_model,
                        destination=destination,
                        departure_date=departure_date,
                        return_date=return_date,
                        estimated_cost=estimated_cost,
                        policies=policies_cache,
                        purpose=purpose,
                        travel_class=travel_class,
                        hotel_preference=hotel_preference
                    )
                    
                    # Display AI travel plan
                    st.markdown("### ✈️ AI-Generated Travel Plan")
                    st.markdown("*The following information is for reference only.*")
                    st.markdown(ai_travel_plan)
                    
                    # === DETAILED TRAVEL INFORMATION (Requirement 4) ===
                    st.write("🌍 **Generating Detailed Travel Information...**")
                    
                    # Get comprehensive travel information
                    travel_info = get_travel_information(destination, departure_date, return_date)
                    
                    # Display detailed travel information in organized sections
                    st.markdown("### 📋 *Comprehensive Travel Information*")
                    
                    # Destination Information
                    with st.expander("🌍 **Destination Information**", expanded=True):
                        dest_info = travel_info['destination_info']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Country:** {dest_info.get('country', 'N/A')}")
                            st.write(f"**Language:** {dest_info.get('language', 'N/A')}")
                            st.write(f"**Currency:** {dest_info.get('currency', 'N/A')}")
                        with col2:
                            st.write(f"**Business Hours:** {dest_info.get('business_hours', 'N/A')}")
                            st.write(f"**Transportation:** {dest_info.get('transportation', 'N/A')}")
                        st.write(f"**Cultural Notes:** {dest_info.get('cultural_notes', 'N/A')}")
                        st.write(f"**Business Districts:** {dest_info.get('business_districts', 'N/A')}")
                    
                    # Travel Advisories
                    with st.expander("🛡️ **Travel Advisories & Safety**"):
                        advisories = travel_info['travel_advisories']
                        st.write(f"**Safety Level:** {advisories.get('safety_level', 'N/A')}")
                        st.write(f"**Health Requirements:** {advisories.get('health_requirements', 'N/A')}")
                        st.write(f"**Documentation:** {advisories.get('documentation', 'N/A')}")
                        st.write(f"**COVID Restrictions:** {advisories.get('covid_restrictions', 'N/A')}")
                        
                        emergency = advisories.get('emergency_contacts', {})
                        st.write("**Emergency Contacts:**")
                        st.write(f"• Local Emergency: {emergency.get('local_emergency', 'N/A')}")
                        st.write(f"• Embassy: {emergency.get('embassy', 'N/A')}")
                    
                    # Cost Estimates
                    with st.expander("💰 **Detailed Cost Breakdown**"):
                        costs = travel_info['estimated_costs']
                        duration = costs.get('duration_days', 1)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accommodation", f"RM{costs.get('accommodation', 0):.0f}")
                            st.metric("Meals", f"RM{costs.get('meals', 0):.0f}")
                        with col2:
                            st.metric("Local Transport", f"RM{costs.get('local_transport', 0):.0f}")
                            st.metric("Total Estimate", f"RM{costs.get('total_estimate', 0):.0f}")
                        
                        st.write(f"**Cost Tier:** {costs.get('cost_tier', 'N/A')}")
                        st.write(f"**Duration:** {duration} days")
                        
                        daily = costs.get('daily_breakdown', {})
                        if daily:
                            st.write("**Daily Breakdown:**")
                            st.write(f"• Hotel: RM{daily.get('hotel', 0)}/night")
                            st.write(f"• Meals: RM{daily.get('meals', 0)}/day")
                            st.write(f"• Transport: RM{daily.get('transport', 0)}/day")
                    
                    # Currency & Banking
                    with st.expander("💱 **Currency & Banking Information**"):
                        currency = travel_info['currency_info']
                        st.write(f"**Local Currency:** {currency.get('local_currency', 'N/A')}")
                        st.write(f"**Exchange Rate:** {currency.get('exchange_rate', 'N/A')}")
                        st.write(f"**Payment Methods:** {currency.get('payment_methods', 'N/A')}")
                        st.write(f"**Cash Recommendations:** {currency.get('cash_recommendations', 'N/A')}")
                        st.write(f"**Banking:** {currency.get('banking', 'N/A')}")
                    
                    # Time Zone Information
                    with st.expander("🕐 **Time Zone & Schedule Information**"):
                        timezone = travel_info['time_zone']
                        st.write(f"**Local Time:** {timezone.get('local_time', 'N/A')}")
                        st.write(f"**Time Difference:** {timezone.get('time_difference', 'N/A')}")
                        st.write(f"**Business Hours (Local):** {timezone.get('business_hours_local', 'N/A')}")
                        
                        if timezone.get('jet_lag_tips'):
                            st.write("**Jet Lag Tips:**")
                            for tip in timezone['jet_lag_tips']:
                                st.write(f"• {tip}")
                    
                    # Check if request was approved for calendar creation
                    if "✅ APPROVED" in validation_result or "Request ID:" in validation_result:
                        st.write("📅 **Generating travel calendar...**")
                        try:
                            # Create comprehensive travel calendar
                            result = create_comprehensive_travel_calendar(
                                destination=destination,
                                departure_date=departure_date,
                                return_date=return_date,
                                purpose=purpose or "Business Travel",
                                travel_class=travel_class,
                                hotel_preference=hotel_preference or "3-star",
                                estimated_cost=estimated_cost or 0,
                                employee_email=employee_email
                            )
                            
                            if result and result[0]:  # Check if function returned valid results
                                calendar_filename, event_count, calendar_path = result
                                
                                # Read the calendar content from the file
                                with open(calendar_path, 'r', encoding='utf-8') as f:
                                    calendar_content = f.read()
                                
                                st.success(f"✅ Travel calendar generated successfully! ({event_count} events created)")
                                st.session_state['calendar_content'] = calendar_content
                                st.session_state['calendar_filename'] = calendar_filename
                            else:
                                st.error("Failed to generate calendar")
                        except Exception as cal_e:
                            st.warning(f"Calendar generation failed: {str(cal_e)}")
                    
                    st.write("✅ **Travel request processed successfully!**")
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    
        else:
            st.error("❌ Please fill in all required fields")

# End of application
