# app_simple.py - Pure Local Search Version (No AI Dependencies)

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine, text

# --- SIMPLE IMPORTS ---
from ics import Calendar, Event

# --- CONFIGURATION ---
load_dotenv()

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
        st.error(f"Database connection failed: {e}")
        st.stop()

DATABASE_URL = get_database_url()
CALENDAR_FILE = "trip_event.ics"

# --- GET POLICIES FROM DATABASE ---
@st.cache_data
def load_policies():
    """Load policies from database and return as serializable dictionaries"""
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
# --- SMART LOCAL SEARCH SYSTEM ---
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
st.title("✈️ Travel Policy Advisor")
st.info("🔍 **Smart Local Search** - No AI dependencies, pure keyword matching")

tab1, tab2 = st.tabs(["Policy Q&A", "Submit Travel Request"])

with tab1:
    st.header("Ask About Travel Policies")
    st.markdown("**💡 Try specific questions like:**")
    st.markdown("- What are the hotel cost limits?")
    st.markdown("- Do I need approval for international travel?")
    st.markdown("- What's the meal allowance?")
    st.markdown("- Flight class rules for long trips?")
    
    # Load policies
    try:
        policies = load_policies()
        st.success(f"✅ Loaded {len(policies)} travel policies")
    except Exception as e:
        st.error(f"❌ Error loading policies: {e}")
        st.stop()
    
    # Chat interface
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])
    
    if prompt := st.chat_input("e.g., What are the hotel booking rules?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                response = smart_policy_search(prompt, policies)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Submit Travel Request")
    st.info("📋 Basic travel request submission with calendar creation")
    
    with st.form("travel_request"):
        col1, col2 = st.columns(2)
        with col1:
            destination = st.text_input("Destination", placeholder="e.g., New York")
            departure_date = st.date_input("Departure Date")
            estimated_cost = st.number_input("Estimated Cost ($)", min_value=0.0, step=50.0)
        with col2:
            return_date = st.date_input("Return Date")
            purpose = st.text_area("Purpose of Travel", placeholder="e.g., Client meeting")
            employee_email = st.text_input("Employee Email", placeholder="your.email@company.com")
        
        submitted = st.form_submit_button("Submit Request")
        
        if submitted:
            if destination and purpose and employee_email:
                st.success("✅ Travel request submitted!")
                st.info(f"""
                **Request Summary:**
                - Destination: {destination}
                - Dates: {departure_date} to {return_date}
                - Cost: ${estimated_cost}
                - Purpose: {purpose}
                
                📞 **Next Steps:** Contact your manager for approval.
                """)
                
                # Create calendar event
                calendar_result = create_calendar_event(
                    destination=destination,
                    departure_date=str(departure_date),
                    return_date=str(return_date),
                    purpose=purpose
                )
                st.info(calendar_result)
            else:
                st.error("Please fill in all required fields.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ System Info")
    st.success("✅ Local Search Active")
    st.info("🚀 No AI dependencies")
    
    st.header("📋 Quick Policy Guide")
    st.markdown("""
    **Flight Costs:** Business class for 6+ hour flights
    **Hotel:** Max $300/night in major cities  
    **Meals:** $75 daily allowance
    **Approval:** Manager approval for international travel
    """)
    
    st.header("🔄 Database")
    if USE_ONLINE_DATABASE:
        st.info("🌐 Online (Prisma)")
    else:
        st.info("🏠 Local (PostgreSQL)")
    
    st.header("💡 Search Tips")
    st.markdown("""
    **Be specific:**
    - "hotel cost limits"
    - "flight class rules" 
    - "meal allowance"
    - "approval requirements"
    """)
    
    # Show all policies
    if st.button("📋 Show All Policies"):
        try:
            policies = load_policies()
            st.markdown("**All Travel Policies:**")
            for i, policy in enumerate(policies, 1):
                st.markdown(f"**{i}. {policy['rule_name']}**")
                st.markdown(f"{policy['description']}")
                st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")
