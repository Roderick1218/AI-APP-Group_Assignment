# app.py (Golden Version - Final)

import os
import sqlite3
import streamlit as st
import shutil
from dotenv import load_dotenv
from datetime import datetime

# --- IMPORTS FOR TOOLS ---
from ics import Calendar, Event
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# --- 0. CONFIGURATION ---
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.error("Error: Please set your Google API Key in a .env file.")
    st.info("Please create a file named '.env' in the project root and add the following line: \n\nGOOGLE_API_KEY=\"YOUR_API_KEY\"")
    st.stop()

DB_FILE = os.path.join("data", "travel.db")
VECTOR_DB_PATH = os.path.join("data", "chroma_db_policy")
CALENDAR_FILE = "trip_event.ics"

# --- 1. TOOL DEFINITION ---

class CalendarToolInputSchema(BaseModel):
    summary: str = Field(description="The title or summary of the calendar event.")
    start_date_str: str = Field(description="The start date of the trip in 'YYYY-MM-DD' format.")
    end_date_str: str = Field(description="The end date of the trip in 'YYYY-MM-DD' format.")
    location: str = Field(description="The destination city of the trip.")

def create_calendar_event(summary: str, start_date_str: str, end_date_str: str, location: str) -> str:
    try:
        c = Calendar()
        e = Event()
        e.name = summary
        e.begin = datetime.strptime(start_date_str, '%Y-%m-%d')
        e.end = datetime.strptime(end_date_str, '%Y-%m-%d')
        e.location = location
        c.events.add(e)
        with open(CALENDAR_FILE, 'w') as f:
            f.write(c.serialize())
        return f"Successfully created calendar event file: {CALENDAR_FILE}"
    except Exception as e:
        return f"Failed to create calendar event file. Error: {e}"

# --- 2. CORE LOGIC (Loading and Caching Agents) ---

@st.cache_resource
def get_qa_chain():
    st.info("Initializing Policy Q&A Agent (using Google Gemini)...")
    
    # Connect to database and get policies
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT rule_name, description FROM travel_policies")
    policies = cursor.fetchall()
    conn.close()
    
    if not policies:
        st.error("No policies found in database! Please run setup_database.py first.")
        st.stop()
    
    st.info(f"Found {len(policies)} policies in database")
    
    # Create more detailed documents from policies
    policy_documents = []
    for name, desc in policies:
        # Create comprehensive content for each policy
        content = f"""Policy: {name}
Description: {desc}
Rule Name: {name}
Details: {desc}
Travel Policy: {name} - {desc}"""
        
        policy_documents.append(Document(
            page_content=content,
            metadata={"source": "database", "rule_name": name, "policy_type": "travel"}
        ))
    
    st.info(f"Created {len(policy_documents)} policy documents for vector database")
    
    # Initialize embeddings
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use the global VECTOR_DB_PATH
    vector_db_path = VECTOR_DB_PATH
    
    # Check if vector database exists and try to load it, if fails recreate
    try:
        if os.path.exists(vector_db_path):
            st.info("Attempting to load existing vector database...")
            vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)
            # Test if database works
            test_results = vector_db.similarity_search("test policy", k=1)
            if len(test_results) == 0:
                raise Exception("Vector database is empty")
            st.info("Successfully loaded existing vector database")
        else:
            raise Exception("Vector database doesn't exist")
    except Exception as e:
        st.info(f"Creating new vector database... ({str(e)})")
        # Create new vector database in a different location first
        temp_path = vector_db_path + "_temp"
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
            except:
                pass
        
        vector_db = Chroma.from_documents(
            documents=policy_documents,
            embedding=embedding_function,
            persist_directory=temp_path
        )
        vector_db.persist()
        
        # Try to replace the old database
        try:
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)
            shutil.move(temp_path, vector_db_path)
            st.info("Vector database created and moved to final location")
        except:
            # If moving fails, just use the temp location
            vector_db_path = temp_path
            st.info("Vector database created in temporary location")
    
    st.info("Vector database ready for use")
    
    # Create retriever with more documents returned
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1,  # Slightly higher for more natural responses
        convert_system_message_to_human=True
    )
    
    # Create QA chain with custom prompt
    from langchain.prompts import PromptTemplate
    
    prompt_template = """You are a helpful travel policy assistant. Use the following policy information to answer questions about travel policies.

Context from travel policies:
{context}

Question: {question}

Instructions:
- Provide specific, helpful answers based on the policy information
- If you find relevant policy information, cite the specific policy name
- If the exact information isn't available, provide what you can from related policies
- Be conversational and helpful
- For meal allowances, refer to the "Meal Per Diem" policy if available

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    st.success("Policy Q&A Agent is ready!")
    return qa_chain

@st.cache_resource
def get_validation_agent():
    st.info("Initializing Request Validation Agent with Calendar Tool...")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    calendar_tool = StructuredTool.from_function(
        func=create_calendar_event,
        name="CalendarEventCreator",
        description="Use this tool to create a calendar event file (.ics) for an approved trip.",
        args_schema=CalendarToolInputSchema
    )
    
    # +++ THE ULTIMATE PROMPT +++
    prefix = """You are a fully automated travel desk agent with multi-step query capabilities. Your entire process is as follows:

            **Step 1: Information Gathering (If Necessary)**
            The user will provide an email address, but many tables use `employee_id`. Your first task is to query the `employees` table to find the corresponding `employee_id` for the user's email. You will use this ID for all subsequent queries.

            **Step 2: Audit the Request**
            Using the `employee_id`, you must now analyze the user's travel request against the database policies.

            **Step 3: Generate the Final Report**
            Next, you must generate a report based on your audit. This report MUST use the following format EXACTLY:

            **Compliance Summary:** [Compliant, Partially Compliant, or Non-Compliant]

            **Explanation:**

            *   **Violations:**
                *   [List each VIOLATED policy. If none, state "No violations found."]
            *   **Passed Policies:**
                *   [List ONLY policies that were explicitly passed. If none, state "No policies were explicitly passed."]
            *   **Uncertain Policies:**
                *   [List policies that could not be evaluated. If none, state "No uncertain policies."]

            **Recommendations:**
            *   [Provide actionable steps.]

            **Step 4: Schedule the Trip (Conditional)**
            If, and only if, the **Compliance Summary** in your Step 3 report is 'Compliant' or 'Partially Compliant', your immediate and final action MUST be to call the `CalendarEventCreator` tool. Extract all necessary arguments from the user's original request.

            Your final output to the user should ONLY be the report from Step 3.
            """
    
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        prefix=prefix,
        extra_tools=[calendar_tool]
    )
    st.success("Request Validation Agent is ready!")
    return agent_executor

# --- 3. STREAMLIT UI ---

st.title("✈️ Intelligent Travel Policy Advisor")
tab1, tab2 = st.tabs(["Policy Q&A", "Submit & Validate Request"])

with tab1:
    # Unchanged
    st.header("Ask a Question About Travel Policies")
    qa_chain = get_qa_chain()
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("e.g., What are the rules for long flights?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(prompt)
                st.markdown(response['result'])
        st.session_state.messages.append({"role": "assistant", "content": response['result']})

with tab2:
    st.header("Submit a New Travel Request for Validation")
    validation_agent = get_validation_agent()

    with st.form("request_form"):
        employee_email = st.text_input("Employee Email", "employee@example.com")
        destination = st.text_input("Destination", "London")
        departure_date = st.date_input("Departure Date")
        return_date = st.date_input("Return Date")
        purpose = st.text_area("Purpose of Travel", "Attend the 2025 Global Tech Summit")
        estimated_cost = st.number_input("Estimated Cost (USD)", min_value=0, value=1200)
        submitted = st.form_submit_button("Validate and Schedule")

    if submitted:
        if os.path.exists(CALENDAR_FILE): os.remove(CALENDAR_FILE)

        request_details = (
            f"Please validate the following travel request and create a calendar event for it:\n"
            f"- Employee Email: '{employee_email}'\n"
            f"- Destination: {destination}\n"
            f"- Start Date: {departure_date.strftime('%Y-%m-%d')}\n"
            f"- End Date: {return_date.strftime('%Y-%m-%d')}\n"
            f"- Purpose: {purpose}\n"
            f"- Estimated Cost: ${estimated_cost}\n"
        )
        st.info("Your request has been submitted. The AI agent is now processing it...")
        
        with st.spinner("Agent is checking database, policies, and creating calendar event..."):
            try:
                response = validation_agent.invoke({"input": request_details})
                st.success("Processing Complete!")
                st.markdown(response['output'])
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")

        if os.path.exists(CALENDAR_FILE):
            st.markdown("---")
            st.subheader("🗓️ Download Calendar Event")
            with open(CALENDAR_FILE, "rb") as file:
                st.download_button(
                    label="Download .ics file",
                    data=file,
                    file_name=CALENDAR_FILE,
                    mime="text/calendar"
                )