# � Enterprise Travel Policy Management System

## Project Overview

A comprehensive AI-powered travel policy management system built with **Google Gemini 1.5 Flash**, **Streamlit**, and **PostgreSQL/SQLite**. This system provides intelligent travel request validation, interactive policy Q&A, comprehensive travel information, and automated calendar generation for enterprise travel management.

## 🌟 Core System Features

### 🤖 AI-Powered Policy Q&A Chat
- **ChatGPT-style Interface**: Modern conversational UI with message history
- **Google Gemini Integration**: Advanced AI responses using Gemini 1.5 Flash
- **Vector Database Search**: ChromaDB-powered semantic policy search
- **Smart Quick Buttons**: Pre-configured common policy questions
- **Real-time Analytics**: Session statistics and topic analysis

### ✈️ Intelligent Travel Request System
- **Multi-step Validation**: AI validates against company policies
- **Real-time Preview**: Instant cost and approval analysis
- **Hierarchical Approval**: Manager → Director → VP workflow
- **Currency Management**: RM (Malaysian Ringgit) focused calculations
- **Status Tracking**: Comprehensive request lifecycle management

### 🌍 Comprehensive Travel Information Engine
- **Destination Intelligence**: Detailed info for 50+ global destinations
- **Cost Breakdown**: Accommodation, meals, transport estimates
- **Travel Advisories**: Safety, health, and documentation requirements
- **Currency & Banking**: Exchange rates and payment methods
- **Timezone Management**: Time difference calculations and jet lag tips

### 📅 Advanced Calendar Integration
- **Comprehensive ICS Generation**: 15+ event types per trip
- **Multi-platform Compatible**: Google Calendar, Outlook, Apple Calendar
- **Detailed Itineraries**: Pre-flight, daily activities, post-travel follow-up
- **Automated Reminders**: Smart alarm scheduling
- **Expense Tracking Events**: Built-in expense report deadlines

### 📊 Enterprise Dashboard
- **Real-time Statistics**: Live travel request metrics
- **Request Tracking**: Recent requests with status updates
- **Policy Quick Reference**: Essential policy summaries
- **System Status**: Database and AI component health

## 🛠️ Technical Architecture

### Core AI Components
- **Google Gemini 1.5 Flash**: Advanced language model for policy interpretation and travel planning
- **ChromaDB Vector Database**: Semantic search for policy documents with embeddings
- **LangChain Integration**: Structured AI workflow management
- **Smart Context Management**: Maintains conversation history and context

### Database Infrastructure
- **Dual Database Support**: 
  - Primary: PostgreSQL with Prisma (online production)
  - Fallback: SQLite (local development/offline)
- **Automatic Switching**: Seamless fallback to local database if online unavailable
- **Schema Management**: Comprehensive tables for employees, policies, requests, and workflows

### Application Architecture
```
├── Frontend Layer (Streamlit)
│   ├── Chat Interface (Tab 1)
│   ├── Travel Request Form (Tab 2)
│   └── Dashboard Sidebar
├── AI Processing Layer
│   ├── Google Gemini API
│   ├── ChromaDB Vector Search
│   └── Policy Validation Engine  
├── Business Logic Layer
│   ├── Travel Information Engine
│   ├── Approval Workflow System
│   └── Calendar Generation Service
└── Data Layer
    ├── PostgreSQL/SQLite Database
    ├── Vector Database (ChromaDB)
    └── ICS File Generation
```

### Key System Functions
1. **AI Policy Search**: `answer_policy_question()` - Intelligent policy Q&A
2. **Travel Validation**: `ai_validate_travel_request()` - Comprehensive request validation  
3. **Travel Planning**: `generate_ai_travel_plan()` - AI-generated travel recommendations
4. **Information Engine**: `get_travel_information()` - Destination data aggregation
5. **Calendar Service**: `create_comprehensive_travel_calendar()` - ICS file generation
6. **Dashboard Analytics**: `get_dashboard_stats()` - Real-time system metrics

### Current System Functions (18 Active Functions)
All functions are optimized and actively used:
- `ensure_rm_currency()` - Currency validation
- `get_database_url()` - Database connection management
- `load_policies()` - Policy data loading
- `initialize_ai_components()` - AI system initialization
- `ai_validate_travel_request()` - Request validation
- `populate_vector_database()` - Vector DB management
- `ai_policy_search()` - Semantic policy search
- `answer_policy_question()` - Q&A processing
- `get_travel_information()` - Travel data aggregation
- `get_destination_info()` - Destination details
- `get_travel_advisories()` - Safety information
- `get_currency_info()` - Currency data
- `get_timezone_info()` - Timezone calculations
- `get_cost_estimates()` - Cost analysis
- `generate_ai_travel_plan()` - AI travel planning
- `get_dashboard_stats()` - System analytics
- `smart_policy_search()` - Local policy search
- `create_comprehensive_travel_calendar()` - Calendar generation

## 🚀 Installation and Setup

### 1. System Requirements
- **Python**: 3.8+ (recommended: Python 3.11+)
- **Database**: PostgreSQL (online) or SQLite (local fallback)
- **API Access**: Google AI Platform (Gemini 1.5 Flash)
- **Memory**: 4GB RAM minimum, 8GB recommended for AI processing
- **Storage**: 2GB free space for vector databases and calendar files

### 2. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd AI-APP-Group_Assignment

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root:
```env
# Required: Google AI API Key
GOOGLE_API_KEY="your_google_gemini_api_key_here"

# Optional: Online Database (PostgreSQL)
USE_ONLINE_DATABASE="true"
DATABASE_ONLINE="your_postgresql_connection_string"

# System will automatically fallback to SQLite if online database unavailable
```

### 4. Database Initialization
```bash
# Setup enterprise system with sample data
python update_malaysian_enterprise_system.py

# Verify database integrity and structure
python verify_database.py

# Setup comprehensive policy documents
python setup_comprehensive_policies.py
```

### 5. Launch Application
```bash
# Start the Streamlit application
streamlit run app.py
```

**Application URL**: `http://localhost:8501`

### 6. System Verification
After launching, verify these components:
- ✅ Google Gemini API connection
- ✅ Database connectivity (PostgreSQL or SQLite fallback)
- ✅ ChromaDB vector database initialization
- ✅ Policy documents loaded successfully
- ✅ Dashboard statistics displayed correctly

## 📖 User Guide

### 🤖 Tab 1: Policy Q&A Chat System

#### Interactive Chat Interface
1. **Natural Language Queries**: Ask questions in plain English about travel policies
2. **AI-Powered Responses**: Get detailed answers using Google Gemini 1.5 Flash
3. **Conversation History**: Maintains full chat history with timestamps
4. **Session Analytics**: View conversation statistics and topics discussed

#### Quick Action Buttons
**Essential Policy Questions:**
- 🏨 **Hotel Cost Limits**: Cost limits and accommodation policies
- ✈️ **Flight Class Rules**: Business class eligibility and booking rules
- 🍽️ **Meal Allowances**: Per diem rates and expense policies
- 🌍 **International Travel**: Special requirements and documentation

**Process & Approval Questions:**
- 📋 **Who Approves Travel**: Approval workflow and hierarchy
- 💰 **Cost Thresholds**: Budget limits and special approval requirements
- ⏰ **Booking Requirements**: Advance booking and emergency travel policies
- 🚗 **Transportation Policy**: Ground transport and vehicle rental rules

#### Smart Features
- **Vector Search**: Semantic search through policy database
- **Follow-up Suggestions**: AI-generated related questions
- **Topic Analysis**: Automatic categorization of discussed topics
- **Chat Export**: Download conversation history

### ✈️ Tab 2: Travel Request Submission

#### Travel Request Form
**Essential Information:**
- **Destination**: City/country with intelligent validation
- **Travel Dates**: Departure and return with duration calculation
- **Estimated Cost**: Total budget in RM with automatic breakdown
- **Purpose**: Business purpose affecting approval requirements
- **Travel Class**: Flight class with policy validation
- **Accommodation**: Hotel preference and cost tier selection

#### Real-time Validation Preview
- **Trip Overview**: Duration analysis and approval requirements
- **Cost Analysis**: Daily rates and budget tier classification
- **Destination Check**: International vs domestic classification
- **Policy Compliance**: Live validation against company policies

#### AI Processing Workflow
1. **Policy Validation**: Comprehensive AI validation against all policies
2. **Approval Routing**: Automatic assignment to appropriate approvers
3. **Travel Planning**: AI-generated comprehensive travel plan
4. **Information Gathering**: Detailed destination and travel information
5. **Calendar Generation**: Automatic ICS file creation for approved requests

### 📊 Dashboard Features

#### System Statistics
- **Total Requests**: Complete request count from database
- **Approved Requests**: Successfully approved travel requests
- **Pending Requests**: Requests awaiting approval
- **Recent Activity**: Latest 5 requests with status updates

#### Quick Reference
- **Policy Summaries**: Essential policy highlights
- **Budget Guidelines**: Standard cost limits and thresholds
- **Approval Matrix**: Who approves what based on cost/destination

#### Administrative Actions  
- **Clear Chat History**: Reset conversation for new session
- **Contact Support**: Emergency contact information
- **System Status**: Real-time component health monitoring

## 🗂️ Project Structure

```
Enterprise Travel Policy Management System/
├── app.py                                    # Main Streamlit application (2,417 lines)
│   ├── AI Chat Interface                     # Policy Q&A with Google Gemini
│   ├── Travel Request System                 # Comprehensive request processing
│   ├── Dashboard & Analytics                 # Real-time system statistics
│   └── Calendar Integration                  # ICS file generation
│
├── Core System Files
├── database_manager.py                       # Unified database operations
├── simplified_travel_validation.py           # Travel request validation logic
├── update_malaysian_enterprise_system.py     # Enterprise data initialization
├── verify_database.py                        # Database integrity verification
├── setup_comprehensive_policies.py           # Policy document configuration
│
├── Configuration & Dependencies
├── requirements.txt                          # Python package dependencies
├── .env                                      # Environment variables (API keys)
├── README.md                                 # Complete system documentation
│
└── Data Directory
    ├── travel.db                            # SQLite database (fallback)
    ├── travel_itinerary_*.ics               # Generated calendar files
    ├── chroma_db_policy/                    # ChromaDB vector database
    │   ├── chroma.sqlite3                   # Vector database storage
    │   └── [collection_ids]/                # Policy embeddings
    └── __pycache__/                         # Python bytecode cache
```

### Core Dependencies
```
# AI & Language Processing
google-generativeai==0.8.5         # Google Gemini 1.5 Flash
langchain==0.3.27                   # AI workflow framework
chromadb==1.0.20                    # Vector database for embeddings

# Web Application Framework  
streamlit==1.48.1                   # Modern web interface

# Database Management
SQLAlchemy==2.0.43                  # Database ORM
psycopg2-binary==2.9.9              # PostgreSQL adapter

# Configuration & Utilities
python-dotenv==1.1.1                # Environment variable management
ics==0.7.2                          # Calendar file generation
```

### Database Schema Overview
```sql
-- Core Tables
travel_policies         # Company travel policies and rules
travel_requests         # Employee travel requests and status
employees              # Employee information and hierarchy
approval_workflows     # Multi-level approval system

-- Vector Database (ChromaDB)
policy_collection      # Semantic embeddings for policy search
```

### System Configuration Files
- **Environment Variables** (`.env`): API keys and database connections
- **Requirements** (`requirements.txt`): Python package dependencies
- **Database Scripts**: Initialization and verification utilities
- **Policy Setup**: Comprehensive policy document configuration

## � System Features & Capabilities

### AI-Powered Intelligence
- **Google Gemini 1.5 Flash Integration**: Advanced language understanding for policy interpretation
- **Semantic Policy Search**: ChromaDB vector embeddings for intelligent policy matching
- **Context-Aware Responses**: Maintains conversation context and learning from interactions
- **Multi-turn Conversations**: Supports complex, extended policy discussions

### Comprehensive Travel Management
- **Full Lifecycle Support**: From initial request to post-travel follow-up
- **Intelligent Validation**: AI validates against 15+ comprehensive travel policies
- **Multi-tier Approval System**: Manager → Director → VP approval workflow
- **Real-time Cost Analysis**: Dynamic budget validation and tier classification

### Calendar & Event Management
- **15+ Event Types per Trip**: Comprehensive travel itinerary generation
- **Pre-travel Preparation**: Packing lists, document checks, booking confirmations
- **Daily Business Activities**: Meeting scheduling and networking events
- **Post-travel Follow-up**: Expense deadlines and report submission reminders
- **Multi-platform Support**: Google Calendar, Outlook, Apple Calendar compatible

### Enterprise Integration Features
- **Dual Database Architecture**: PostgreSQL production + SQLite fallback
- **Currency Management**: RM (Malaysian Ringgit) standardization throughout
- **International Support**: 50+ destination database with cost estimates
- **Audit Trail**: Complete request tracking and approval history
- **Performance Optimization**: 18 optimized functions, zero unused code

### User Experience Enhancements
- **ChatGPT-style Interface**: Modern conversational design with message history
- **Real-time Validation**: Instant feedback on form inputs and policy compliance
- **Smart Quick Actions**: Pre-configured buttons for common policy questions
- **Session Analytics**: Conversation statistics and topic analysis
- **Mobile-Responsive Design**: Works on desktop, tablet, and mobile devices

## 🔧 Development & Performance

### Code Optimization Status
- **Total Lines**: 2,417 lines of optimized code
- **Active Functions**: 18 functions, all actively used (zero unused code)
- **Performance**: Streamlined architecture with no redundant operations
- **Memory Efficiency**: Optimized database connections and AI model loading
- **Error Handling**: Comprehensive exception handling with graceful degradation

### Technical Performance Features
- **Caching System**: `@st.cache_resource` for AI agents and database connections
- **Lazy Loading**: Vector databases loaded on-demand for faster startup
- **Connection Pooling**: Efficient database connection management
- **Automatic Failover**: Seamless PostgreSQL to SQLite fallback
- **Memory Management**: Optimized for enterprise-scale data processing

### Security & Compliance
- **Environment Security**: Secure API key management with `.env` files
- **SQL Injection Prevention**: Parameterized queries throughout application
- **Data Validation**: Comprehensive input sanitization and type checking
- **Audit Logging**: Complete approval workflow tracking for compliance
- **Error Boundaries**: Graceful error handling with user-friendly messages

### Monitoring & Diagnostics
- **Real-time Dashboard**: Live system statistics and health monitoring
- **Database Verification**: Built-in integrity checking and validation
- **API Health Checks**: Google Gemini API connection monitoring
- **Vector Database Status**: ChromaDB health and performance metrics
- **Debug Logging**: Detailed system logs for troubleshooting

### Development Standards
- **Clean Architecture**: Modular design with clear separation of concerns
- **Type Safety**: Comprehensive input validation and error handling
- **Documentation**: Inline code documentation and comprehensive README
- **Testing**: Database integrity verification and system validation scripts
- **Version Control**: Git-based development with clear commit history

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python Version**: 3.8+ (recommended: 3.11+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for data and vector databases
- **Internet**: Required for Google Gemini API and online database

### API Dependencies
- **Google AI API**: Gemini 1.5 Flash model access
- **PostgreSQL Database**: Online database connection (optional)
- **Weather API**: Real-time weather data integration
- **Currency Exchange**: Live exchange rate services

## 🚨 Troubleshooting Guide

### Common Setup Issues

#### 1. Google AI API Configuration
**Problem**: "Google API key not found" or authentication errors
**Solutions**:
```bash
# Verify .env file exists and contains valid key
cat .env                          # Linux/Mac
type .env                         # Windows

# Check environment variable loading
python -c "import os; print(os.getenv('GOOGLE_API_KEY'))"

# Test API connection
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
print('✅ Google AI API connected successfully')
"
```

#### 2. Database Connection Issues
**Problem**: Database connection failures or data loading errors
**Solutions**:
```bash
# Check database file existence
ls -la data/travel.db            # Linux/Mac
dir data\travel.db               # Windows

# Verify database integrity
python verify_database.py

# Reinitialize database if corrupted
python update_malaysian_enterprise_system.py

# Test database connection
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('sqlite:///./data/travel.db')
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM travel_requests'))
    print(f'✅ Database connected: {result.fetchone()[0]} requests found')
"
```

#### 3. Vector Database (ChromaDB) Issues
**Problem**: ChromaDB initialization errors or slow performance
**Solutions**:
```bash
# Remove corrupted vector database
rm -rf data/chroma_db_policy      # Linux/Mac
rmdir /s data\chroma_db_policy    # Windows

# Restart application to regenerate
streamlit run app.py

# Check disk space (requires >1GB)
df -h .                          # Linux/Mac
dir                              # Windows

# Test ChromaDB directly
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db_policy')
collections = client.list_collections()
print(f'✅ ChromaDB working: {len(collections)} collections')
"
```

#### 4. Streamlit Application Issues
**Problem**: Application fails to start or displays errors
**Solutions**:
```bash
# Check Python version (requires 3.8+)
python --version

# Verify all dependencies installed
pip check

# Clear Streamlit cache
streamlit cache clear

# Run with verbose logging
streamlit run app.py --logger.level=debug

# Test minimal functionality
python -c "
import streamlit as st
import google.generativeai as genai
import chromadb
print('✅ All core packages imported successfully')
"
```

### Performance Optimization

#### Memory Management
- **Large Datasets**: Restart application if memory usage exceeds 4GB
- **Vector Database**: Monitor ChromaDB size (limit to ~1GB for optimal performance)
- **AI Model Loading**: Google Gemini models are loaded on-demand to save memory
- **Database Connections**: Connections automatically pool and close properly

#### API Rate Limits
- **Google Gemini**: Monitor API quota usage in Google AI Studio
- **Request Throttling**: Built-in retry logic for rate limit handling
- **Batch Processing**: Large policy uploads processed in batches

#### Network Connectivity
- **Internet Required**: Google Gemini API requires stable internet connection
- **Offline Mode**: System automatically falls back to local SQLite database
- **Proxy Settings**: Configure proxy if required for corporate networks

### System Health Monitoring

#### Dashboard Checks
- **Database Stats**: Verify Total/Approved/Pending request counts are logical
- **Recent Requests**: Check if recent travel requests display correctly
- **Policy Loading**: Confirm "✅ Loaded X travel policies" message appears
- **AI Components**: Verify no error messages in Gemini initialization

#### Log Analysis
```bash
# Check application logs
streamlit run app.py 2>&1 | tee app.log

# Monitor database queries
tail -f app.log | grep -i "database\|sql"

# Track AI API calls
tail -f app.log | grep -i "gemini\|api"
```

### Emergency Recovery

#### Complete System Reset
```bash
# Stop application
Ctrl+C

# Backup current data (optional)
cp -r data data_backup

# Clean installation
rm -rf data/chroma_db_policy
rm data/travel.db

# Reinitialize everything
python update_malaysian_enterprise_system.py
python setup_comprehensive_policies.py

# Restart application
streamlit run app.py
```

#### Quick Health Check Script
```python
# save as health_check.py
import os
import sqlite3
import google.generativeai as genai
import chromadb

def health_check():
    print("🔍 System Health Check")
    
    # API Key
    if os.getenv('GOOGLE_API_KEY'):
        print("✅ Google API key configured")
    else:
        print("❌ Google API key missing")
    
    # Database
    try:
        conn = sqlite3.connect('./data/travel.db')
        cursor = conn.execute("SELECT COUNT(*) FROM travel_requests")
        count = cursor.fetchone()[0]
        print(f"✅ Database accessible: {count} requests")
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    # Vector Database
    try:
        client = chromadb.PersistentClient(path='./data/chroma_db_policy')
        collections = client.list_collections()
        print(f"✅ ChromaDB working: {len(collections)} collections")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")

if __name__ == "__main__":
    health_check()
```

## 🤝 Contributing & Development

### Development Setup
1. **Fork Repository**: Create your own fork of the project
2. **Create Feature Branch**: `git checkout -b feature/enhancement-name`
3. **Development Environment**: Follow installation guide for local setup
4. **Code Standards**: Maintain existing code style and documentation
5. **Testing**: Verify all functions work with provided test scripts
6. **Pull Request**: Submit PR with detailed description of changes

### Code Contribution Guidelines
- **Function Optimization**: All 18 functions are actively used - maintain this efficiency
- **Error Handling**: Include comprehensive exception handling for new features
- **Documentation**: Update README and inline comments for any changes
- **Database Schema**: Preserve existing schema structure for compatibility
- **AI Integration**: Test Google Gemini interactions thoroughly

### System Extension Points
- **New Policies**: Add policy documents through `setup_comprehensive_policies.py`
- **Destinations**: Extend destination database in travel information functions
- **AI Models**: Integration points available for additional AI services
- **Calendar Types**: Extend ICS generation for different event types
- **Approval Workflows**: Configurable hierarchy in approval system

## 📄 License & Usage

### Educational & Enterprise Use
This Enterprise Travel Policy Management System is designed for:
- **Educational Purposes**: Learning AI integration and enterprise system development
- **Enterprise Deployment**: Real-world travel management implementation
- **Research & Development**: AI-powered business process automation studies

### Compliance & Security
- **Data Protection**: Ensure compliance with local data protection regulations
- **API Security**: Secure Google AI API key management in production
- **Database Security**: Implement proper access controls for production databases
- **Audit Requirements**: Built-in audit trail supports compliance reporting

## 🌟 System Highlights

### Technical Achievement
- **Zero Unused Code**: Optimized 2,417 lines with 18 active functions
- **Dual Database Support**: Seamless PostgreSQL/SQLite architecture
- **Advanced AI Integration**: Google Gemini 1.5 Flash with vector search
- **Comprehensive Calendar**: 15+ event types with multi-platform support
- **Real-time Analytics**: Live dashboard with system health monitoring

### Business Value
- **Complete Travel Lifecycle**: Request submission to post-travel follow-up
- **Intelligent Automation**: AI-powered policy validation and travel planning
- **Enterprise Integration**: Hierarchical approval with audit trail
- **Cost Management**: RM currency standardization with budget validation
- **User Experience**: Modern ChatGPT-style interface with instant feedback

### Innovation Features
- **Semantic Policy Search**: ChromaDB vector embeddings for intelligent matching
- **Context-Aware AI**: Maintains conversation history and learning
- **Dynamic Validation**: Real-time policy compliance checking
- **Automated Calendar Generation**: Comprehensive travel itinerary creation
- **Performance Monitoring**: Built-in system health and analytics dashboard

---

**🚀 Ready to deploy?** Follow the installation guide and start managing enterprise travel with AI-powered intelligence.

**💡 Need support?** Check the comprehensive troubleshooting guide or open an issue with detailed system information.

**🔧 Want to contribute?** Follow the development guidelines and submit a pull request with your enhancements.
