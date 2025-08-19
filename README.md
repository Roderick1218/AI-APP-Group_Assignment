# Intelligent Travel Policy Advisor

## Project Overview

An AI-powered intelligent travel policy management system built with Google Gemini and Streamlit. The system provides:

1. **Policy Q&A**: Answer questions about company travel policies
2. **Request Validation**: Automatically validate employee travel requests against company policies  
3. **Calendar Integration**: Auto-generate calendar event files for approved travel requests

## Key Features

### 🤖 AI-Driven Policy Q&A
- Vector database storage for policy information
- Natural language query support
- Intelligent responses powered by Google Gemini model

### 📊 Smart Request Validation
- Automatic database queries to verify employee information
- Budget limits and policy compliance checking
- Detailed compliance report generation

### 📅 Automatic Calendar Creation
- Auto-generate .ics calendar files for eligible travel requests
- Download support for personal calendar applications

## Technical Architecture

### Backend Technologies
- **AI Model**: Google Gemini 1.5 Flash
- **Vector Database**: Chroma
- **Relational Database**: SQLite
- **Frameworks**: LangChain, Streamlit

### Database Schema
- `employees`: Employee information table
- `travel_policies`: Travel policies table
- `travel_requests`: Travel requests table
- `approval_workflows`: Approval workflow table

## Installation and Setup

### 1. Environment Requirements
Ensure Python 3.8+ is installed

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Google API Key
Create a `.env` file and add:
```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 4. Initialize Database
```bash
python setup_database.py
```

### 5. Run Application
```bash
streamlit run app.py
```

The application will start at http://localhost:8501

## Usage Guide

### Policy Q&A Tab
1. Enter questions about travel policies in the chat interface
2. AI provides accurate answers based on company policy database

### Request Validation Tab
1. Fill in employee email, destination, dates, and other information
2. Click "Validate and Schedule" button
3. The system will:
   - Verify employee identity
   - Check policy compliance
   - Generate compliance report
   - Create calendar events for eligible requests

## Project Structure
```
├── app.py                 # Main application
├── setup_database.py      # Database initialization script
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables
├── .gitignore           # Git ignore file
└── data/
    ├── travel.db        # SQLite database
    └── chroma_db_policy/ # Vector database
```

## Development Features

### Caching Mechanism
- Uses `@st.cache_resource` to cache AI agents for improved performance
- Smart vector database loading to avoid redundant creation

### Error Handling
- Comprehensive environment variable validation
- Database connection error handling
- User-friendly error messages

### Security
- API key management through environment variables
- Parameterized database queries to prevent SQL injection

## Notes

1. Ensure Google API key is valid with sufficient quota
2. Initial run creates vector database, may take a few minutes
3. Generated .ics files can be imported into most calendar applications

## Troubleshooting

### Common Issues
1. **API Key Error**: Check if the key in `.env` file is correct
2. **Database Error**: Run `python setup_database.py` to reinitialize
3. **Dependency Issues**: Use `pip install -r requirements.txt` to reinstall

### Support
For issues, check terminal output for detailed error messages and ensure all dependencies are properly installed.
