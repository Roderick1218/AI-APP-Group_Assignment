#!/usr/bin/env python3
"""
Test script for comprehensive travel policy Q&A system
"""

from app import answer_policy_question, load_policies

def test_comprehensive_qa():
    """Test the AI assistant with various types of questions"""
    
    print("🧪 Testing Comprehensive Travel Policy Q&A System")
    print("=" * 60)
    
    # Load policies to check database
    policies = load_policies()
    print(f"📋 Loaded {len(policies)} policies from database")
    
    # Test questions covering different categories
    test_questions = [
        # Flight questions
        "What are the flight class rules for international travel?",
        "Can I book business class for a 5-hour flight?",
        "What's the advance booking requirement for flights?",
        
        # Hotel questions
        "What are the hotel cost limits for major cities?",
        "Hotel accommodation standards and requirements?",
        "Extended stay policies for long assignments?",
        
        # Meal and dining
        "Daily meal allowance for international travel?",
        "Business entertainment and client dining rules?",
        "Per diem options for extended trips?",
        
        # International travel
        "International travel approval requirements?",
        "Visa and passport requirements for business travel?",
        "Health insurance for overseas assignments?",
        
        # Approval workflows
        "Approval workflow for different cost levels?",
        "Emergency travel approval process?",
        "Self-approval limits for executives?",
        
        # Transportation
        "Ground transportation and taxi policies?",
        "Rental car policies and restrictions?",
        "Airport parking and mileage reimbursement?",
        
        # Expenses
        "Expense reporting requirements and deadlines?",
        "Corporate credit card usage policies?",
        "Receipt requirements for reimbursement?",
        
        # Special circumstances
        "Family emergency travel policies?",
        "Conference and training travel rules?",
        "Environmental and sustainability guidelines?"
    ]
    
    print(f"\n🔍 Testing {len(test_questions)} questions across all policy categories:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        try:
            response = answer_policy_question(question)
            # Check if response is helpful (not just "I don't know")
            if "I don't know" in response or len(response) < 50:
                print(f"❌ Limited response: {response[:100]}...")
            else:
                print(f"✅ Comprehensive response: {len(response)} characters")
                # Show first few lines of response
                lines = response.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        print(f"   {line[:80]}...")
                        break
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)
    
    print("\n🎯 Test completed! Check the results above.")
    print("✅ = Good comprehensive response")
    print("❌ = Limited or error response")

if __name__ == "__main__":
    test_comprehensive_qa()
