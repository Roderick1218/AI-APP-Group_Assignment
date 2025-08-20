#!/usr/bin/env python3
"""
Complete Travel Policy Database Setup
This script creates comprehensive travel policies for the AI assistant
"""

import os
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
USE_ONLINE_DATABASE = os.getenv("USE_ONLINE_DATABASE", "true").lower() == "true"
DATABASE_ONLINE_URL = os.getenv("DATABASE_ONLINE")

def get_database_connection():
    """Get database connection URL"""
    if USE_ONLINE_DATABASE and DATABASE_ONLINE_URL:
        return DATABASE_ONLINE_URL.replace("postgres://", "postgresql://", 1)
    else:
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "travel_db")
        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "0183813235")
        return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def comprehensive_travel_policies():
    """Return comprehensive travel policy dataset"""
    return [
        # Flight Policies
        {
            'rule_name': 'Flight Booking Policy',
            'description': 'Economy class for flights under 6 hours. Business class allowed for flights over 6 hours or for Director level and above. Book at least 14 days in advance for best rates. Emergency travel exceptions require VP approval.'
        },
        {
            'rule_name': 'Flight Class Upgrade Rules',
            'description': 'Business class automatic for C-Level executives on all flights. Directors get business class on flights over 4 hours. Managers get business class on international flights over 8 hours. Premium economy allowed for flights 3-6 hours.'
        },
        {
            'rule_name': 'Flight Change and Cancellation',
            'description': 'Free changes allowed up to 24 hours before departure for business reasons. Cancellation fees covered by company for work-related changes. Personal changes are employee responsibility. Refundable tickets required for uncertain travel dates.'
        },
        
        # Hotel Policies
        {
            'rule_name': 'Hotel Accommodation Standards',
            'description': 'Standard business hotels up to $300/night in major cities (NYC, London, Tokyo, Dubai). $200/night limit for other destinations. 4-star or equivalent quality required. Extended stay discounts should be negotiated for stays over 7 days.'
        },
        {
            'rule_name': 'Hotel Booking Requirements',
            'description': 'Corporate rates must be used when available. Book through approved travel agency or company portal. Receipt required for all hotel expenses. Room service and minibar charges are not reimbursed unless for business entertainment.'
        },
        {
            'rule_name': 'Extended Stay Policy',
            'description': 'Stays over 30 days qualify for monthly corporate housing rates. Apartment-style accommodations preferred for stays over 14 days. Kitchen facilities encouraged for long-term stays. Laundry service covered for stays over 7 days.'
        },
        
        # Meal and Expense Policies
        {
            'rule_name': 'Daily Meal Allowance',
            'description': 'Daily meal allowance: $75 for domestic travel, $100 for international travel, $125 for high-cost cities (NYC, London, Tokyo, Zurich). Receipts required for all meals over $25. Business dining with clients covered separately with approval.'
        },
        {
            'rule_name': 'Business Entertainment',
            'description': 'Client meals and entertainment require pre-approval for amounts over $150 per person. Alcohol is covered for business dinners only. Team meals limited to $50 per person. Company events and conferences have separate budget allocations.'
        },
        {
            'rule_name': 'Per Diem Alternatives',
            'description': 'Per diem option available for trips over 5 days: $60/day domestic, $80/day international. No receipts required with per diem option. Cannot combine per diem with actual expense reimbursement. Choose option at trip booking.'
        },
        
        # International Travel
        {
            'rule_name': 'International Travel Requirements',
            'description': 'International travel requires VP approval for costs over $3000. Passport must be valid for 6+ months. Company travel insurance mandatory. Check visa requirements 30 days before travel. Emergency contact information must be filed.'
        },
        {
            'rule_name': 'Visa and Documentation',
            'description': 'Company covers all business visa costs. Employee responsible for tourist visa if combining business/personal travel. Work permits required for stays over 90 days. Legal department must approve extended international assignments.'
        },
        {
            'rule_name': 'International Health and Safety',
            'description': 'Health insurance coverage verification required. Vaccinations covered by company when required for business travel. Travel to high-risk countries requires security briefing. 24/7 emergency assistance hotline provided.'
        },
        
        # Approval Workflows
        {
            'rule_name': 'Standard Approval Workflow',
            'description': 'Trips under $2000: Manager approval. $2000-$5000: Director approval. Over $5000: VP approval. Executive level (VP+) can self-approve up to $10000. Board approval required for international travel over $15000.'
        },
        {
            'rule_name': 'Emergency Travel Approval',
            'description': 'Emergency travel can be approved verbally by any Director+ level. Written approval must follow within 24 hours. Emergency defined as: family emergency, critical client situation, or urgent business need. Retroactive approval process available.'
        },
        {
            'rule_name': 'Multi-City Trip Approval',
            'description': 'Multi-city trips require detailed itinerary submission. Each destination must be business-justified. Personal stopovers allowed but at employee expense. Total trip cost determines approval level needed.'
        },
        
        # Transportation
        {
            'rule_name': 'Ground Transportation',
            'description': 'Airport transfers: taxi/rideshare up to $75, or rental car if staying 3+ days. Public transportation encouraged in metro areas. Parking fees at airports covered up to $25/day. Mileage reimbursement for personal vehicle at IRS rate.'
        },
        {
            'rule_name': 'Rental Car Policy',
            'description': 'Mid-size or compact cars only unless business justification for larger vehicle. Rental insurance covered through corporate policy. Fuel costs covered with receipts. GPS/navigation systems covered. Personal use restrictions apply.'
        },
        {
            'rule_name': 'Ride Sharing and Taxis',
            'description': 'Uber/Lyft business profiles must be used when available. Receipts required for all rides over $25. Tipping included in reimbursement up to 20%. Pool/shared rides encouraged for environmental responsibility.'
        },
        
        # Technology and Communication
        {
            'rule_name': 'International Communication',
            'description': 'International phone/data plans covered for business travel over 3 days. Roaming charges covered up to $200 per trip. Company phones required for travel to certain countries. Wi-Fi costs at hotels covered when not complimentary.'
        },
        {
            'rule_name': 'Business Equipment Travel',
            'description': 'Laptop shipping costs covered for trips over 2 weeks. Additional monitor covered for extended stays. Business center fees up to $50/day. Printing and internet at airports covered for business needs.'
        },
        
        # Special Circumstances
        {
            'rule_name': 'Family Emergency Travel',
            'description': 'Immediate family emergencies covered: spouse, children, parents, siblings. Bereavement travel: economy class, standard accommodation. Emergency return trips covered. Flexibility in approval process for family emergencies.'
        },
        {
            'rule_name': 'Conference and Training Travel',
            'description': 'Professional conferences require advance approval and budget allocation. Training courses at external facilities covered including travel. Educational seminars need department head approval. Continuing education credits tracked.'
        },
        {
            'rule_name': 'Client Site Extended Assignments',
            'description': 'Extended client site work (30+ days) has special accommodation rates. Temporary housing allowance available. Monthly travel home covered for assignments over 60 days. Family visit allowance for assignments over 90 days.'
        },
        
        # Expense Reporting
        {
            'rule_name': 'Expense Report Requirements',
            'description': 'All expenses must be reported within 30 days of trip completion. Receipts required for all expenses over $25. Credit card statements accepted for lost receipts under $75. Foreign currency conversion at prevailing rates.'
        },
        {
            'rule_name': 'Corporate Card Usage',
            'description': 'Corporate credit cards mandatory for all business travel. Personal use strictly prohibited. Monthly reconciliation required. Pre-trip spending limits can be increased with approval. Emergency cash advances available.'
        },
        
        # Sustainability and Preferences
        {
            'rule_name': 'Environmental Travel Policy',
            'description': 'Video conferencing should be considered before travel. Carbon offset programs available for frequent travelers. Train travel encouraged for distances under 400 miles. Hybrid/electric rental cars preferred when available.'
        },
        {
            'rule_name': 'Traveler Preferences and Accessibility',
            'description': 'Dietary restrictions and accessibility needs accommodated. Loyalty program memberships encouraged for frequent travelers. Preferred seating arrangements honored when possible. Medical accommodation requests processed priority.'
        }
    ]

def setup_comprehensive_database():
    """Setup comprehensive travel policy database"""
    try:
        # Get database connection
        database_url = get_database_connection()
        engine = create_engine(database_url)
        
        print("🔗 Connecting to database...")
        
        # Get comprehensive policies
        policies = comprehensive_travel_policies()
        
        with engine.connect() as connection:
            # Clear existing policies (optional - comment out to keep existing data)
            print("🗑️ Clearing existing policies...")
            connection.execute(text("DELETE FROM travel_policies"))
            connection.commit()
            
            # Insert comprehensive policies
            print(f"📥 Inserting {len(policies)} comprehensive travel policies...")
            
            for policy in policies:
                connection.execute(
                    text("""
                        INSERT INTO travel_policies (rule_name, description)
                        VALUES (:rule_name, :description)
                    """),
                    {
                        'rule_name': policy['rule_name'],
                        'description': policy['description']
                    }
                )
            
            connection.commit()
            
            # Verify insertion
            result = connection.execute(text("SELECT COUNT(*) FROM travel_policies"))
            count = result.fetchone()[0]
            
            print(f"✅ Successfully inserted {count} travel policies!")
            print("\n📋 Policy Categories Added:")
            print("• Flight booking and class rules")
            print("• Hotel accommodation standards")
            print("• Meal allowances and per diem")
            print("• International travel requirements")
            print("• Approval workflows and thresholds")
            print("• Ground transportation policies")
            print("• Technology and communication")
            print("• Emergency and special circumstances")
            print("• Expense reporting requirements")
            print("• Environmental and accessibility policies")
            
            return True
            
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def test_policy_queries():
    """Test some sample queries to verify database setup"""
    try:
        database_url = get_database_connection()
        engine = create_engine(database_url)
        
        print("\n🧪 Testing sample policy queries...")
        
        with engine.connect() as connection:
            # Test queries
            test_queries = [
                "SELECT rule_name FROM travel_policies WHERE description LIKE '%flight%' LIMIT 3",
                "SELECT rule_name FROM travel_policies WHERE description LIKE '%hotel%' LIMIT 3",
                "SELECT rule_name FROM travel_policies WHERE description LIKE '%meal%' LIMIT 3",
                "SELECT rule_name FROM travel_policies WHERE description LIKE '%approval%' LIMIT 3"
            ]
            
            for query in test_queries:
                result = connection.execute(text(query))
                policies = result.fetchall()
                category = query.split("'%")[1].split("%'")[0]
                print(f"  📌 {category.title()} policies: {len(policies)} found")
                for policy in policies:
                    print(f"    • {policy[0]}")
        
        print("\n✅ Database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up comprehensive travel policy database...")
    print("=" * 60)
    
    if setup_comprehensive_database():
        test_policy_queries()
        print("\n🎉 Database setup completed successfully!")
        print("Your AI assistant can now answer comprehensive travel policy questions!")
    else:
        print("\n❌ Database setup failed. Please check your database connection.")
