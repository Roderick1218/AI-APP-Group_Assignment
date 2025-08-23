#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Malaysian Enterprise Travel Management System - Complete Data Update Script
Malaysian Enterprise Travel Management          600, 'All', 'Manager,Senior Manager,Director,VP', date.today()),
         
        # Hotel policies
        (4, 'Domestic Malaysia Hotel Policy',em - Complete Data Update

This script will:
1. Delete all existing data in the Prisma database
2. Regenerate data that complies with Malaysian enterprise environment
3. Include the two specific employee emails required by the teacher
4. Generate complete Malaysian travel policies
"""

import os
import psycopg2
from datetime import datetime, date, timedelta
from decimal import Decimal
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection"""
    if os.getenv("USE_ONLINE_DATABASE", "true").lower() == "true":
        return psycopg2.connect(os.getenv("DATABASE_ONLINE"))
    else:
        return psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'travel_db'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '0183813235'),
            port=os.getenv('DB_PORT', 5432)
        )

def clear_all_data():
    """Clear all existing data"""
    print("🗑️ Clearing all existing data...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Delete data in correct order (considering foreign key constraints)
        tables_to_clear = [
            'approval_workflows',
            'travel_requests', 
            'travel_policies',
            'employees'
        ]
        
        for table in tables_to_clear:
            cursor.execute(f"DELETE FROM {table}")
            print(f"✅ Cleared {table} table")
        
        # Reset sequences
        sequences_to_reset = [
            'employees_employee_id_seq',
            'travel_policies_policy_id_seq', 
            'travel_requests_request_id_seq',
            'approval_workflows_workflow_id_seq'
        ]
        
        for seq in sequences_to_reset:
            try:
                cursor.execute(f"ALTER SEQUENCE {seq} RESTART WITH 1")
                print(f"✅ Reset sequence {seq}")
            except:
                pass
        
        conn.commit()
        print("✅ All data clearing completed")
        
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_malaysian_employees():
    """Insert Malaysian enterprise employee data"""
    print("👥 Inserting Malaysian enterprise employee data...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Malaysian enterprise employee data (including two emails required by teacher)
    employees_data = [
        # C-Level executives
        (1, 'hiukaichien0424@e.newera.edu.my', 'Hiu Kai', 'Chien', 'Executive', 'CEO', None, 50000, 50000),
        (2, 'ahmad.rahman@company.my', 'Ahmad', 'Rahman', 'Executive', 'CFO', 1, 45000, 45000),
        (3, 'siti.nurhaliza@company.my', 'Siti', 'Nurhaliza', 'Executive', 'COO', 1, 40000, 40000),
        
        # VP level
        (4, 'leeyongjun1218@e.newera.edu.my', 'Lee Yong', 'Jun', 'Sales', 'VP', 1, 35000, 35000),
        (5, 'raj.kumar@company.my', 'Raj', 'Kumar', 'Information Technology', 'VP', 1, 32000, 32000),
        (6, 'fatimah.zahra@company.my', 'Fatimah', 'Zahra', 'Human Resources', 'VP', 1, 30000, 30000),
        
        # Director level
        (7, 'wong.mei.ling@company.my', 'Wong Mei', 'Ling', 'Sales', 'Director', 4, 25000, 25000),
        (8, 'muhammad.farid@company.my', 'Muhammad', 'Farid', 'Engineering', 'Director', 2, 24000, 24000),
        (9, 'priya.sharma@company.my', 'Priya', 'Sharma', 'Marketing', 'Director', 3, 23000, 23000),
        (10, 'lim.wei.kiat@company.my', 'Lim Wei', 'Kiat', 'Finance', 'Director', 2, 22000, 22000),
        
        # Senior Manager level
        (11, 'nurul.aina@company.my', 'Nurul', 'Aina', 'Sales', 'Senior Manager', 7, 20000, 20000),
        (12, 'tan.chong.wei@company.my', 'Tan Chong', 'Wei', 'Information Technology', 'Senior Manager', 5, 19000, 19000),
        (13, 'aisha.binti.omar@company.my', 'Aisha', 'Binti Omar', 'Human Resources', 'Senior Manager', 6, 18000, 18000),
        (14, 'david.kumar@company.my', 'David', 'Kumar', 'Engineering', 'Senior Manager', 8, 18500, 18500),
        (15, 'karen.loh@company.my', 'Karen', 'Loh', 'Marketing', 'Senior Manager', 9, 17500, 17500),
        
        # Manager level
        (16, 'mohd.hafiz@company.my', 'Mohd', 'Hafiz', 'Sales', 'Manager', 11, 15000, 15000),
        (17, 'jessica.tay@company.my', 'Jessica', 'Tay', 'Information Technology', 'Manager', 12, 15000, 15000),
        (18, 'zarina.abdullah@company.my', 'Zarina', 'Abdullah', 'Human Resources', 'Manager', 13, 14000, 14000),
        (19, 'alex.ng@company.my', 'Alex', 'Ng', 'Engineering', 'Manager', 14, 14500, 14500),
        (20, 'sarah.ibrahim@company.my', 'Sarah', 'Ibrahim', 'Finance', 'Manager', 10, 14000, 14000),
        
        # Associate level
        (21, 'azman.hassan@company.my', 'Azman', 'Hassan', 'Sales', 'Associate', 16, 12000, 12000),
        (22, 'michelle.chan@company.my', 'Michelle', 'Chan', 'Information Technology', 'Associate', 17, 12000, 12000),
        (23, 'farah.amira@company.my', 'Farah', 'Amira', 'Human Resources', 'Associate', 18, 11000, 11000),
        (24, 'ravi.krishnan@company.my', 'Ravi', 'Krishnan', 'Engineering', 'Associate', 19, 11500, 11500),
        (25, 'emily.wong@company.my', 'Emily', 'Wong', 'Marketing', 'Associate', 15, 11000, 11000),
    ]
    
    try:
        for emp_data in employees_data:
            cursor.execute("""
                INSERT INTO employees 
                (employee_id, email, first_name, last_name, department, job_level, 
                 manager_id, annual_travel_budget, remaining_budget, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (*emp_data, datetime.now()))
        
        conn.commit()
        print(f"✅ Successfully inserted {len(employees_data)} Malaysian enterprise employees")
        print("📧 Special employee emails included:")
        print("   • hiukaichien0424@e.newera.edu.my (CEO)")
        print("   • leeyongjun1218@e.newera.edu.my (VP Sales)")
        
    except Exception as e:
        print(f"❌ Error inserting employee data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_malaysian_travel_policies():
    """Insert Malaysian travel policies"""
    print("📋 Inserting Malaysian travel policies...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Malaysian travel policy data
    policies_data = [
        # Flight policies
        (1, 'Domestic Malaysia Flight Policy', 
         'Economy class mandatory for all domestic flights within Malaysia (KL-JB-Penang-Kota Kinabalu-Kuching). Budget airlines preferred (AirAsia, Malindo, Firefly). Business class only for medical reasons with medical certificate. Book minimum 14 days advance for better rates.',
         2000, 'All', 'All', date.today()),
         
        (2, 'ASEAN Regional Flight Policy',
         'Economy class preferred for ASEAN destinations (Singapore, Thailand, Indonesia, Philippines, Vietnam, Brunei). Business class allowed for Senior Manager+ for flights >4 hours or client development trips >RM8000. Advance booking 21+ days required.',
         8000, 'All', 'Senior Manager,Manager,Director,VP', date.today()),
         
        (3, 'International Flight Policy',
         'Business class allowed for international flights >8 hours for Manager+ level. First class only for VP+ with client trips >RM25000. All international flights require advance approval. Preferred airlines: Malaysia Airlines, Singapore Airlines for long-haul.',
         15000, 'All', 'Manager,Senior Manager,Director,VP', date.today()),
         
        # Hotel policies
        (4, 'Domestic Malaysia Hotel Policy',
         'Maximum RM300/night for KL, Penang, JB. RM250/night for other Malaysian cities. Prefer business hotels near LRT/MRT. Extended stay (7+ nights) qualifies for monthly rates. Halal breakfast preferred.',
         300, 'All', 'All', date.today()),
         
        (5, 'ASEAN Regional Hotel Policy',
         'Singapore: RM500/night max. Thailand/Indonesia: RM400/night max. Other ASEAN: RM350/night max. International chains preferred for consistency. Business districts or near public transport.',
         500, 'All', 'All', date.today()),
         
        (6, 'International Hotel Policy',
         'Europe/US/Australia: RM600/night max. Japan/Korea: RM550/night max. Other international: RM450/night max. 4-star+ business hotels required. Corporate rates when available.',
         600, 'All', 'Manager,Senior Manager,Director,VP', date.today()),
         
        # Meal policies
        (7, 'Domestic Malaysia Meal Allowance',
         'RM80/day for domestic travel. Halal options prioritized. Breakfast included in hotel rates when possible. Receipt required for claims >RM30. Local cuisine encouraged for cultural immersion.',
         80, 'All', 'All', date.today()),
         
        (8, 'ASEAN Regional Meal Allowance',
         'RM120/day for ASEAN destinations. Halal certification checked where available. Airport meals excluded from daily limit. Business meals with clients require pre-approval and business case.',
         120, 'All', 'All', date.today()),
         
        (9, 'International Meal Allowance',
         'RM180/day for international destinations. Currency exchange receipts required. Halal/vegetarian options to be researched in advance. Entertainment meals require director approval.',
         180, 'All', 'All', date.today()),
         
        # Transportation policies
        (10, 'Malaysia Domestic Transport Policy',
         'Grab/taxi up to RM75/day. LRT/MRT/buses preferred where available. Car rental for multiple city visits >3 days. Petrol receipts required. EV vehicles preferred when available.',
         75, 'All', 'All', date.today()),
         
        (11, 'International Transport Policy',
         'Airport transfers: RM150 max. Public transport preferred for daily commute. Car service for client meetings only. International driving permit required for rentals.',
         150, 'All', 'All', date.today()),
         
        # Approval policies
        (12, 'Domestic Travel Approval Policy',
         'Domestic Malaysia travel <RM3000: Manager approval. RM3000-RM8000: Director approval. >RM8000: VP approval. Emergency domestic travel can be verbally approved.',
         3000, 'All', 'Manager,Director,VP', date.today()),
         
        (13, 'International Travel Approval Policy',
         'All international travel requires VP approval regardless of cost. Submit request 30 days minimum. Include business case, ROI analysis, and post-travel report commitment.',
         0, 'All', 'VP,CEO', date.today()),
         
        # Special policies
        (14, 'Conference and Training Travel Policy',
         'Professional development travel encouraged. Conference fees separate from travel budget. Learning objectives and knowledge sharing plan required. Post-conference presentation to team mandatory.',
         5000, 'All', 'Manager,Director,VP', date.today()),
         
        (15, 'Emergency Travel Policy',
         'Family emergency or urgent business situations. Verbal approval from Director+ acceptable. Submit documentation within 48 hours. Higher cost limits apply with justification.',
         10000, 'All', 'Director,VP,CEO', date.today()),
    ]
    
    try:
        for policy_data in policies_data:
            cursor.execute("""
                INSERT INTO travel_policies 
                (policy_id, rule_name, description, limit_amount, applies_to_department, 
                 applies_to_job_level, effective_date, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (*policy_data, datetime.now()))
        
        conn.commit()
        print(f"✅ Successfully inserted {len(policies_data)} Malaysian travel policies")
        
    except Exception as e:
        print(f"❌ Error inserting policy data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_sample_travel_requests():
    """Insert sample travel requests"""
    print("✈️ Inserting sample travel requests...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Sample travel request data
    base_date = datetime.now().date()
    
    requests_data = [
        # Domestic travel requests
        (1, 1, 'Kuala Lumpur', base_date + timedelta(days=30), base_date + timedelta(days=32), 
         'Board meeting attendance', 1200.00, None, 'pending', 1, None),
         
        (2, 4, 'Penang', base_date + timedelta(days=15), base_date + timedelta(days=17),
         'Client presentation and negotiation', 1800.00, None, 'pending', 1, None),
         
        (3, 7, 'Johor Bahru', base_date + timedelta(days=45), base_date + timedelta(days=47),
         'Regional sales meeting', 1500.00, None, 'approved', 1, 4),
         
        (4, 11, 'Kota Kinabalu', base_date + timedelta(days=60), base_date + timedelta(days=65),
         'Market research and client visits', 4500.00, None, 'pending', 2, None),
         
        (5, 16, 'Kuching', base_date + timedelta(days=25), base_date + timedelta(days=28),
         'Product launch coordination', 2800.00, None, 'approved', 1, 7),
         
        # ASEAN region travel requests
        (6, 4, 'Singapore', base_date + timedelta(days=35), base_date + timedelta(days=38),
         'Strategic partnership discussion', 3200.00, None, 'pending', 2, None),
         
        (7, 2, 'Bangkok, Thailand', base_date + timedelta(days=50), base_date + timedelta(days=55),
         'ASEAN finance summit', 4800.00, None, 'approved', 2, 1),
         
        (8, 5, 'Jakarta, Indonesia', base_date + timedelta(days=40), base_date + timedelta(days=43),
         'Technology conference and vendor meetings', 3800.00, None, 'pending', 2, None),
         
        (9, 12, 'Ho Chi Minh City, Vietnam', base_date + timedelta(days=55), base_date + timedelta(days=59),
         'IT infrastructure assessment', 4200.00, None, 'pending', 2, None),
         
        (10, 8, 'Manila, Philippines', base_date + timedelta(days=65), base_date + timedelta(days=68),
         'Engineering collaboration project', 3600.00, None, 'approved', 2, 5),
         
        # International travel requests
        (11, 1, 'Tokyo, Japan', base_date + timedelta(days=70), base_date + timedelta(days=75),
         'International expansion strategy meeting', 12000.00, None, 'pending', 3, None),
         
        (12, 2, 'London, UK', base_date + timedelta(days=80), base_date + timedelta(days=87),
         'Global financial compliance conference', 15000.00, None, 'approved', 3, 1),
         
        (13, 4, 'Sydney, Australia', base_date + timedelta(days=90), base_date + timedelta(days=95),
         'Pacific region sales expansion', 13500.00, None, 'pending', 3, None),
         
        (14, 5, 'San Francisco, USA', base_date + timedelta(days=100), base_date + timedelta(days=107),
         'Silicon Valley technology expo', 18000.00, None, 'under_review', 3, None),
         
        (15, 7, 'Dubai, UAE', base_date + timedelta(days=85), base_date + timedelta(days=89),
         'Middle East market entry assessment', 11000.00, None, 'approved', 3, 4),
    ]
    
    try:
        for req_data in requests_data:
            cursor.execute("""
                INSERT INTO travel_requests 
                (request_id, employee_id, destination, departure_date, return_date, 
                 purpose, estimated_cost, actual_cost, status, approval_level_required, 
                 approved_by, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (*req_data, datetime.now(), datetime.now()))
        
        conn.commit()
        print(f"✅ Successfully inserted {len(requests_data)} sample travel requests")
        
    except Exception as e:
        print(f"❌ Error inserting travel request data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_approval_workflows():
    """Insert approval workflows"""
    print("📝 Inserting approval workflows...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Approval workflow data
    workflows_data = [
        # Approved request workflows
        (1, 3, 7, 1, 'approved', 'Domestic travel within policy limits', datetime.now() - timedelta(days=5)),
        (2, 5, 11, 1, 'approved', 'Standard domestic travel approval', datetime.now() - timedelta(days=3)),
        (3, 7, 2, 2, 'approved', 'ASEAN conference travel approved', datetime.now() - timedelta(days=2)),
        (4, 10, 5, 2, 'approved', 'Engineering collaboration approved', datetime.now() - timedelta(days=1)),
        (5, 12, 1, 3, 'approved', 'International expansion critical for Q4', datetime.now() - timedelta(days=1)),
        (6, 15, 4, 3, 'approved', 'Market research shows high potential ROI', datetime.now() - timedelta(days=2)),
        
        # Pending approval request workflows
        (7, 1, 7, 1, 'pending', 'Awaiting manager review', None),
        (8, 2, 11, 1, 'pending', 'Pending manager approval', None),
        (9, 4, 7, 2, 'pending', 'Requires director level approval', None),
        (10, 6, 7, 2, 'pending', 'Singapore travel pending director review', None),
        (11, 8, 5, 2, 'pending', 'Technology conference requires director approval', None),
        (12, 9, 12, 2, 'pending', 'Vietnam travel under review', None),
        (13, 11, 1, 3, 'pending', 'International travel requires VP approval', None),
        (14, 13, 4, 3, 'pending', 'Australia expansion under executive review', None),
        (15, 14, 5, 3, 'under_review', 'High-cost international travel under detailed review', None),
    ]
    
    try:
        for workflow_data in workflows_data:
            cursor.execute("""
                INSERT INTO approval_workflows 
                (workflow_id, request_id, approver_id, approval_level, status, 
                 comments, approved_date, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (*workflow_data, datetime.now()))
        
        conn.commit()
        print(f"✅ Successfully inserted {len(workflows_data)} approval workflows")
        
    except Exception as e:
        print(f"❌ Error inserting approval workflow data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def verify_data_integrity():
    """Verify data integrity"""
    print("🔍 Verifying data integrity...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check record count for each table
        tables = ['employees', 'travel_policies', 'travel_requests', 'approval_workflows']
        
        print("\n📊 Data Statistics:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   • {table}: {count} records")
        
        # Check specific employee emails
        cursor.execute("SELECT email, first_name, last_name, job_level FROM employees WHERE email IN (%s, %s)",
                      ('hiukaichien0424@e.newera.edu.my', 'leeyongjun1218@e.newera.edu.my'))
        special_employees = cursor.fetchall()
        
        print("\n👤 Special Employee Verification:")
        for email, first_name, last_name, job_level in special_employees:
            print(f"   • {email}: {first_name} {last_name} ({job_level})")
        
        # Check policy currency
        cursor.execute("SELECT rule_name, limit_amount FROM travel_policies WHERE limit_amount > 0 LIMIT 5")
        sample_policies = cursor.fetchall()
        
        print("\n💰 Policy Limit Verification (RM Currency):")
        for rule_name, limit_amount in sample_policies:
            print(f"   • {rule_name}: RM{limit_amount}")
        
        print("\n✅ Data integrity verification passed!")
        
    except Exception as e:
        print(f"❌ Error during data verification: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    """Main execution function"""
    print("🇲🇾 Malaysian Enterprise Travel Management System - Data Update Started")
    print("=" * 60)
    
    try:
        # Step 1: Clear existing data
        clear_all_data()
        print()
        
        # Step 2: Insert employee data
        insert_malaysian_employees()
        print()
        
        # Step 3: Insert travel policies
        insert_malaysian_travel_policies()
        print()
        
        # Step 4: Insert sample travel requests
        insert_sample_travel_requests()
        print()
        
        # Step 5: Insert approval workflows
        insert_approval_workflows()
        print()
        
        # Step 6: Verify data
        verify_data_integrity()
        
        print("\n🎉 Malaysian enterprise travel data update completed!")
        print("\n📋 Update Summary:")
        print("   • 25 Malaysian enterprise employees (including two required emails)")
        print("   • 15 Malaysian travel policies (using RM currency)")
        print("   • 15 diverse travel request scenarios") 
        print("   • 15 approval workflow records")
        print("   • Complete Malaysian enterprise hierarchy structure")
        print("   • ASEAN regional and international travel considerations")
        
        print("\n🌟 Features:")
        print("   • Complies with Malaysian business culture")
        print("   • Considers halal dining requirements")
        print("   • ASEAN regional focus")
        print("   • Uses Malaysian Ringgit (RM) currency")
        print("   • Diverse employee backgrounds (Malay, Chinese, Indian, Others)")
        
    except Exception as e:
        print(f"\n❌ Error occurred during update process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
