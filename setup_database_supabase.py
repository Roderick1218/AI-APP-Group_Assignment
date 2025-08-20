# setup_database_supabase.py
import os
import psycopg2
from dotenv import load_dotenv

from app import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE")

def test_connection():
    """Test connection to Supabase database"""
    try:
        print(f"Attempting to connect to Supabase database...")
        print(f"Host: {DB_HOST}")
        print(f"Database: {DB_NAME}")
        print(f"User: {DB_USER}")
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'  # Supabase requires SSL
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"✅ Successfully connected to Supabase!")
        print(f"Database version: {db_version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def setup_tables():
    """Create tables in Supabase database"""
    try:
        print("Connecting to Supabase database...")
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        
        cursor = conn.cursor()
        
        print("Creating tables...")
        
        # 1. Employees table (employees)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL UNIQUE,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            department VARCHAR(100),
            job_level VARCHAR(50),
            manager_id INTEGER,
            annual_travel_budget DECIMAL(10,2),
            remaining_budget DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # 2. Travel policies table (travel_policies)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS travel_policies (
            policy_id SERIAL PRIMARY KEY,
            rule_name VARCHAR(255) NOT NULL,
            description TEXT,
            limit_amount DECIMAL(10,2),
            applies_to_department VARCHAR(100),
            applies_to_job_level VARCHAR(50),
            effective_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # 3. Travel requests table (travel_requests)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS travel_requests (
            request_id SERIAL PRIMARY KEY,
            employee_id INTEGER,
            destination VARCHAR(255),
            departure_date DATE,
            return_date DATE,
            purpose TEXT,
            estimated_cost DECIMAL(10,2),
            actual_cost DECIMAL(10,2),
            status VARCHAR(50),
            approval_level_required INTEGER,
            approved_by INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        );
        """)
        
        # 4. Approval workflows table (approval_workflows)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS approval_workflows (
            workflow_id SERIAL PRIMARY KEY,
            request_id INTEGER,
            approver_id INTEGER,
            approval_level INTEGER,
            status VARCHAR(50),
            comments TEXT,
            approved_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (request_id) REFERENCES travel_requests(request_id),
            FOREIGN KEY (approver_id) REFERENCES employees(employee_id)
        );
        """)
        
        print("Tables created successfully.")
        
        # Insert sample data
        print("Inserting sample data...")
        
        # Check if employees already exist
        cursor.execute("SELECT COUNT(*) FROM employees")
        employee_count = cursor.fetchone()[0]
        
        if employee_count == 0:
            # Insert employees
            cursor.execute("""
            INSERT INTO employees (email, first_name, last_name, department, job_level, manager_id, annual_travel_budget, remaining_budget)
            VALUES 
                ('manager@example.com', 'Manager', 'One', 'Sales', 'Manager', NULL, 10000, 10000),
                ('employee@example.com', 'Employee', 'One', 'Sales', 'Junior', 1, 5000, 5000)
            """)
            
            # Insert policies
            cursor.execute("""
            INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level)
            VALUES 
                ('Flight Cost Cap', 'Business class for flights over 6 hours.', 1500, 'All', 'All'),
                ('Hotel Per Night', 'Maximum hotel cost per night in major cities.', 300, 'All', 'All'),
                ('Meal Per Diem', 'Daily allowance for meals.', 75, 'All', 'All'),
                ('Approval Requirement', 'All international travel requires manager approval.', NULL, 'All', 'All')
            """)
            
            print("Sample data inserted successfully.")
        else:
            print("Sample data already exists.")
        
        # Commit the transaction
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Supabase database setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up tables: {e}")
        return False
    
    return True

def create_rls_policies():
    """Create Row Level Security (RLS) policies for Supabase"""
    try:
        print("Setting up Row Level Security policies...")
        
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        
        cursor = conn.cursor()
        
        # Enable RLS on tables
        tables = ['employees', 'travel_policies', 'travel_requests', 'approval_workflows']
        
        for table in tables:
            cursor.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
            
            # Create policies (allowing all operations for now - you can customize later)
            cursor.execute(f"""
            CREATE POLICY "Allow all operations" ON {table}
            FOR ALL USING (true);
            """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("RLS policies created successfully.")
        return True
        
    except Exception as e:
        print(f"Note: RLS setup failed (this is normal if policies already exist): {e}")
        return True  # Return True as this is not critical

if __name__ == "__main__":
    print("Setting up Supabase database...")
    
    # Test connection first
    if test_connection():
        if setup_tables():
            create_rls_policies()
            print("🎉 Supabase database setup completed successfully!")
        else:
            print("❌ Failed to setup tables.")
    else:
        print("❌ Failed to connect to Supabase. Please check your configuration.")
        print("\nPlease ensure you have:")
        print("1. Created a Supabase project")
        print("2. Updated the .env file with your Supabase credentials")
        print("3. Your database is accessible")
