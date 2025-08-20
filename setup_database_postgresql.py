# setup_database_postgresql.py
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "travel_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0183813235")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    # Connect to PostgreSQL database
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    print(f"Connected to PostgreSQL database: {DB_NAME}")

    # --- Create tables ---

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
        applies_to_department VARCHAR(100), -- 'All' or specific department
        applies_to_job_level VARCHAR(50), -- 'All' or specific job level
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
        status VARCHAR(50), -- e.g., 'Pending', 'Approved', 'Rejected'
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
        status VARCHAR(50), -- 'Pending', 'Approved', 'Rejected'
        comments TEXT,
        approved_date TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (request_id) REFERENCES travel_requests(request_id),
        FOREIGN KEY (approver_id) REFERENCES employees(employee_id)
    );
    """)

    print("Tables created successfully!")

    # --- Insert sample data ---
    try:
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM employees")
        employee_count = cursor.fetchone()[0]
        
        if employee_count == 0:
            # Insert employee data
            cursor.execute("""
            INSERT INTO employees (email, first_name, last_name, department, job_level, manager_id, annual_travel_budget, remaining_budget) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, ('manager@example.com', 'Manager', 'One', 'Sales', 'Manager', None, 10000, 10000))
            
            cursor.execute("""
            INSERT INTO employees (email, first_name, last_name, department, job_level, manager_id, annual_travel_budget, remaining_budget) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, ('employee@example.com', 'Employee', 'One', 'Sales', 'Junior', 1, 5000, 5000))

            print("Sample employees inserted!")

        # Check if policies already exist
        cursor.execute("SELECT COUNT(*) FROM travel_policies")
        policy_count = cursor.fetchone()[0]
        
        if policy_count == 0:
            # Insert policy data
            cursor.execute("""
            INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) 
            VALUES (%s, %s, %s, %s, %s)
            """, ('Flight Cost Cap', 'Business class for flights over 6 hours.', 1500, 'All', 'All'))
            
            cursor.execute("""
            INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) 
            VALUES (%s, %s, %s, %s, %s)
            """, ('Hotel Per Night', 'Maximum hotel cost per night in major cities.', 300, 'All', 'All'))
            
            cursor.execute("""
            INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) 
            VALUES (%s, %s, %s, %s, %s)
            """, ('Meal Per Diem', 'Daily allowance for meals.', 75, 'All', 'All'))
            
            cursor.execute("""
            INSERT INTO travel_policies (rule_name, description, applies_to_department, applies_to_job_level) 
            VALUES (%s, %s, %s, %s)
            """, ('Approval Requirement', 'All international travel requires manager approval.', 'All', 'All'))

            print("Sample policies inserted!")
        else:
            print("Sample data already exists. Skipping insertion.")

    except psycopg2.IntegrityError as e:
        print(f"Sample data might already exist. Error: {e}")

    # Commit changes
    conn.commit()
    print(f"PostgreSQL database '{DB_NAME}' setup completed successfully!")

except psycopg2.OperationalError as e:
    print(f"Could not connect to PostgreSQL database: {e}")
    print("Please make sure:")
    print("1. PostgreSQL server is running")
    print(f"2. Database '{DB_NAME}' exists")
    print("3. User has correct permissions")
    print("4. Connection parameters in .env file are correct")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'conn' in locals():
        cursor.close()
        conn.close()
        print("Database connection closed.")
