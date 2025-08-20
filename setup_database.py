# setup_database.py
import sqlite3
import os

DB_FILE = os.path.join("data", "travel.db")

# 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

# Connect to SQLite database (if file doesn't exist, it will be created)
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# --- Create tables ---

# 1. Employees table (employees)
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    first_name TEXT,
    last_name TEXT,
    department TEXT,
    job_level TEXT,
    manager_id INTEGER,
    annual_travel_budget REAL,
    remaining_budget REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# 2. Travel policies table (travel_policies)
cursor.execute("""
CREATE TABLE IF NOT EXISTS travel_policies (
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    description TEXT,
    limit_amount REAL,
    applies_to_department TEXT, -- 'All' or specific department
    applies_to_job_level TEXT, -- 'All' or specific job level
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# 3. Travel requests table (travel_requests)
cursor.execute("""
CREATE TABLE IF NOT EXISTS travel_requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    destination TEXT,
    departure_date DATE,
    return_date DATE,
    purpose TEXT,
    estimated_cost REAL,
    actual_cost REAL,
    status TEXT, -- e.g., 'Pending', 'Approved', 'Rejected'
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
    workflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER,
    approver_id INTEGER,
    approval_level INTEGER,
    status TEXT, -- 'Pending', 'Approved', 'Rejected'
    comments TEXT,
    approved_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES travel_requests(request_id),
    FOREIGN KEY (approver_id) REFERENCES employees(employee_id)
);
""")

# --- Insert sample data ---

try:
    # Insert employee data
    cursor.execute("INSERT INTO employees (email, first_name, last_name, department, job_level, manager_id, annual_travel_budget, remaining_budget) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   ('manager@example.com', 'Manager', 'One', 'Sales', 'Manager', None, 10000, 10000))
    cursor.execute("INSERT INTO employees (email, first_name, last_name, department, job_level, manager_id, annual_travel_budget, remaining_budget) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   ('employee@example.com', 'Employee', 'One', 'Sales', 'Junior', 1, 5000, 5000))

    # Insert policy data
    cursor.execute("INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) VALUES (?, ?, ?, ?, ?)",
                   ('Flight Cost Cap', 'Business class for flights over 6 hours.', 1500, 'All', 'All'))
    cursor.execute("INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) VALUES (?, ?, ?, ?, ?)",
                   ('Hotel Per Night', 'Maximum hotel cost per night in major cities.', 300, 'All', 'All'))
    cursor.execute("INSERT INTO travel_policies (rule_name, description, limit_amount, applies_to_department, applies_to_job_level) VALUES (?, ?, ?, ?, ?)",
                   ('Meal Per Diem', 'Daily allowance for meals.', 75, 'All', 'All'))
    cursor.execute("INSERT INTO travel_policies (rule_name, description, applies_to_department, applies_to_job_level) VALUES (?, ?, ?, ?)",
                   ('Approval Requirement', 'All international travel requires manager approval.', 'All', 'All'))

except sqlite3.IntegrityError:
    print("Sample data might already exist. Skipping insertion.")


# 提交更改并关闭连接
conn.commit()
conn.close()

print(f"Database '{DB_FILE}' created and populated successfully.")