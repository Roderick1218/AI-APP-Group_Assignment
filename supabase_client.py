# supabase_client.py
import os
from typing import Optional, List, Dict, Any
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Supabase database client for the travel policy app"""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'postgres')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD')
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    def get_connection(self):
        """Get database connection with SSL support for Supabase"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                sslmode='require'
            )
            return conn
        except Exception as e:
            print(f"Failed to connect to Supabase: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if connection to Supabase is working"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_employees(self) -> List[Dict[str, Any]]:
        """Get all employees from the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT employee_id, email, first_name, last_name, 
                           department, job_level, annual_travel_budget, remaining_budget
                    FROM employees
                    ORDER BY employee_id
                """)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error fetching employees: {e}")
            return []
    
    def get_travel_policies(self) -> List[Dict[str, Any]]:
        """Get all travel policies from the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT policy_id, rule_name, description, limit_amount,
                           applies_to_department, applies_to_job_level
                    FROM travel_policies
                    ORDER BY policy_id
                """)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error fetching travel policies: {e}")
            return []
    
    def create_travel_request(self, employee_id: int, destination: str, 
                            departure_date: str, return_date: str, 
                            purpose: str, estimated_cost: float) -> bool:
        """Create a new travel request"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO travel_requests 
                    (employee_id, destination, departure_date, return_date, purpose, estimated_cost, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'Pending')
                """, (employee_id, destination, departure_date, return_date, purpose, estimated_cost))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error creating travel request: {e}")
            return False
    
    def get_travel_requests(self, employee_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get travel requests, optionally filtered by employee"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if employee_id:
                    cursor.execute("""
                        SELECT tr.*, e.first_name, e.last_name, e.email
                        FROM travel_requests tr
                        JOIN employees e ON tr.employee_id = e.employee_id
                        WHERE tr.employee_id = %s
                        ORDER BY tr.created_at DESC
                    """, (employee_id,))
                else:
                    cursor.execute("""
                        SELECT tr.*, e.first_name, e.last_name, e.email
                        FROM travel_requests tr
                        JOIN employees e ON tr.employee_id = e.employee_id
                        ORDER BY tr.created_at DESC
                    """)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error fetching travel requests: {e}")
            return []
    
    def update_request_status(self, request_id: int, status: str, approver_id: Optional[int] = None) -> bool:
        """Update the status of a travel request"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE travel_requests 
                    SET status = %s, approved_by = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE request_id = %s
                """, (status, approver_id, request_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating request status: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                stats = {}
                
                tables = ['employees', 'travel_policies', 'travel_requests', 'approval_workflows']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            print(f"Error fetching database stats: {e}")
            return {}

# Global client instance
supabase_client = SupabaseClient()
