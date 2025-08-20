#!/usr/bin/env python3
"""
=======================================================================
DATABASE_MANAGER.PY - 统一数据库管理模块
=======================================================================

本文件包含所有与数据库相关的功能，包括：
• 数据库连接和配置
• 政策数据管理 (CRUD操作)
• 员工数据管理
• 旅行请求管理
• 审批工作流管理
• 数据库初始化和设置
• 数据验证和测试功能

作者: AI Assistant
日期: 2025-08-21
版本: v1.0 - 企业旅行政策管理系统
=======================================================================
"""

import os
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 加载环境变量
load_dotenv()

# =======================================================================
# 📊 DATABASE CONFIGURATION - 数据库配置管理
# =======================================================================

class DatabaseConfig:
    """数据库配置管理类"""
    
    def __init__(self):
        self.USE_ONLINE_DATABASE = os.getenv("USE_ONLINE_DATABASE", "true").lower() == "true"
        self.DATABASE_ONLINE_URL = os.getenv("DATABASE_ONLINE")
        
        # 本地数据库配置
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = os.getenv("DB_PORT", "5432")
        self.DB_NAME = os.getenv("DB_NAME", "travel_db")
        self.DB_USER = os.getenv("DB_USER", "postgres")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "0183813235")
    
    def get_database_url(self):
        """获取数据库连接URL"""
        if self.USE_ONLINE_DATABASE and self.DATABASE_ONLINE_URL:
            try:
                # 测试在线数据库连接
                conn = psycopg2.connect(self.DATABASE_ONLINE_URL)
                conn.close()
                print("✅ Successfully connected to online Prisma database")
                return self.DATABASE_ONLINE_URL.replace("postgres://", "postgresql://", 1)
            except Exception as e:
                print(f"❌ Online connection failed: {e}")
                print("🔄 Falling back to local PostgreSQL...")
        
        # 使用本地数据库
        local_url = f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        print("📍 Using local PostgreSQL database...")
        return local_url
    
    def get_engine(self):
        """获取SQLAlchemy引擎"""
        return create_engine(self.get_database_url())

# 全局数据库配置实例
db_config = DatabaseConfig()

# =======================================================================
# 📋 TRAVEL POLICY MANAGEMENT - 旅行政策管理
# =======================================================================

class TravelPolicyManager:
    """旅行政策数据管理类"""
    
    @staticmethod
    def load_policies():
        """从数据库加载所有旅行政策"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                policies_result = connection.execute(text("SELECT rule_name, description FROM travel_policies"))
                policies = policies_result.fetchall()
            
            policy_list = []
            for name, desc in policies:
                policy_dict = {
                    'rule_name': name,
                    'description': desc,
                    'content': f"Policy Name: {name}\\nDetails: {desc}\\nTravel Policy: {name} - {desc}"
                }
                policy_list.append(policy_dict)
            
            print(f"✅ Loaded {len(policy_list)} travel policies from database")
            return policy_list
        
        except Exception as e:
            print(f"❌ Error loading policies: {e}")
            return TravelPolicyManager.get_default_policies()
    
    @staticmethod
    def get_default_policies():
        """获取默认旅行政策（数据库不可用时的备用方案）"""
        return [
            {
                'rule_name': 'Flight Booking Policy',
                'description': 'Economy class for flights under 6 hours. Business class allowed for flights over 6 hours or for Director level and above. Book at least 14 days in advance for best rates. Emergency travel exceptions require VP approval.',
                'content': 'Flight Booking Policy: Economy class for flights under 6 hours. Business class allowed for flights over 6 hours or for Director level and above. Book at least 14 days in advance for best rates. Emergency travel exceptions require VP approval.'
            },
            {
                'rule_name': 'Hotel Accommodation Standards',
                'description': 'Standard business hotels up to $300/night in major cities (NYC, London, Tokyo, Dubai). $200/night limit for other destinations. 4-star or equivalent quality required. Extended stay discounts should be negotiated for stays over 7 days.',
                'content': 'Hotel Accommodation Standards: Standard business hotels up to $300/night in major cities (NYC, London, Tokyo, Dubai). $200/night limit for other destinations. 4-star or equivalent quality required. Extended stay discounts should be negotiated for stays over 7 days.'
            },
            {
                'rule_name': 'Daily Meal Allowance',
                'description': 'Daily meal allowance: $75 for domestic travel, $100 for international travel, $125 for high-cost cities (NYC, London, Tokyo, Zurich). Receipts required for all meals over $25. Business dining with clients covered separately with approval.',
                'content': 'Daily Meal Allowance: Daily meal allowance: $75 for domestic travel, $100 for international travel, $125 for high-cost cities (NYC, London, Tokyo, Zurich). Receipts required for all meals over $25. Business dining with clients covered separately with approval.'
            },
            {
                'rule_name': 'International Travel Requirements',
                'description': 'International travel requires VP approval for costs over $3000. Passport must be valid for 6+ months. Company travel insurance mandatory. Check visa requirements 30 days before travel. Emergency contact information must be filed.',
                'content': 'International Travel Requirements: International travel requires VP approval for costs over $3000. Passport must be valid for 6+ months. Company travel insurance mandatory. Check visa requirements 30 days before travel. Emergency contact information must be filed.'
            },
            {
                'rule_name': 'Standard Approval Workflow',
                'description': 'Trips under $2000: Manager approval. $2000-$5000: Director approval. Over $5000: VP approval. Executive level (VP+) can self-approve up to $10000. Board approval required for international travel over $15000.',
                'content': 'Standard Approval Workflow: Trips under $2000: Manager approval. $2000-$5000: Director approval. Over $5000: VP approval. Executive level (VP+) can self-approve up to $10000. Board approval required for international travel over $15000.'
            }
        ]
    
    @staticmethod
    def setup_comprehensive_policies():
        """设置完整的旅行政策数据"""
        comprehensive_policies = [
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
        
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                # 清除现有政策
                print("🗑️ Clearing existing policies...")
                connection.execute(text("DELETE FROM travel_policies"))
                connection.commit()
                
                # 插入新政策
                print(f"📥 Inserting {len(comprehensive_policies)} comprehensive travel policies...")
                
                for policy in comprehensive_policies:
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
                
                # 验证插入
                result = connection.execute(text("SELECT COUNT(*) FROM travel_policies"))
                count = result.fetchone()[0]
                
                print(f"✅ Successfully inserted {count} travel policies!")
                return True
                
        except Exception as e:
            print(f"❌ Error setting up policies: {e}")
            return False

# =======================================================================
# 👥 EMPLOYEE MANAGEMENT - 员工数据管理
# =======================================================================

class EmployeeManager:
    """员工数据管理类"""
    
    @staticmethod
    def get_employee_data(employee_id):
        """获取员工信息"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                result = connection.execute(
                    text("""
                        SELECT employee_id, first_name, last_name, job_level, remaining_budget, department
                        FROM employees 
                        WHERE employee_id = :emp_id
                    """),
                    {'emp_id': employee_id}
                )
                
                employee = result.fetchone()
                if employee:
                    return employee
                else:
                    print(f"❌ Employee {employee_id} not found")
                    return None
                    
        except Exception as e:
            print(f"❌ Error fetching employee data: {e}")
            return None
    
    @staticmethod
    def update_employee_budget(employee_id, new_budget):
        """更新员工预算"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                connection.execute(
                    text("""
                        UPDATE employees 
                        SET remaining_budget = :budget
                        WHERE employee_id = :emp_id
                    """),
                    {'budget': new_budget, 'emp_id': employee_id}
                )
                connection.commit()
                print(f"✅ Updated budget for employee {employee_id}")
                return True
                
        except Exception as e:
            print(f"❌ Error updating employee budget: {e}")
            return False

# =======================================================================
# 🧾 TRAVEL REQUEST MANAGEMENT - 旅行请求管理
# =======================================================================

class TravelRequestManager:
    """旅行请求管理类"""
    
    @staticmethod
    def submit_travel_request(employee_id, destination, departure_date, return_date, 
                            estimated_cost, purpose, business_justification):
        """提交旅行请求"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                result = connection.execute(
                    text("""
                        INSERT INTO travel_requests 
                        (employee_id, destination, departure_date, return_date, 
                         estimated_cost, purpose, business_justification, status, created_at)
                        VALUES (:emp_id, :dest, :dep_date, :ret_date, :cost, :purpose, :justification, 'pending', NOW())
                        RETURNING request_id
                    """),
                    {
                        'emp_id': employee_id,
                        'dest': destination,
                        'dep_date': departure_date,
                        'ret_date': return_date,
                        'cost': estimated_cost,
                        'purpose': purpose,
                        'justification': business_justification
                    }
                )
                
                request_id = result.fetchone()[0]
                connection.commit()
                
                print(f"✅ Travel request {request_id} submitted successfully")
                return request_id
                
        except Exception as e:
            print(f"❌ Error submitting travel request: {e}")
            return None
    
    @staticmethod
    def get_travel_request(request_id):
        """获取旅行请求详情"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                result = connection.execute(
                    text("""
                        SELECT * FROM travel_requests 
                        WHERE request_id = :req_id
                    """),
                    {'req_id': request_id}
                )
                
                return result.fetchone()
                
        except Exception as e:
            print(f"❌ Error fetching travel request: {e}")
            return None

# =======================================================================
# ✅ APPROVAL WORKFLOW MANAGEMENT - 审批工作流管理
# =======================================================================

class ApprovalWorkflowManager:
    """审批工作流管理类"""
    
    @staticmethod
    def create_approval_workflow(request_id, approval_steps):
        """创建审批工作流"""
        try:
            engine = db_config.get_engine()
            
            with engine.connect() as connection:
                for step in approval_steps:
                    # 查找审批人
                    approver_result = connection.execute(
                        text("""
                            SELECT employee_id, first_name, last_name, email
                            FROM employees
                            WHERE job_level = :approver_type
                            AND department = :department
                            LIMIT 1
                        """),
                        {
                            'approver_type': step['approver_type'],
                            'department': step['department']
                        }
                    )
                    
                    approver = approver_result.fetchone()
                    if approver:
                        approver_id = approver[0]
                        
                        # 创建审批步骤
                        connection.execute(
                            text("""
                                INSERT INTO approval_workflows
                                (request_id, approval_level, approver_id, status, comments, created_at)
                                VALUES (:req_id, :level, :approver_id, :status, :comments, NOW())
                            """),
                            {
                                'req_id': request_id,
                                'level': step['level'],
                                'approver_id': approver_id,
                                'status': 'pending',
                                'comments': step.get('comments', 'Pending approval')
                            }
                        )
                
                connection.commit()
                print(f"✅ Approval workflow created for request {request_id}")
                return True
                
        except Exception as e:
            print(f"❌ Error creating approval workflow: {e}")
            return False

# =======================================================================
# 🧪 DATABASE TESTING AND VALIDATION - 数据库测试验证
# =======================================================================

class DatabaseValidator:
    """数据库验证和测试类"""
    
    @staticmethod
    def verify_database_setup():
        """验证数据库设置和数据完整性"""
        try:
            engine = db_config.get_engine()
            
            print("🔗 Connecting to database...")
            
            with engine.connect() as connection:
                # 检查政策总数
                result = connection.execute(text("SELECT COUNT(*) FROM travel_policies"))
                total_count = result.fetchone()[0]
                print(f"📊 Total policies in database: {total_count}")
                
                # 检查政策分类
                categories = [
                    ('Flight', ['flight', 'air', 'plane']),
                    ('Hotel', ['hotel', 'accommodation']),
                    ('Meal', ['meal', 'allowance', 'dining']),
                    ('International', ['international', 'visa', 'passport']),
                    ('Approval', ['approval', 'workflow']),
                    ('Transportation', ['transport', 'taxi', 'car']),
                    ('Expense', ['expense', 'receipt']),
                    ('Emergency', ['emergency', 'urgent'])
                ]
                
                print("\\n📋 Policy Categories:")
                for category, keywords in categories:
                    query = "SELECT COUNT(*) FROM travel_policies WHERE " + " OR ".join([
                        f"description LIKE '%{keyword}%'" for keyword in keywords
                    ])
                    result = connection.execute(text(query))
                    count = result.fetchone()[0]
                    print(f"  • {category}: {count} policies")
                
                return True
                
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            return False
    
    @staticmethod
    def test_policy_search():
        """测试政策搜索功能"""
        print("\\n🔍 Testing policy search functionality...")
        
        test_queries = [
            "flight class rules",
            "hotel cost limits",
            "meal allowance rates",
            "international travel approval",
            "emergency travel policy"
        ]
        
        policies = TravelPolicyManager.load_policies()
        
        for query in test_queries:
            relevant = [p for p in policies if any(
                keyword.lower() in p['description'].lower() 
                for keyword in query.split()
            )]
            print(f"  📌 '{query}': {len(relevant)} relevant policies found")
        
        return True

# =======================================================================
# 🚀 MAIN SETUP AND INITIALIZATION - 主要设置和初始化
# =======================================================================

def initialize_database():
    """初始化数据库（设置完整的政策数据）"""
    print("🚀 Initializing Travel Policy Database...")
    print("=" * 60)
    
    # 设置完整政策
    if TravelPolicyManager.setup_comprehensive_policies():
        print("\\n✅ Policy setup completed!")
        
        # 验证数据库
        if DatabaseValidator.verify_database_setup():
            print("\\n✅ Database verification passed!")
            
            # 测试搜索功能
            DatabaseValidator.test_policy_search()
            
            print("\\n🎉 Database initialization completed successfully!")
            return True
    
    print("\\n❌ Database initialization failed!")
    return False

# =======================================================================
# 📚 UTILITY FUNCTIONS - 实用工具函数
# =======================================================================

def get_database_stats():
    """获取数据库统计信息"""
    try:
        engine = db_config.get_engine()
        
        with engine.connect() as connection:
            stats = {}
            
            # 政策数量
            result = connection.execute(text("SELECT COUNT(*) FROM travel_policies"))
            stats['total_policies'] = result.fetchone()[0]
            
            # 员工数量
            try:
                result = connection.execute(text("SELECT COUNT(*) FROM employees"))
                stats['total_employees'] = result.fetchone()[0]
            except:
                stats['total_employees'] = 0
            
            # 旅行请求数量
            try:
                result = connection.execute(text("SELECT COUNT(*) FROM travel_requests"))
                stats['total_requests'] = result.fetchone()[0]
            except:
                stats['total_requests'] = 0
            
            return stats
            
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
        return {'total_policies': 0, 'total_employees': 0, 'total_requests': 0}

# =======================================================================
# 🎯 EXPORT FUNCTIONS - 导出函数 (供app.py使用)
# =======================================================================

# 数据库配置
def get_database_url():
    """获取数据库URL (供app.py使用)"""
    return db_config.get_database_url()

# 政策管理
def load_policies():
    """加载政策 (供app.py使用)"""
    return TravelPolicyManager.load_policies()

def get_default_policies():
    """获取默认政策 (供app.py使用)"""
    return TravelPolicyManager.get_default_policies()

# 员工管理
def get_employee_data(employee_id):
    """获取员工数据 (供app.py使用)"""
    return EmployeeManager.get_employee_data(employee_id)

# 旅行请求管理
def submit_travel_request(employee_id, destination, departure_date, return_date, 
                         estimated_cost, purpose, business_justification):
    """提交旅行请求 (供app.py使用)"""
    return TravelRequestManager.submit_travel_request(
        employee_id, destination, departure_date, return_date,
        estimated_cost, purpose, business_justification
    )

# =======================================================================
# 🏃‍♂️ MAIN EXECUTION - 主程序执行
# =======================================================================

if __name__ == "__main__":
    # 如果直接运行此文件，执行数据库初始化
    initialize_database()
