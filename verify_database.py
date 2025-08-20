#!/usr/bin/env python3
"""
Simple database verification script
"""

import os
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_database():
    """Verify the database has comprehensive policies"""
    try:
        # Database Configuration
        USE_ONLINE_DATABASE = os.getenv("USE_ONLINE_DATABASE", "true").lower() == "true"
        DATABASE_ONLINE_URL = os.getenv("DATABASE_ONLINE")
        
        if USE_ONLINE_DATABASE and DATABASE_ONLINE_URL:
            database_url = DATABASE_ONLINE_URL.replace("postgres://", "postgresql://", 1)
        else:
            DB_HOST = os.getenv("DB_HOST", "localhost")
            DB_PORT = os.getenv("DB_PORT", "5432")
            DB_NAME = os.getenv("DB_NAME", "travel_db")
            DB_USER = os.getenv("DB_USER", "postgres")
            DB_PASSWORD = os.getenv("DB_PASSWORD", "0183813235")
            database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        engine = create_engine(database_url)
        
        print("🔗 Connecting to database...")
        
        with engine.connect() as connection:
            # Check total policies
            result = connection.execute(text("SELECT COUNT(*) FROM travel_policies"))
            total_count = result.fetchone()[0]
            print(f"📊 Total policies in database: {total_count}")
            
            # Check policy categories
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
            
            print("\n📋 Policy Categories:")
            for category, keywords in categories:
                query = "SELECT COUNT(*) FROM travel_policies WHERE " + " OR ".join([
                    f"description LIKE '%{keyword}%'" for keyword in keywords
                ])
                result = connection.execute(text(query))
                count = result.fetchone()[0]
                print(f"  • {category}: {count} policies")
            
            # Show sample policies
            print("\n📝 Sample Policies:")
            result = connection.execute(text("SELECT rule_name, description FROM travel_policies LIMIT 5"))
            policies = result.fetchall()
            for i, (name, desc) in enumerate(policies, 1):
                print(f"  {i}. {name}")
                print(f"     {desc[:100]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Verifying Comprehensive Travel Policy Database")
    print("=" * 50)
    
    if verify_database():
        print("\n✅ Database verification completed successfully!")
        print("Your AI assistant has access to comprehensive travel policies.")
    else:
        print("\n❌ Database verification failed.")
