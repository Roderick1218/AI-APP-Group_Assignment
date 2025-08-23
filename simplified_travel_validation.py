#!/usr/bin/env python3 不用看
"""
Simplified Travel Request Validation System
Simplified Travel Request Validation System
"""

def simplified_travel_validation(destination, departure_date, return_date, estimated_cost, 
                                purpose, employee_email, travel_class="Economy", hotel_preference="Standard"):
    """
    Simplified Travel Request Validation System
    Submit travel request → AI validates against policies → Routes for approval → Database insertion
    """
    import psycopg2
    import os
    from datetime import datetime
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        # 1. Connect to database
        conn = psycopg2.connect(os.getenv('DATABASE_ONLINE'))
        cursor = conn.cursor()
        
        # 2. Get employee information and budget
        cursor.execute("""
            SELECT employee_id, first_name, last_name, department, job_level, 
                   annual_travel_budget, remaining_budget, manager_id
            FROM employees 
            WHERE email = %s
        """, (employee_email,))
        
        employee_data = cursor.fetchone()
        if not employee_data:
            return {
                'status': 'ERROR',
                'message': '❌ Employee not found in database',
                'employee_email': employee_email
            }
        
        employee_id, first_name, last_name, department, job_level, annual_budget, remaining_budget, manager_id = employee_data
        
        # Convert Decimal type to float to avoid type errors
        annual_budget = float(annual_budget) if annual_budget else 0.0
        remaining_budget = float(remaining_budget) if remaining_budget else 0.0
        
        # 3. Budget check
        budget_ok = remaining_budget >= estimated_cost
        budget_status = "✅ Within Budget" if budget_ok else "❌ Exceeds Budget"
        
        # 4. AI policy validation (simplified version)
        policy_violations = []
        warnings = []
        
        # Basic policy checks
        if estimated_cost > 15000:
            policy_violations.append("Cost exceeds RM15,000 limit")
        
        if not budget_ok:
            policy_violations.append(f"Insufficient budget: RM{remaining_budget:.2f} remaining")
        
        # International travel check
        domestic_cities = ['kuala lumpur', 'kl', 'johor bahru', 'penang', 'sabah', 'sarawak']
        is_international = destination.lower() not in domestic_cities and 'malaysia' not in destination.lower()
        
        if is_international and estimated_cost > 8000:
            warnings.append("International travel requires additional approvals")
        
        # 5. Determine approval level
        approval_level = determine_approval_level(estimated_cost, department, is_international)
        
        # 6. Calculate compliance score
        compliance_score = 100
        if policy_violations:
            compliance_score -= len(policy_violations) * 25
        if warnings:
            compliance_score -= len(warnings) * 10
        
        compliance_score = max(compliance_score, 0)
        
        # 7. Determine final status
        if policy_violations:
            final_status = "❌ REJECTED"
            status_code = "rejected"
        else:
            final_status = "✅ APPROVED" if compliance_score >= 80 else "⚠️ NEEDS REVIEW"
            status_code = "approved" if compliance_score >= 80 else "pending"
        
        # 8. Insert travel request into database
        request_id = None
        if status_code in ['approved', 'pending']:
            cursor.execute("""
                INSERT INTO travel_requests 
                (employee_id, destination, departure_date, return_date, purpose, 
                 estimated_cost, status, approval_level_required, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING request_id
            """, (employee_id, destination, departure_date, return_date, purpose, 
                  estimated_cost, status_code, approval_level))
            
            request_id = cursor.fetchone()[0]
            
            # 9. Create approval workflow
            if approval_level > 0:
                cursor.execute("""
                    INSERT INTO approval_workflows 
                    (request_id, approver_id, approval_level, status, created_at)
                    VALUES (%s, %s, %s, 'pending', NOW())
                """, (request_id, manager_id, approval_level))
            
            # 10. Update remaining budget (if auto-approved)
            if status_code == 'approved':
                cursor.execute("""
                    UPDATE employees 
                    SET remaining_budget = remaining_budget - %s 
                    WHERE employee_id = %s
                """, (estimated_cost, employee_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # 11. Return simplified result
        result = {
            'status': final_status,
            'request_id': request_id,
            'employee': f"{first_name} {last_name}",
            'department': department,
            'compliance_score': compliance_score,
            'budget_check': {
                'annual_budget': annual_budget,
                'remaining_budget': remaining_budget,
                'request_amount': estimated_cost,
                'status': budget_status,
                'utilization': f"{((annual_budget - remaining_budget + estimated_cost) / annual_budget * 100):.1f}%"
            },
            'policy_check': {
                'violations': policy_violations,
                'warnings': warnings,
                'approval_level': approval_level
            },
            'next_steps': get_next_steps(status_code, approval_level, is_international)
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'❌ System error: {str(e)}',
            'employee_email': employee_email
        }

def determine_approval_level(cost, department, is_international):
    """Determine required approval level"""
    if cost <= 3000:
        return 0  # Auto-approve
    elif cost <= 8000:
        return 1  # Direct manager approval required
    elif cost <= 15000:
        return 2  # Department director approval required
    else:
        return 3  # Senior management approval required

def get_next_steps(status, approval_level, is_international):
    """Get next steps"""
    if status == 'approved':
        steps = ["✅ Request approved - proceed with booking"]
    elif status == 'pending':
        if approval_level == 1:
            steps = ["⏳ Waiting for manager approval"]
        elif approval_level == 2:
            steps = ["⏳ Waiting for director approval"]
        else:
            steps = ["⏳ Waiting for senior management approval"]
    else:
        steps = ["❌ Request rejected - contact HR for assistance"]
    
    if is_international and status != 'rejected':
        steps.append("🌍 International travel - check passport validity")
        steps.append("🛡️ International travel insurance required")
    
    return steps

# Test function
if __name__ == "__main__":
    from datetime import date
    
    # Test case
    test_result = simplified_travel_validation(
        destination="Singapore",
        departure_date=date(2024, 12, 20),
        return_date=date(2024, 12, 25), 
        estimated_cost=5000.0,
        purpose="Business Meeting",
        employee_email="leeyongjun1218@e.newera.edu.my"
    )
    
    print("🧪 Test Result:")
    if isinstance(test_result, dict):
        print(f"Status: {test_result.get('status', 'Unknown')}")
        print(f"Request ID: {test_result.get('request_id', 'N/A')}")
        print(f"Compliance Score: {test_result.get('compliance_score', 'N/A')}")
    else:
        # If returned is a string report
        print("Returned report:")
        print(test_result[:500] + "..." if len(test_result) > 500 else test_result)
