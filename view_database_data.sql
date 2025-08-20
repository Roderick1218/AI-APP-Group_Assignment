-- PostgreSQL Database Query Script
-- For viewing all data in the travel_db database

-- 1. View all tables
SELECT 'Available Tables:' as info;
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;

-- 2. View employee information
SELECT 'EMPLOYEES DATA:' as info;
SELECT 
    employee_id,
    email,
    first_name,
    last_name,
    department,
    job_level,
    manager_id,
    annual_travel_budget,
    remaining_budget,
    created_at
FROM employees
ORDER BY employee_id;

-- 3. View travel policies
SELECT 'TRAVEL POLICIES DATA:' as info;
SELECT 
    policy_id,
    rule_name,
    description,
    limit_amount,
    applies_to_department,
    applies_to_job_level,
    effective_date,
    created_at
FROM travel_policies
ORDER BY policy_id;

-- 4. View travel requests
SELECT 'TRAVEL REQUESTS DATA:' as info;
SELECT 
    request_id,
    employee_id,
    destination,
    departure_date,
    return_date,
    purpose,
    estimated_cost,
    actual_cost,
    status,
    approval_level_required,
    approved_by,
    created_at
FROM travel_requests
ORDER BY request_id;

-- 5. View approval workflows
SELECT 'APPROVAL WORKFLOWS DATA:' as info;
SELECT 
    workflow_id,
    request_id,
    approver_id,
    approval_level,
    status,
    comments,
    approved_date,
    created_at
FROM approval_workflows
ORDER BY workflow_id;

-- 6. Database statistics
SELECT 'DATABASE STATISTICS:' as info;
SELECT 
    'employees' as table_name,
    COUNT(*) as record_count
FROM employees
UNION ALL
SELECT 
    'travel_policies' as table_name,
    COUNT(*) as record_count
FROM travel_policies
UNION ALL
SELECT 
    'travel_requests' as table_name,
    COUNT(*) as record_count
FROM travel_requests
UNION ALL
SELECT 
    'approval_workflows' as table_name,
    COUNT(*) as record_count
FROM approval_workflows;
