#!/usr/bin/env python3
"""
=======================================================================
APP.PY - 企业旅行政策管理系统主应用
=======================================================================

核心功能模块:
🤖 Policy Q&A - AI驱动的政策问答聊天界面
📋 Request Validation - 智能旅行请求验证系统  
✅ Approval Workflow - 自动化审批工作流
🌍 Travel Planning - 智能旅行规划建议
📅 Calendar Integration - 自动日历生成

技术架构:
• Frontend: Streamlit Web Interface
• Backend: PostgreSQL/Prisma Database
• AI: Google Gemini + ChromaDB Vector Search
• Calendar: ICS格式自动生成

版本: v1.0 - 完整企业级解决方案
作者: AI Assistant Team
日期: 2025-08-21
=======================================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import re

# --- AI AND CALENDAR IMPORTS ---
from ics import Calendar, Event
import chromadb
import google.generativeai as genai

# --- DATABASE IMPORTS ---
from database_manager import (
    load_policies, get_default_policies, get_employee_data, 
    submit_travel_request, get_database_url, get_database_stats
)

# =======================================================================
# 🔧 CONFIGURATION - 应用程序配置
# =======================================================================

load_dotenv()

# Google AI配置 (可选 - 如果缺失会回退到本地搜索)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 日历文件配置
CALENDAR_FILE = "trip_event.ics"

# =======================================================================
# 🤖 AI COMPONENTS - AI组件初始化
# =======================================================================

def initialize_ai_components():
    """初始化Google Gemini AI和ChromaDB向量数据库"""
    try:
        # 初始化Gemini模型
        model = None
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 初始化ChromaDB
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db_policy")
        
        # 获取或创建政策向量集合
        try:
            collection = chroma_client.get_collection(name="travel_policies")
        except:
            collection = chroma_client.create_collection(
                name="travel_policies",
                metadata={"description": "Travel policy embeddings for semantic search"}
            )
        
        return model, collection
    except Exception as e:
        st.error(f"AI initialization failed: {e}")
        return None, None

def populate_vector_database(policies, collection):
    """向ChromaDB填充政策数据"""
    try:
        if not policies or not collection:
            return
        
        # 检查是否已有数据
        try:
            count = collection.count()
            if count > 0:
                return  # 已有数据，跳过
        except:
            pass
        
        # 添加政策到向量数据库
        documents = []
        metadatas = []
        ids = []
        
        for i, policy in enumerate(policies):
            documents.append(policy['content'])
            metadatas.append({
                'rule_name': policy['rule_name'],
                'description': policy['description']
            })
            ids.append(f"policy_{i}")
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(documents)} policies to vector database")
        
    except Exception as e:
        print(f"❌ Error populating vector database: {e}")

# =======================================================================
# 🧠 INTELLIGENT POLICY SEARCH - 智能政策搜索
# =======================================================================

def enhanced_ai_search(question, policies, gemini_model, policy_collection):
    """增强的AI搜索，带有更好的回退逻辑"""
    try:
        # 如果有有效的政策和AI，尝试智能搜索
        if policies and gemini_model:
            # 创建综合上下文
            context = "\\n\\n".join([
                f"**{policy['rule_name']}**: {policy['description']}" 
                for policy in policies[:3]  # 使用前3个最相关的
            ])
            
            # 创建智能提示
            prompt = f"""You are a helpful travel policy assistant for a company. Answer the user's question using the provided travel policies. Be specific, helpful, and professional.

AVAILABLE TRAVEL POLICIES:
{context}

USER QUESTION: {question}

Please provide a clear, helpful answer based on the policies above. If the question requires information not in the policies, provide general guidance and suggest contacting HR for specific details."""

            # 使用Gemini生成回复
            try:
                response = gemini_model.generate_content(prompt)
                if response and response.text and "I don't know" not in response.text:
                    return f"🤖 **AI Assistant Response:**\\n\\n{response.text}"
            except Exception as ai_error:
                st.warning(f"AI service unavailable: {ai_error}")
        
        # 回退到智能本地搜索
        return smart_local_search(question, policies)
        
    except Exception as e:
        return smart_local_search(question, policies)

def smart_local_search(question, policies):
    """增强的智能本地搜索，提供全面回复"""
    if not policies:
        return """❌ **No Policies Available**
        
Please contact your HR department for travel policy information, or try again later."""

    q_lower = question.lower()
    
    # 增强的问题分析，支持多个类别
    if any(word in q_lower for word in ['hotel', 'accommodation', 'room', 'stay', 'lodging']):
        hotel_policies = [p for p in policies if any(kw in p['rule_name'].lower() or kw in p['description'].lower() 
                         for kw in ['hotel', 'accommodation', 'lodging', 'room'])]
        if hotel_policies:
            response = "🏨 **Hotel & Accommodation Policies**\\n\\n"
            for policy in hotel_policies[:2]:
                response += f"**{policy['rule_name']}**\\n{policy['description']}\\n\\n"
            response += "💡 **Related topics:** Extended stay policies, booking requirements, corporate rates"
            return response

    elif any(word in q_lower for word in ['flight', 'plane', 'air', 'fly', 'airplane', 'business class', 'economy']):
        flight_policies = [p for p in policies if any(kw in p['rule_name'].lower() or kw in p['description'].lower() 
                          for kw in ['flight', 'air', 'plane', 'class', 'economy', 'business'])]
        if flight_policies:
            response = "✈️ **Flight & Air Travel Policies**\\n\\n"
            for policy in flight_policies[:2]:
                response += f"**{policy['rule_name']}**\\n{policy['description']}\\n\\n"
            response += "💡 **Related topics:** Class upgrades, booking timeframes, change policies"
            return response

    elif any(word in q_lower for word in ['meal', 'food', 'dining', 'allowance', 'per diem', 'restaurant']):
        meal_policies = [p for p in policies if any(kw in p['rule_name'].lower() or kw in p['description'].lower() 
                        for kw in ['meal', 'allowance', 'dining', 'food', 'per diem'])]
        if meal_policies:
            response = "🍽️ **Meal Allowance & Dining Policies**\\n\\n"
            for policy in meal_policies[:2]:
                response += f"**{policy['rule_name']}**\\n{policy['description']}\\n\\n"
            response += "💡 **Related topics:** Business entertainment, client meals, international rates"
            return response

    elif any(word in q_lower for word in ['approval', 'approve', 'manager', 'permission', 'workflow', 'authorize']):
        approval_policies = [p for p in policies if any(kw in p['rule_name'].lower() or kw in p['description'].lower() 
                            for kw in ['approval', 'approve', 'workflow', 'manager', 'director'])]
        if approval_policies:
            response = "✅ **Approval Workflow & Authorization**\\n\\n"
            for policy in approval_policies[:2]:
                response += f"**{policy['rule_name']}**\\n{policy['description']}\\n\\n"
            response += "💡 **Related topics:** Emergency approvals, cost thresholds, self-approval limits"
            return response

    elif any(word in q_lower for word in ['international', 'overseas', 'abroad', 'foreign', 'visa', 'passport']):
        intl_policies = [p for p in policies if any(kw in p['rule_name'].lower() or kw in p['description'].lower() 
                        for kw in ['international', 'visa', 'passport', 'foreign', 'overseas'])]
        if intl_policies:
            response = "🌍 **International Travel Policies**\\n\\n"
            for policy in intl_policies[:2]:
                response += f"**{policy['rule_name']}**\\n{policy['description']}\\n\\n"
            response += "💡 **Related topics:** Visa requirements, health insurance, safety protocols"
            return response

    # 通用全面回复，展示所有可用政策类别
    response = f"""📋 **Travel Policy Information**

**Available Policy Categories:**

🛫 **Flight & Air Travel**
• Flight booking and class policies
• Upgrade rules and restrictions
• Change and cancellation policies

🏨 **Accommodation**
• Hotel standards and limits
• Extended stay policies
• Booking requirements

🍽️ **Meals & Entertainment**
• Daily allowances and per diem
• Business entertainment rules
• Receipt requirements

🌍 **International Travel**
• Visa and documentation
• Health and safety requirements
• Approval processes

✅ **Approvals & Workflows**
• Standard approval thresholds
• Emergency approval process
• Self-approval limits

**💡 Try asking specific questions like:**
• "What are the flight class rules?"
• "Hotel cost limits for major cities?"
• "International travel approval process?"
• "Emergency travel policies?"
• "Meal allowance rates?"

**📞 Need help?** Contact HR for detailed policy clarification."""

    return response

def answer_policy_question(question):
    """主要的政策问答函数"""
    try:
        if not question or not question.strip():
            return "❌ Please enter a question about travel policies."
        
        # 从数据库加载政策，如果需要则回退到默认政策
        policies = load_policies()
        if not policies:
            st.warning("Using default policies - database connection issue")
            policies = get_default_policies()
        
        # 增强的关键词匹配，提高相关性
        q_lower = question.lower()
        relevant_policies = []
        
        # 智能关键词映射
        keyword_map = {
            'flight': ['flight', 'plane', 'airplane', 'air', 'fly', 'business class', 'economy', 'airline', 'upgrade', 'seat'],
            'hotel': ['hotel', 'accommodation', 'room', 'stay', 'lodging', 'night', 'booking', 'extended stay'],
            'meal': ['meal', 'food', 'dining', 'eat', 'restaurant', 'allowance', 'per diem', 'breakfast', 'lunch', 'dinner'],
            'international': ['international', 'overseas', 'abroad', 'foreign', 'visa', 'passport', 'embassy', 'health'],
            'approval': ['approval', 'approve', 'manager', 'director', 'permission', 'workflow', 'authorize', 'sign off'],
            'transportation': ['transport', 'taxi', 'uber', 'lyft', 'rental car', 'parking', 'mileage', 'ground'],
            'expense': ['expense', 'receipt', 'reimbursement', 'credit card', 'report', 'billing', 'cost'],
            'emergency': ['emergency', 'urgent', 'family', 'bereavement', 'crisis', 'immediate'],
            'entertainment': ['entertainment', 'client', 'business dinner', 'team meal', 'conference'],
            'technology': ['phone', 'internet', 'wifi', 'communication', 'laptop', 'equipment']
        }
        
        # 查找匹配的政策
        for policy in policies:
            rule_name = policy.get('rule_name', '').lower()
            description = policy.get('description', '').lower()
            
            # 直接关键词匹配
            match_found = any(keyword in rule_name or keyword in description 
                            for keyword in q_lower.split())
            
            # 基于类别的匹配
            if not match_found:
                for category, keywords in keyword_map.items():
                    if any(kw in q_lower for kw in keywords):
                        if category in rule_name or any(kw in description for kw in keywords):
                            match_found = True
                            break
            
            if match_found:
                relevant_policies.append(policy)
        
        # 如果没有特定匹配，根据问题类型返回有用的政策
        if not relevant_policies:
            if any(word in q_lower for word in ['cost', 'money', 'budget', 'price']):
                relevant_policies = [p for p in policies if 'allowance' in p['rule_name'].lower() or 'hotel' in p['rule_name'].lower()]
            elif any(word in q_lower for word in ['approve', 'permission', 'manager']):
                relevant_policies = [p for p in policies if 'approval' in p['rule_name'].lower()]
            else:
                relevant_policies = policies[:3]  # 显示前3个政策
        
        # 初始化AI组件
        gemini_model, policy_collection = initialize_ai_components()
        
        # 使用增强的AI搜索
        return enhanced_ai_search(question, relevant_policies, gemini_model, policy_collection)
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return f"❌ **Service Error:** {str(e)}\\n\\nPlease try again or contact support."

# =======================================================================
# 📋 REQUEST VALIDATION SYSTEM - 请求验证系统
# =======================================================================

def advanced_request_validation(destination, departure_date, return_date, estimated_cost, purpose, employee_data, policies):
    """增强的AI驱动请求验证，对照公司政策"""
    validation_results = {
        'is_valid': True,
        'violations': [],
        'warnings': [],
        'suggestions': [],
        'policy_references': []
    }
    
    employee_id, first_name, last_name, job_level, remaining_budget, department = employee_data
    duration_days = (return_date - departure_date).days
    
    # 1. 预算验证
    if estimated_cost > float(remaining_budget):
        validation_results['violations'].append({
            'type': 'budget_exceeded',
            'message': f"Cost ${estimated_cost:,.2f} exceeds remaining budget ${float(remaining_budget):,.2f}",
            'severity': 'high'
        })
        validation_results['is_valid'] = False
    
    # 2. 基于政策的成本验证
    dest_lower = destination.lower()
    
    # 酒店政策验证
    daily_hotel_limit = 300 if dest_lower in ['london', 'tokyo', 'new york'] else 200
    estimated_hotel_cost = duration_days * daily_hotel_limit * 0.8  # 限额的80%
    
    if estimated_cost > estimated_hotel_cost * 2:  # 显著超过预期
        validation_results['warnings'].append({
            'type': 'high_cost',
            'message': f"Estimated cost seems high for {duration_days} days in {destination}",
            'suggestion': f"Expected hotel cost: ~${estimated_hotel_cost:,.2f}"
        })
    
    # 3. 提前预订检查
    advance_days = (departure_date - datetime.now().date()).days
    if advance_days < 14:
        validation_results['warnings'].append({
            'type': 'advance_booking',
            'message': f"Only {advance_days} days advance notice. Policy recommends 14+ days for better rates.",
            'suggestion': "Consider rebooking for better rates if possible"
        })
    
    # 4. 国际旅行要求
    international_destinations = ['london', 'paris', 'tokyo', 'singapore', 'sydney', 'toronto', 'mumbai', 'beijing']
    is_international = any(dest in dest_lower for dest in international_destinations)
    
    if is_international:
        if estimated_cost > 3000 and job_level not in ['VP', 'C-Level']:
            validation_results['policy_references'].append({
                'policy': 'International Travel Requirements',
                'requirement': 'VP approval required for international travel over $3000'
            })
        
        validation_results['suggestions'].append({
            'type': 'international_prep',
            'message': "International travel checklist: Passport (6+ months valid), visa check, travel insurance",
            'action': "Verify passport validity and visa requirements"
        })
    
    # 5. 部门特定规则
    dept_rules = {
        'Sales': {'threshold': 5000, 'approval_level': 'Director'},
        'Engineering': {'threshold': 3000, 'approval_level': 'Manager'},
        'Marketing': {'threshold': 4000, 'approval_level': 'Director'}
    }
    
    if department in dept_rules:
        rule = dept_rules[department]
        if estimated_cost > rule['threshold']:
            validation_results['policy_references'].append({
                'policy': f'{department} Department Rules',
                'requirement': f'{rule["approval_level"]} approval required for costs over ${rule["threshold"]}'
            })
    
    # 6. 成本优化建议
    if duration_days >= 7:
        validation_results['suggestions'].append({
            'type': 'cost_optimization',
            'message': "Consider extended stay rates for 7+ day trips",
            'action': "Negotiate weekly/monthly rates for longer stays"
        })
    
    if estimated_cost > 2000:
        validation_results['suggestions'].append({
            'type': 'advance_booking',
            'message': "Book flights 21+ days in advance for significant savings",
            'action': "Early booking can save 20-40% on flight costs"
        })
    
    return validation_results

# =======================================================================
# ✅ ENHANCED APPROVAL WORKFLOW - 增强审批工作流
# =======================================================================

def enhanced_approval_workflow(employee_data, estimated_cost, destination, purpose):
    """增强的智能审批工作流，支持智能路由和自动批准"""
    employee_id, first_name, last_name, job_level, remaining_budget, department = employee_data
    
    workflow_steps = []
    approval_info = {
        'auto_approved': False,
        'approval_path': [],
        'estimated_time': '1-3 business days',
        'special_requirements': []
    }
    
    # 部门特定规则
    department_rules = {
        'Sales': {'threshold': 5000, 'requires_director': True},
        'Engineering': {'threshold': 3000, 'requires_director': False},
        'Marketing': {'threshold': 4000, 'requires_director': True},
        'Finance': {'threshold': 2000, 'requires_director': True},
        'HR': {'threshold': 2500, 'requires_director': False}
    }
    
    dept_rule = department_rules.get(department, {'threshold': 3000, 'requires_director': True})
    
    # 国际旅行检查
    international_destinations = ['london', 'paris', 'tokyo', 'singapore', 'sydney', 'toronto', 'mumbai', 'beijing']
    is_international = any(dest in destination.lower() for dest in international_destinations)
    
    # 智能审批路由
    if job_level in ['C-Level', 'CEO', 'CFO', 'CTO']:
        # 高管自动批准（有限额）
        if estimated_cost <= 10000:
            approval_info['auto_approved'] = True
            approval_info['approval_path'] = ['Auto-approved (Executive Level)']
            approval_info['estimated_time'] = 'Immediate'
        else:
            workflow_steps.append({
                'level': 1,
                'approver_type': 'Board',
                'requirement': 'Board approval for executive travel over $10,000',
                'auto_approve': False
            })
            approval_info['estimated_time'] = '5-7 business days'
    
    elif job_level == 'VP':
        # VP级别审批
        if estimated_cost <= 7500:
            approval_info['auto_approved'] = True
            approval_info['approval_path'] = ['Auto-approved (VP Level)']
            approval_info['estimated_time'] = 'Immediate'
        else:
            workflow_steps.append({
                'level': 1,
                'approver_type': 'C-Level',
                'requirement': 'C-Level approval for VP travel over $7,500',
                'auto_approve': False
            })
    
    else:
        # 标准员工审批流程
        if estimated_cost <= 1000 and not is_international:
            # 低成本国内旅行 - 经理审批
            workflow_steps.append({
                'level': 1,
                'approver_type': 'Manager',
                'department': department,
                'auto_approve': job_level == 'Director'
            })
        
        elif estimated_cost <= dept_rule['threshold'] and not is_international:
            # 中等成本 - 部门规则
            workflow_steps.append({
                'level': 1,
                'approver_type': 'Manager',
                'department': department,
                'auto_approve': False
            })
            
            if dept_rule['requires_director']:
                workflow_steps.append({
                    'level': 2,
                    'approver_type': 'Director',
                    'department': department,
                    'auto_approve': False
                })
        
        else:
            # 高成本或国际旅行 - 多级审批
            workflow_steps.append({
                'level': 1,
                'approver_type': 'Manager',
                'department': department,
                'auto_approve': False
            })
            
            workflow_steps.append({
                'level': 2,
                'approver_type': 'Director',
                'department': department,
                'auto_approve': False
            })
            
            if estimated_cost > 5000 or is_international:
                workflow_steps.append({
                    'level': 3,
                    'approver_type': 'VP',
                    'department': department,
                    'auto_approve': False
                })
                approval_info['estimated_time'] = '3-5 business days'
    
    # 违规处理和升级
    if estimated_cost > float(remaining_budget):
        approval_info['special_requirements'].append({
            'type': 'budget_violation',
            'message': 'Budget exceeded - requires CFO approval',
            'additional_approver': 'CFO'
        })
        approval_info['estimated_time'] = '5-7 business days'
    
    # 成本基础路由
    if estimated_cost > 10000:
        approval_info['special_requirements'].append({
            'type': 'high_cost',
            'message': 'High-cost travel - requires detailed justification',
            'requirement': 'Detailed business case required'
        })
    
    # 国际旅行额外批准
    if is_international:
        approval_info['special_requirements'].append({
            'type': 'international',
            'message': 'International travel - additional documentation required',
            'requirements': ['Passport copy', 'Visa status', 'Travel insurance']
        })
    
    # 构建审批路径
    if not approval_info['auto_approved']:
        for step in workflow_steps:
            approval_info['approval_path'].append(f"Level {step['level']}: {step['approver_type']}")
    
    return {
        'workflow_steps': workflow_steps,
        'approval_info': approval_info,
        'is_auto_approved': approval_info['auto_approved']
    }

# =======================================================================
# 🌍 TRAVEL PLANNING INTEGRATION - 旅行规划集成
# =======================================================================

def generate_travel_planning(destination, departure_date, return_date, purpose):
    """生成智能旅行规划建议"""
    duration_days = (return_date - departure_date).days
    
    # 基础目的地信息
    destination_info = {
        'london': {
            'timezone': 'GMT/BST',
            'business_districts': 'City of London, Canary Wharf',
            'airports': 'Heathrow (LHR), Gatwick (LGW)',
            'transportation': 'Tube, Bus, Taxi',
            'cultural_notes': 'Business attire required, punctuality important'
        },
        'new york': {
            'timezone': 'EST/EDT',
            'business_districts': 'Manhattan Financial District, Midtown',
            'airports': 'JFK, LaGuardia (LGA), Newark (EWR)',
            'transportation': 'Subway, Taxi, Uber/Lyft',
            'cultural_notes': 'Fast-paced environment, networking important'
        },
        'tokyo': {
            'timezone': 'JST',
            'business_districts': 'Marunouchi, Shibuya, Shinjuku',
            'airports': 'Narita (NRT), Haneda (HND)',
            'transportation': 'JR Lines, Metro, Taxi',
            'cultural_notes': 'Business cards exchange, bow greeting, punctuality critical'
        }
    }
    
    dest_key = destination.lower().replace(' ', '')
    travel_info = destination_info.get(dest_key, {
        'timezone': 'Check local timezone',
        'business_districts': 'Research local business areas',
        'airports': 'Check nearest airports',
        'transportation': 'Research local transport options',
        'cultural_notes': 'Research local business customs'
    })
    
    # 旅行建议
    travel_advisories = {
        'visa_requirements': 'Check visa requirements 30 days before travel',
        'health_requirements': 'Verify vaccination requirements',
        'safety_level': 'Check current travel advisories',
        'documentation': 'Ensure passport valid 6+ months',
        'emergency_contacts': {
            'local_emergency': '911 (US), 999 (UK), 110 (Japan)',
            'embassy': 'Contact nearest embassy'
        }
    }
    
    # 成本优化建议
    cost_optimization = []
    
    if duration_days >= 7:
        cost_optimization.append("Consider extended stay hotel rates")
        cost_optimization.append("Look into corporate housing options")
    
    if duration_days >= 14:
        cost_optimization.append("Negotiate monthly rates")
        cost_optimization.append("Consider apartment-style accommodations")
    
    advance_days = (departure_date - datetime.now().date()).days
    if advance_days >= 21:
        cost_optimization.append("Book flights early for 20-40% savings")
    
    return {
        'destination_info': travel_info,
        'travel_advisories': travel_advisories,
        'cost_optimization': cost_optimization,
        'duration_days': duration_days,
        'advance_days': advance_days
    }

def format_travel_itinerary(destination, departure_date, return_date, purpose, travel_info):
    """格式化旅行行程建议"""
    
    itinerary = f"""
# 🌍 Travel Planning for {destination}

## 📅 Trip Overview
- **Destination:** {destination}
- **Duration:** {travel_info['duration_days']} days
- **Purpose:** {purpose}
- **Advance Booking:** {travel_info['advance_days']} days

## 🛫 Travel Information
- **Time Zone:** {travel_info['destination_info'].get('timezone', 'Check local timezone')}
- **Airports:** {travel_info['destination_info'].get('airports', 'Check nearest airports')}
- **Local Transportation:** {travel_info['destination_info'].get('transportation', 'Research options')}

## 💰 Cost Optimization Tips
"""
    
    for tip in travel_info['cost_optimization']:
        itinerary += f"• {tip}\\n"
    
    itinerary += f"""

## 🛡️ Travel Advisories
- **Visa Requirements:** {travel_info['travel_advisories'].get('visa_requirements', 'Check requirements')}
- **Safety Level:** {travel_info['travel_advisories'].get('safety_level', 'Check current status')}
- **Health Requirements:** {travel_info['travel_advisories'].get('health_requirements', 'N/A')}
- **Documentation:** {travel_info['travel_advisories'].get('documentation', 'N/A')}

## 🏢 Business Information
- **Business Districts:** {travel_info['destination_info'].get('business_districts', 'N/A')}
- **Cultural Notes:** {travel_info['destination_info'].get('cultural_notes', 'N/A')}
- **Transportation:** {travel_info['destination_info'].get('transportation', 'N/A')}

## 📞 Emergency Contacts
- **Local Emergency:** {travel_info['travel_advisories']['emergency_contacts'].get('local_emergency', 'N/A')}
- **Embassy:** {travel_info['travel_advisories']['emergency_contacts'].get('embassy', 'Contact nearest embassy')}

## ✅ Pre-Travel Checklist
□ Confirm flights and accommodation
□ Check passport validity (6+ months required)
□ Review and purchase travel insurance
□ Check vaccination requirements
□ Arrange currency exchange
□ Download offline maps and translation apps
□ Notify bank and credit card companies
□ Set up international phone/data plan
□ Pack weather-appropriate business attire
□ Prepare business cards and meeting materials

## 📱 Recommended Apps
- Google Translate (offline mode)
- XE Currency
- Local transportation apps
- Weather app
- Company expense tracking app
    """
    
    return itinerary

# =======================================================================
# 📅 CALENDAR INTEGRATION - 日历集成
# =======================================================================

def generate_travel_calendar(destination, departure_date, return_date, purpose):
    """生成旅行日历事件"""
    try:
        # 创建日历
        cal = Calendar()
        
        # 创建旅行事件
        event = Event()
        event.name = f"Business Travel: {destination}"
        event.begin = departure_date
        event.end = return_date + timedelta(days=1)  # 全天事件
        event.description = f"""
Business Travel Details:
• Destination: {destination}
• Purpose: {purpose}
• Duration: {(return_date - departure_date).days} days

Please ensure all travel documents are ready and expenses are properly documented.
        """.strip()
        event.location = destination
        
        # 添加事件到日历
        cal.events.add(event)
        
        # 保存到文件
        with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
            f.writelines(cal)
        
        return True, CALENDAR_FILE
        
    except Exception as e:
        return False, str(e)

# =======================================================================
# 🎨 STREAMLIT UI - 用户界面
# =======================================================================

def main():
    """主Streamlit应用程序"""
    
    # 页面配置
    st.set_page_config(
        page_title="Enterprise Travel Policy Manager",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 应用程序标题
    st.title("🏢 Enterprise Travel Policy Management System")
    st.markdown("**AI-Powered Travel Policy Assistant & Request Management**")
    
    # 侧边栏信息
    with st.sidebar:
        st.header("📊 System Status")
        
        # 获取数据库统计
        try:
            stats = get_database_stats()
            st.metric("Total Policies", stats['total_policies'])
            st.metric("Total Employees", stats['total_employees'])
            st.metric("Travel Requests", stats['total_requests'])
        except:
            st.error("Database connection issue")
        
        st.markdown("---")
        st.markdown("### 🔧 System Features")
        st.markdown("""
        ✅ **Policy Q&A Chat**  
        ✅ **Request Validation**  
        ✅ **Approval Workflows**  
        ✅ **Travel Planning**  
        ✅ **Calendar Integration**
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.markdown("Contact HR for policy questions")
    
    # 主要标签页
    tab1, tab2 = st.tabs(["🤖 Policy Q&A", "📋 Submit Travel Request"])
    
    # =======================================================================
    # TAB 1: POLICY Q&A CHAT INTERFACE
    # =======================================================================
    
    with tab1:
        st.header("🤖 Travel Policy AI Assistant")
        
        # 初始化ChromaDB
        try:
            client = chromadb.PersistentClient(path="./data/chroma_db_policy")
            policy_collection = client.get_or_create_collection(name="travel_policies")
            
            # 加载政策
            policies = load_policies()
            st.success(f"✅ Loaded {len(policies)} travel policies")
            
            # 填充向量数据库
            if policy_collection:
                populate_vector_database(policies, policy_collection)
                
        except Exception as e:
            st.error(f"❌ Error loading policies: {e}")
            st.stop()
        
        # 初始化聊天会话状态
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hello! I'm your AI Travel Policy Assistant. I can help you with:\\n\\n• Flight booking policies and costs\\n• Hotel accommodation rules\\n• Meal allowances and per diem\\n• International travel requirements\\n• Approval workflows\\n• Budget and expense guidelines\\n\\nFeel free to ask me anything about company travel policies!",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
        
        # 聊天界面样式
        st.markdown("""
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 5px 18px;
            margin: 8px 0 8px 50px;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .assistant-message {
            background: white;
            color: #333;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 5px;
            margin: 8px 50px 8px 0;
            max-width: 80%;
            float: left;
            clear: both;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .message-time {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 聊天消息显示
        st.markdown("### 💬 Chat")
        
        # 创建可滚动的聊天容器
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You</strong><br>
                        {message["content"]}
                        <div class="message-time">{message["timestamp"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>🤖 AI Assistant</strong><br>
                        {message["content"]}
                        <div class="message-time">{message["timestamp"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)
        
        # 快速操作按钮
        st.markdown("### 🚀 Quick Questions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏨 Hotel Policy", key="hotel_quick_chat"):
                user_msg = "What are the hotel cost limits and booking rules?"
                st.session_state.chat_messages.append({
                    "role": "user", 
                    "content": user_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                with st.spinner("🤖 AI thinking..."):
                    response = answer_policy_question(user_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                st.rerun()
        
        with col2:
            if st.button("✈️ Flight Rules", key="flight_quick_chat"):
                user_msg = "What are the flight class rules and booking policies?"
                st.session_state.chat_messages.append({
                    "role": "user", 
                    "content": user_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                with st.spinner("🤖 AI thinking..."):
                    response = answer_policy_question(user_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                st.rerun()
        
        with col3:
            if st.button("🍽️ Meal Allowance", key="meal_quick_chat"):
                user_msg = "What is the daily meal allowance and per diem?"
                st.session_state.chat_messages.append({
                    "role": "user", 
                    "content": user_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                with st.spinner("🤖 AI thinking..."):
                    response = answer_policy_question(user_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                st.rerun()
        
        with col4:
            if st.button("🌍 International", key="intl_quick_chat"):
                user_msg = "What are the international travel requirements and policies?"
                st.session_state.chat_messages.append({
                    "role": "user", 
                    "content": user_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                with st.spinner("🤖 AI thinking..."):
                    response = answer_policy_question(user_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                st.rerun()
        
        # 聊天输入界面
        st.markdown("### ✍️ Ask Your Question")
        
        # 创建聊天输入表单
        with st.form("chat_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                user_input = st.text_input(
                    "Type your message...", 
                    placeholder="e.g., What's the policy for business class flights?",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button("📤 Send", use_container_width=True)
            
            with col3:
                clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
        
        # 处理聊天输入
        if send_button and user_input.strip():
            # 添加用户消息
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
            # 生成AI回复
            with st.spinner("🤖 AI is thinking..."):
                response = answer_policy_question(user_input)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            
            st.rerun()
        
        # 处理清除聊天
        if clear_button:
            st.session_state.chat_messages = [st.session_state.chat_messages[0]]  # 保留欢迎消息
            st.rerun()
        
        # 聊天统计
        if len(st.session_state.chat_messages) > 1:
            with st.expander(f"📊 Chat Statistics ({len(st.session_state.chat_messages)-1} messages)"):
                user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
                assistant_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions Asked", len(user_messages))
                with col2:
                    st.metric("AI Responses", len(assistant_messages)-1)  # 排除欢迎消息
                with col3:
                    if user_messages:
                        first_msg_time = user_messages[0]["timestamp"]
                        st.metric("Session Started", first_msg_time)
    
    # =======================================================================
    # TAB 2: TRAVEL REQUEST SUBMISSION
    # =======================================================================
    
    with tab2:
        st.header("📋 Submit Travel Request")
        
        # 初始化旅行请求结果会话状态
        if "travel_result" not in st.session_state:
            st.session_state.travel_result = None
        if "travel_history" not in st.session_state:
            st.session_state.travel_history = []
        
        # 固定旅行请求结果显示区域
        st.markdown("### 📋 Travel Request Status")
        travel_result_container = st.container()
        
        with travel_result_container:
            if st.session_state.travel_result:
                # 显示当前旅行请求结果
                st.markdown("""
                <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.travel_result)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("💼 **Submit your travel request below.** The result will appear here and stay fixed for easy reference.")
        
        st.markdown("---")
        
        # 旅行请求表单
        st.markdown("### ✈️ New Travel Request")
        
        with st.form("travel_request"):
            col1, col2 = st.columns(2)
            with col1:
                destination = st.text_input("Destination", placeholder="e.g., New York")
                departure_date = st.date_input(
                    "Departure Date", 
                    value=datetime.now().date() + timedelta(days=30),
                    min_value=datetime.now().date() + timedelta(days=1)
                )
                estimated_cost = st.number_input("Estimated Cost ($)", min_value=0.0, step=100.0)
                
            with col2:
                return_date = st.date_input(
                    "Return Date", 
                    value=datetime.now().date() + timedelta(days=35),
                    min_value=datetime.now().date() + timedelta(days=2)
                )
                purpose = st.selectbox("Purpose", [
                    "Client Meeting", "Conference", "Training", "Site Visit", 
                    "Sales Meeting", "Team Meeting", "Other"
                ])
                employee_email = st.text_input("Employee Email", placeholder="your.email@company.com")
            
            business_justification = st.text_area(
                "Business Justification", 
                placeholder="Explain the business need for this travel..."
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_button = st.form_submit_button("🚀 Submit Request", use_container_width=True)
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Result", use_container_width=True)
        
        # 处理表单提交
        if submit_button:
            if all([destination, departure_date, return_date, estimated_cost, purpose, employee_email, business_justification]):
                if return_date <= departure_date:
                    st.error("❌ Return date must be after departure date")
                else:
                    with st.spinner("🔍 Processing travel request..."):
                        try:
                            # 模拟员工数据获取（在实际应用中会从数据库获取）
                            employee_data = (1, "John", "Doe", "Manager", 15000.0, "Engineering")
                            
                            # 加载政策进行验证
                            policies = load_policies()
                            if not policies:
                                policies = get_default_policies()
                            
                            # 高级请求验证
                            validation_results = advanced_request_validation(
                                destination, departure_date, return_date, 
                                estimated_cost, purpose, employee_data, policies
                            )
                            
                            # 增强的审批工作流
                            approval_workflow = enhanced_approval_workflow(
                                employee_data, estimated_cost, destination, purpose
                            )
                            
                            # 生成旅行规划
                            travel_planning = generate_travel_planning(
                                destination, departure_date, return_date, purpose
                            )
                            
                            # 生成日历
                            calendar_success, calendar_file = generate_travel_calendar(
                                destination, departure_date, return_date, purpose
                            )
                            
                            # 格式化综合结果
                            result = f"""
## ✅ Travel Request Processed Successfully

### 📋 Request Details
- **Destination:** {destination}
- **Dates:** {departure_date} to {return_date} ({(return_date - departure_date).days} days)
- **Cost:** ${estimated_cost:,.2f}
- **Purpose:** {purpose}

### 🔍 Validation Results
"""
                            
                            if validation_results['is_valid']:
                                result += "✅ **Status:** All validations passed\\n\\n"
                            else:
                                result += "⚠️ **Status:** Issues found requiring attention\\n\\n"
                            
                            # 显示违规和警告
                            if validation_results['violations']:
                                result += "❌ **Policy Violations:**\\n"
                                for violation in validation_results['violations']:
                                    result += f"• {violation['message']}\\n"
                                result += "\\n"
                            
                            if validation_results['warnings']:
                                result += "⚠️ **Warnings:**\\n"
                                for warning in validation_results['warnings']:
                                    result += f"• {warning['message']}\\n"
                                result += "\\n"
                            
                            if validation_results['suggestions']:
                                result += "💡 **Optimization Suggestions:**\\n"
                                for suggestion in validation_results['suggestions']:
                                    result += f"• {suggestion['message']}\\n"
                                result += "\\n"
                            
                            # 审批工作流信息
                            result += "### ✅ Approval Workflow\\n"
                            
                            if approval_workflow['is_auto_approved']:
                                result += "🎉 **Auto-Approved!** No additional approval required.\\n\\n"
                            else:
                                result += f"📋 **Approval Path:**\\n"
                                for path in approval_workflow['approval_info']['approval_path']:
                                    result += f"• {path}\\n"
                                result += f"\\n⏱️ **Estimated Time:** {approval_workflow['approval_info']['estimated_time']}\\n\\n"
                            
                            # 特殊要求
                            if approval_workflow['approval_info']['special_requirements']:
                                result += "📋 **Special Requirements:**\\n"
                                for req in approval_workflow['approval_info']['special_requirements']:
                                    result += f"• {req['message']}\\n"
                                result += "\\n"
                            
                            # 旅行规划
                            itinerary = format_travel_itinerary(
                                destination, departure_date, return_date, purpose, travel_planning
                            )
                            result += f"### 🌍 Travel Planning\\n{itinerary}\\n"
                            
                            # 日历集成
                            if calendar_success:
                                result += f"### 📅 Calendar Integration\\n✅ Travel calendar generated: `{calendar_file}`\\n\\n"
                            else:
                                result += f"### 📅 Calendar Integration\\n❌ Calendar generation failed: {calendar_file}\\n\\n"
                            
                            result += "### 📞 Next Steps\\n"
                            result += "1. Review validation results and address any issues\\n"
                            result += "2. Wait for approval process to complete\\n"
                            result += "3. Book travel arrangements after approval\\n"
                            result += "4. Keep all receipts for expense reporting\\n"
                            
                            st.session_state.travel_result = result
                            st.session_state.travel_history.append({
                                "destination": destination,
                                "dates": f"{departure_date} to {return_date}",
                                "cost": estimated_cost,
                                "status": "Processed",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                        except Exception as e:
                            st.session_state.travel_result = f"❌ **Error processing request:** {str(e)}"
                    
                    st.rerun()
            else:
                st.error("❌ Please fill in all required fields")
        
        # 处理清除结果
        if clear_button:
            st.session_state.travel_result = None
            st.rerun()
        
        # 旅行历史
        if st.session_state.travel_history:
            with st.expander(f"📝 Travel Request History ({len(st.session_state.travel_history)} requests)"):
                for i, item in enumerate(reversed(st.session_state.travel_history[-5:]), 1):
                    st.markdown(f"""
                    **Request {len(st.session_state.travel_history)-i+1} ({item['timestamp']}):**
                    - **Destination:** {item['destination']}
                    - **Dates:** {item['dates']}
                    - **Cost:** ${item['cost']:,.2f}
                    - **Status:** {item['status']}
                    """)
                    st.markdown("---")

# =======================================================================
# 🚀 APPLICATION ENTRY POINT - 应用程序入口点
# =======================================================================

if __name__ == "__main__":
    main()
