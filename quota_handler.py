# quota_handler.py - Quota Management Functions

def create_local_policy_search(policy_documents):
    """Create a local policy search function without AI dependencies"""
    def local_policy_search(question):
        # Get policy context from database
        question_lower = question.lower()
        relevant_policies = []
        
        # Simple keyword matching
        keywords = ['flight', 'hotel', 'meal', 'travel', 'approval', 'cost', 'budget', 'expense', 'per diem']
        
        for doc in policy_documents:
            content_lower = doc.page_content.lower()
            # Check if question keywords match policy content
            if any(keyword in question_lower and keyword in content_lower for keyword in keywords):
                relevant_policies.append(doc.page_content)
        
        if not relevant_policies:
            relevant_policies = [doc.page_content for doc in policy_documents[:2]]  # First 2 policies
        
        # Format response without AI
        response = "**📋 Travel Policy Information:**\n\n"
        for i, policy in enumerate(relevant_policies[:3], 1):
            response += f"**Policy {i}:**\n{policy}\n\n"
        
        response += "💡 **Note**: This is a direct policy lookup (AI quota exceeded). For detailed analysis, please try again tomorrow or upgrade your Google API plan.\n\n"
        response += "📞 **Need Help?** Contact your travel administrator for specific questions."
        
        return {"result": response}
    
    return local_policy_search

def create_quota_aware_qa_chain(policy_documents):
    """Create QA chain with quota management"""
    
    # Try Google AI embeddings first
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return None, embedding_function  # Success - return embedding function
    except Exception as e:
        # Check if it's a quota error
        if "quota" in str(e).lower() or "429" in str(e):
            return create_local_policy_search(policy_documents), None
        else:
            # Try simple AI fallback
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                
                def simple_qa_chain(question):
                    policy_context = "\n".join([doc.page_content for doc in policy_documents])
                    
                    prompt = f"""Based on the following travel policies, answer the question:

Travel Policies:
{policy_context}

Question: {question}

Provide a helpful answer based on the policies above."""
                    
                    response = llm.invoke(prompt)
                    return {"result": response.content}
                
                return simple_qa_chain, None
            except Exception as ai_error:
                if "quota" in str(ai_error).lower() or "429" in str(ai_error):
                    return create_local_policy_search(policy_documents), None
                else:
                    return create_local_policy_search(policy_documents), None
