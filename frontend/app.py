import streamlit as st
import requests
import pandas as pd
from typing import Optional
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Text-to-SQL Generator",
    page_icon="🗄️",
    layout="wide"
)

def check_api_health():
    """Check if the backend API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def generate_sql(schema: str, question: str, context_length: int = 1024, 
                 max_tokens: int = 128, temperature: float = 0.1):
    """Call the API to generate SQL."""
    try:
        payload = {
            "db_schema": schema,
            "question": question,
            "context_length": context_length,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{API_URL}/generate-sql",
            json=payload,
            timeout=None
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to API. Make sure the backend is running."
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP Error: {e.response.status_code} - {e.response.text}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Header
st.title("🗄️ Text-to-SQL Generator")
st.markdown("Convert natural language questions into SQL queries using a fine-tuned Phi-3.5 model")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Health Check
    st.subheader("API Status")
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ Backend API is running")
        with st.expander("API Details"):
            st.json(health_data)
    else:
        st.error("❌ Backend API is not available")
        st.warning("Start the backend with: `uvicorn api:app --reload`")
    
    st.divider()
    
    # Model Parameters
    st.subheader("Model Parameters")
    context_length = st.slider("Context Length", 512, 2048, 1024, 128)
    max_tokens = st.slider("Max Output Tokens", 32, 256, 128, 16)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    
    st.divider()
    
    # Example Schemas
    st.subheader("📚 Example Schemas")
    example_schemas = {
        "E-commerce": "customers(id, name, email), orders(id, customer_id, date, total), products(id, name, price)",
        "HR Database": "employees(id, name, dept_id, salary, hire_date), departments(id, name, manager_id)",
        "School": "students(id, name, grade), courses(id, title, credits), enrollments(student_id, course_id, semester)",
        "Simple": "users(id, name, age, city)"
    }
    
    selected_example = st.selectbox("Load Example Schema", ["Custom"] + list(example_schemas.keys()))
    
    st.divider()
    
    # Example Questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "Find all users older than 18",
        "Count total orders by customer",
        "Get top 5 highest paid employees",
        "List students enrolled in CS courses",
        "Show products with price > 100"
    ]
    for q in example_questions:
        st.caption(f"• {q}")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    
    # Schema input
    if selected_example != "Custom":
        default_schema = example_schemas[selected_example]
    else:
        default_schema = ""
    
    schema = st.text_area(
        "Database Schema",
        value=default_schema,
        height=150,
        placeholder="e.g., users(id, name, age), orders(id, user_id, amount)",
        help="Describe your database tables and columns"
    )
    
    # Question input
    question = st.text_area(
        "Natural Language Question",
        height=100,
        placeholder="e.g., Find all users who are older than 25",
        help="Ask your question in plain English"
    )
    
    # Generate button
    generate_btn = st.button("🚀 Generate SQL", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Output")
    
    if generate_btn:
        if not schema.strip():
            st.error("❌ Please provide a database schema")
        elif not question.strip():
            st.error("❌ Please enter a question")
        elif not is_healthy:
            st.error("❌ Backend API is not available. Please start the backend server.")
        else:
            with st.spinner("Generating SQL..."):
                result = generate_sql(
                    schema=schema,
                    question=question,
                    context_length=context_length,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if result.get("success"):
                st.success("✅ SQL Generated Successfully!")
                
                # Display SQL with syntax highlighting
                sql_query = result.get("sql", "")
                st.code(sql_query, language="sql")
                
                # Copy button
                st.download_button(
                    label="📋 Copy SQL",
                    data=sql_query,
                    file_name="query.sql",
                    mime="text/plain"
                )
                
                # Show raw output in expander
                with st.expander("🔍 View Raw Model Output"):
                    st.text(result.get("raw_output", ""))
                
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# Footer with batch processing
st.divider()

with st.expander("🔄 Batch Processing"):
    st.markdown("### Upload multiple questions at once")
    
    uploaded_file = st.file_uploader(
        "Upload JSON file with questions",
        type=['json'],
        help="Format: [{\"schema\": \"...\", \"question\": \"...\"}]"
    )
    
    if uploaded_file:
        try:
            batch_data = json.load(uploaded_file)
            st.info(f"Loaded {len(batch_data)} questions")
            
            if st.button("Process Batch"):
                results = []
                progress_bar = st.progress(0)
                
                for i, item in enumerate(batch_data):
                    result = generate_sql(
                        schema=item.get("schema", ""),
                        question=item.get("question", ""),
                        context_length=context_length,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    results.append({
                        "question": item.get("question"),
                        "sql": result.get("sql", ""),
                        "success": result.get("success", False)
                    })
                    progress_bar.progress((i + 1) / len(batch_data))
                
                st.success("✅ Batch processing complete!")
                
                # Display results in a table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="sql_results.csv",
                    mime="text/csv"
                )
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Powered by Phi-3.5-mini fine-tuned with LoRA | Built with Streamlit & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)
