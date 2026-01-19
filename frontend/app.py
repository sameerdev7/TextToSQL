import streamlit as st
import requests
import pandas as pd
from typing import Optional
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Text-to-SQL Generator",
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
            "error": "Connection failed. Please ensure the backend service is active."
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP Error {e.response.status_code}: {e.response.text}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Header Section
st.title("Text-to-SQL Generator")
st.markdown("Natural language to SQL query generation powered by fine-tuned Phi-3.5 models.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Health Check
    st.subheader("System Status")
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("Backend API: Connected")
        with st.expander("System Metadata"):
            st.json(health_data)
    else:
        st.error("Backend API: Offline")
        st.info("To start the service, run: `uvicorn api:app --reload`")
    
    st.divider()
    
    # Model Parameters
    st.subheader("Inference Parameters")
    context_length = st.slider("Context Length", 512, 2048, 1024, 128)
    max_tokens = st.slider("Maximum Output Tokens", 32, 256, 128, 16)
    temperature = st.slider("Sampling Temperature", 0.0, 1.0, 0.1, 0.05)
    
    st.divider()
    
    # Example Schemas
    st.subheader("Schema Templates")
    example_schemas = {
        "E-commerce": "customers(id, name, email), orders(id, customer_id, date, total), products(id, name, price)",
        "HR Database": "employees(id, name, dept_id, salary, hire_date), departments(id, name, manager_id)",
        "School": "students(id, name, grade), courses(id, title, credits), enrollments(student_id, course_id, semester)",
        "Simple": "users(id, name, age, city)"
    }
    
    selected_example = st.selectbox("Select Template", ["Custom"] + list(example_schemas.keys()))
    
    st.divider()
    
    # Example Questions
    st.subheader("Sample Queries")
    example_questions = [
        "Find all users older than 18",
        "Count total orders by customer",
        "Get top 5 highest paid employees",
        "List students enrolled in CS courses",
        "Show products with price > 100"
    ]
    for q in example_questions:
        st.caption(f"Query: {q}")

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Specification")
    
    # Schema input
    default_schema = example_schemas[selected_example] if selected_example != "Custom" else ""
    
    schema = st.text_area(
        "Database Schema Definition",
        value=default_schema,
        height=150,
        placeholder="Format: table_name(column1, column2)",
        help="Provide the table structures and column names for context."
    )
    
    # Question input
    question = st.text_area(
        "Natural Language Query",
        height=100,
        placeholder="Enter your request in English",
        help="The specific data request you wish to convert to SQL."
    )
    
    # Generate button
    generate_btn = st.button("Generate SQL Statement", type="primary", use_container_width=True)

with col2:
    st.subheader("Result")
    
    if generate_btn:
        if not schema.strip():
            st.error("Action Required: Database schema definition is missing.")
        elif not question.strip():
            st.error("Action Required: Query input is missing.")
        elif not is_healthy:
            st.error("Service Error: The backend API is currently unreachable.")
        else:
            with st.spinner("Processing request..."):
                result = generate_sql(
                    schema=schema,
                    question=question,
                    context_length=context_length,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if result.get("success"):
                st.success("Query Generation Complete")
                
                # Display SQL with syntax highlighting
                sql_query = result.get("sql", "")
                st.code(sql_query, language="sql")
                
                # Copy button
                st.download_button(
                    label="Download SQL File",
                    data=sql_query,
                    file_name="generated_query.sql",
                    mime="text/plain"
                )
                
                # Show raw output in expander
                with st.expander("Model Raw Output"):
                    st.text(result.get("raw_output", ""))
                
            else:
                st.error(f"Generation Error: {result.get('error', 'An unexpected error occurred.')}")

# Batch Processing Section
st.divider()

with st.expander("Batch Operations"):
    st.markdown("### High-Volume Query Processing")
    
    uploaded_file = st.file_uploader(
        "Upload JSON Batch File",
        type=['json'],
        help="Required Format: [{\"schema\": \"...\", \"question\": \"...\"}]"
    )
    
    if uploaded_file:
        try:
            batch_data = json.load(uploaded_file)
            st.info(f"File loaded: {len(batch_data)} queries detected.")
            
            if st.button("Execute Batch Process"):
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
                
                st.success("Batch processing finalized.")
                
                # Display results in a table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="batch_sql_results.csv",
                    mime="text/csv"
                )
                
        except json.JSONDecodeError:
            st.error("Error: The uploaded file is not a valid JSON document.")
        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.8em;'>
        Implementation: Phi-3.5-mini fine-tuned via LoRA | Framework: Streamlit and FastAPI
    </div>
    """,
    unsafe_allow_html=True
)
