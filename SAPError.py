import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os
import re
import traceback
from dotenv import load_dotenv

# Load environment variables for secure API key storage
load_dotenv()

# -----------------------------------------
# Initialize Session State
# -----------------------------------------
# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")
    
if 'model' not in st.session_state:
    st.session_state.model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# -----------------------------------------
# Configuration and Setup
# -----------------------------------------

# App title and description
st.set_page_config(
    page_title="SAP Error Resolution Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066B3;  /* SAP Blue */
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666666;
        margin-bottom: 20px;
    }
    .solution-box {
        background-color: #F5F5F5;
        border-left: 5px solid #0066B3;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .confidence-high {
        color: #00AA00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF0000;
        font-weight: bold;
    }
    .error-display {
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .debug-info {
        background-color: #F5F5F5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.8rem;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Groq API Configuration
# -----------------------------------------

# Constants for Groq API
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to call the Groq API
def query_groq_api(error_details, error_code=None, module=None, system=None):
    """
    Send a query to the Groq API to find solutions for SAP errors.
    
    Parameters:
    - error_details: The detailed error message or description
    - error_code: Optional SAP error code
    - module: Optional SAP module information
    - system: Optional SAP system information
    
    Returns:
    - API response reformatted as a solutions dictionary, or None if the request failed
    """
    # Create a detailed prompt for the Groq API
    prompt = create_sap_error_prompt(error_details, error_code, module, system)
    
    # Get API key and model from session state
    api_key = st.session_state.get('api_key', '')
    model = st.session_state.get('model', 'llama3-70b-8192')
    
    # Prepare headers with authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert SAP consultant with deep knowledge of SAP errors and troubleshooting. Provide structured solutions for SAP errors with detailed steps, confidence scores, and references."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.2,  # Low temperature for more deterministic/factual responses
        "max_tokens": 4000,  # Limit output size
        "response_format": {"type": "json_object"}  # Request JSON formatted response
    }
        
    try:
        # Make the API request
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse and transform the response into the expected format
            solutions = parse_groq_response(response.json(), error_details)
            return solutions
        else:
            # Try to extract JSON from error response if it contains failed_generation
            error_json = response.json()
            if "error" in error_json and "failed_generation" in error_json["error"]:
                # Extract and process the failed JSON generation
                failed_json = error_json["error"]["failed_generation"]
                extracted_solutions = extract_solutions_from_failed_json(failed_json)
                if extracted_solutions:
                    return {
                        "solutions": extracted_solutions,
                        "search_info": {
                            "query": error_details,
                            "timestamp": datetime.now().isoformat(),
                            "note": "Solutions extracted from failed JSON generation",
                            "total_results": len(extracted_solutions),
                            "returned_results": len(extracted_solutions)
                        }
                    }
                
            # If we couldn't extract solutions, show the error
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to Groq API: {str(e)}")
        return None

def create_sap_error_prompt(error_details, error_code=None, module=None, system=None):
    """
    Create a detailed prompt for the Groq API based on SAP error information.
    
    Parameters:
    - error_details: The detailed error message or description
    - error_code: Optional SAP error code
    - module: Optional SAP module information
    - system: Optional SAP system information
    
    Returns:
    - A formatted prompt string
    """
    # Start with the basic error details
    prompt = f"I need solutions for the following SAP error:\n\n{error_details}\n\n"
    
    # Add additional context if available
    if error_code:
        prompt += f"Error code: {error_code}\n"
    if module:
        prompt += f"SAP Module: {module}\n"
    if system:
        prompt += f"SAP System: {system}\n"
        
    # Add instructions for the response format with explicit guidance for URL length
    prompt += """
Please provide 2-4 possible solutions for this error. For each solution, include:

1. A descriptive title for the solution
2. A detailed description of what might be causing the issue
3. A confidence score (0-100) indicating how likely this solution is to resolve the issue
4. Step-by-step instructions to implement the solution
5. References to any relevant SAP notes, documentation, or community resources
6. Tags or categories that relate to this issue

Important: Keep all URLs short (under 100 characters) and valid. If you don't have a specific URL, use a generic placeholder like "https://help.sap.com/" instead of making up a long URL.

Format your response as a JSON object with this structure:
{
  "solutions": [
    {
      "title": "Solution title",
      "description": "Detailed description of the cause",
      "confidence": 85,
      "steps": ["Step 1", "Step 2", "Step 3"],
      "source": {
        "name": "Reference name",
        "url": "https://help.sap.com/docs/example" 
      },
      "tags": ["tag1", "tag2", "tag3"]
    }
  ]
}
"""
    return prompt

def extract_solutions_from_failed_json(failed_json_text):
    """
    Attempt to extract solution data from malformed JSON text.
    This function tries to salvage usable solution data from JSON that failed validation.
    
    Parameters:
    - failed_json_text: The failed JSON text from the API error
    
    Returns:
    - List of solution dictionaries if extraction was successful, otherwise empty list
    """
    try:
        # Remove any markdown code block indicators (```json)
        json_content = re.sub(r'```json|```', '', failed_json_text)
        
        # Try to find solution objects even if the overall JSON is invalid
        solutions = []
        
        # Extract title, description, confidence and steps
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', json_content)
        description_match = re.search(r'"description"\s*:\s*"([^"]+)"', json_content)
        confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', json_content)
        
        # Find all steps using regex
        steps_matches = re.findall(r'"([^"]+)"', json_content[json_content.find('"steps"'):json_content.find('"source"')])
        steps = [step for step in steps_matches if not step.endswith(":") and not step == "steps"]
        
        # Extract tags if present
        tags_text = json_content[json_content.find('"tags"'):] if '"tags"' in json_content else ""
        tags_matches = re.findall(r'"([^"]+)"', tags_text)
        tags = [tag for tag in tags_matches if not tag.endswith(":") and not tag == "tags"]
        
        if title_match and description_match and confidence_match:
            solution = {
                "title": title_match.group(1),
                "description": description_match.group(1),
                "confidence": int(confidence_match.group(1)),
                "steps": steps if steps else ["Detailed steps unavailable due to JSON processing error"],
                "source": {
                    "name": "SAP Documentation",
                    "url": "https://help.sap.com/"
                },
                "tags": tags if tags else ["Error Resolution"]
            }
            solutions.append(solution)
            
        return solutions
    except Exception as e:
        print(f"Error extracting solutions from failed JSON: {str(e)}")
        return []

def parse_groq_response(groq_response, original_query):
    """
    Parse the Groq API response and transform it into the expected solutions format.
    
    Parameters:
    - groq_response: The JSON response from the Groq API
    - original_query: The original error query for reference
    
    Returns:
    - A dictionary with solutions in the expected format
    """
    try:
        # Extract the content from the Groq response
        content = groq_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        
        # Parse the JSON content
        try:
            solutions_data = json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to salvage data
            if "solutions" in content:
                # Try to extract solutions from the text
                extracted_solutions = extract_solutions_from_failed_json(content)
                if extracted_solutions:
                    solutions_data = {"solutions": extracted_solutions}
                else:
                    solutions_data = {"solutions": []}
            else:
                solutions_data = {"solutions": []}
        
        # Validate and fix solutions data
        if "solutions" not in solutions_data:
            solutions_data = {"solutions": []}
        
        # Ensure each solution has all required fields
        for solution in solutions_data.get("solutions", []):
            if "title" not in solution:
                solution["title"] = "Solution"
            if "description" not in solution:
                solution["description"] = "No detailed description available."
            if "confidence" not in solution:
                solution["confidence"] = 50
            if "steps" not in solution:
                solution["steps"] = ["Steps not provided"]
            if "source" not in solution:
                solution["source"] = {"name": "SAP Documentation", "url": "https://help.sap.com/"}
            if "tags" not in solution:
                solution["tags"] = ["Error Resolution"]
                
            # Ensure source is properly formatted
            if isinstance(solution["source"], str):
                solution["source"] = {"name": solution["source"], "url": "https://help.sap.com/"}
                
            # Truncate excessively long URLs to prevent issues
            if isinstance(solution["source"], dict) and "url" in solution["source"]:
                url = solution["source"]["url"]
                if len(url) > 100:
                    solution["source"]["url"] = url[:100]
                    
        # Add search metadata
        result = {
            "solutions": solutions_data.get("solutions", []),
            "search_info": {
                "query": original_query,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(solutions_data.get("solutions", [])),
                "returned_results": len(solutions_data.get("solutions", []))
            }
        }
        
        return result
    except Exception as e:
        st.error(f"Error parsing Groq response: {str(e)}")
        # Return a minimal valid response structure
        return {
            "solutions": [],
            "search_info": {
                "query": original_query,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        }

# -----------------------------------------
# Utility Functions
# -----------------------------------------

# Function to determine confidence level color
def get_confidence_class(confidence_score):
    """
    Return the appropriate CSS class based on the confidence score.
    
    Parameters:
    - confidence_score: Numeric confidence score (assumed 0-100)
    
    Returns:
    - CSS class name as string
    """
    if confidence_score >= 75:
        return "confidence-high"
    elif confidence_score >= 50:
        return "confidence-medium"
    else:
        return "confidence-low"

def parse_sap_error_message(error_text):
    """
    Parse standard SAP error message format to extract key information.
    This function can be expanded based on common SAP error patterns.
    
    Parameters:
    - error_text: Full SAP error message text
    
    Returns:
    - Dictionary with parsed information
    """
    parsed_info = {
        "error_code": None,
        "component": None,
        "message": error_text
    }
    
    # Try to extract error codes that match patterns like "ERROR_CODE: DBIF_REPO_SQL_ERROR"
    error_code_match = re.search(r'(?:ERROR|error|Error)[\s_:]+([A-Z0-9_]+)', error_text)
    if error_code_match:
        parsed_info["error_code"] = error_code_match.group(1)
    
    # Extract component information if present
    component_match = re.search(r'(?:Component|MODULE)[\s:]+([A-Z0-9_/]+)', error_text, re.IGNORECASE)
    if component_match:
        parsed_info["component"] = component_match.group(1)
    
    return parsed_info

# -----------------------------------------
# Streamlit UI Components
# -----------------------------------------

# Main app header
st.markdown('<p class="main-header">SAP Error Resolution Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find solutions to SAP errors using Groq AI</p>', unsafe_allow_html=True)

# Sidebar for app navigation and filters
with st.sidebar:
    st.header("Settings")
    
    # API configuration option
    with st.expander("API Configuration"):
        api_key_input = st.text_input(
            "Groq API Key", 
            value=st.session_state.get('api_key', ''), 
            type="password"
        )
        model_selection = st.selectbox(
            "Groq Model",
            options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        
        if st.button("Save Configuration"):
            st.session_state.api_key = api_key_input
            st.session_state.model = model_selection
            st.success("Configuration saved!")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        debug_mode = st.checkbox("Debug Mode", value=False, 
                               help="Show detailed error information for debugging")
        
        st.markdown("**Retry Options**")
        max_retries = st.number_input("Max API Retries", min_value=1, max_value=5, value=2, 
                                    help="Number of times to retry API call if it fails")
    
    # Filter options
    st.header("Filter Results")
    min_confidence = st.slider("Minimum Confidence Score", 0, 100, 30)
    
    # Option to save search history
    save_history = st.checkbox("Save Search History", value=True)
    
    # Help information
    st.markdown("---")
    st.markdown("""
    **How to use this tool:**
    1. Enter your SAP error message
    2. Add additional details if available
    3. Click "Find Solutions"
    4. Review the suggested solutions
    5. Implement the solution in your SAP system
    
    This application uses the Groq API to generate solutions for SAP errors.
    You must provide your own Groq API key in the Settings panel.
    """)

# Create tabs for different functionality
tab1, tab2, tab3 = st.tabs(["Error Resolution", "Search History", "About"])

# Tab 1: Main error resolution interface
with tab1:
    # Form for user input
    with st.form("error_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Primary error information
            error_details = st.text_area(
                "Enter SAP Error Message or Description",
                height=150,
                help="Paste the complete error message or describe the issue in detail"
            )
            
            error_code = st.text_input(
                "Error Code (if available)",
                help="Enter specific SAP error code, e.g., DBIF_REPO_SQL_ERROR"
            )
        
        with col2:
            # Additional context information
            module = st.text_input(
                "SAP Module",
                help="Specify the SAP module (e.g., FI, MM, SD, ABAP)"
            )
            
            system = st.text_input(
                "SAP System",
                help="Enter your SAP system info (e.g., ECC 6.0, S/4HANA 1909)"
            )
            
            # Optional advanced settings
            with st.expander("Advanced Options"):
                auto_detect = st.checkbox(
                    "Auto-detect Error Code",
                    value=True,
                    help="Automatically try to extract error code from the message"
                )
                
                include_community = st.checkbox(
                    "Include Community Solutions",
                    value=True,
                    help="Include solutions from SAP community and forums"
                )
        
        # Submit button
        submitted = st.form_submit_button("Find Solutions")
    
    # Process the search when form is submitted
    if submitted:
        if not st.session_state.get('api_key'):
            st.warning("Please enter your Groq API key in the Settings panel before searching for solutions.")
        elif not error_details:
            st.warning("Please enter an error message or description")
        else:
            # Auto-detect error code if enabled and not manually provided
            if auto_detect and not error_code:
                parsed_info = parse_sap_error_message(error_details)
                if parsed_info["error_code"]:
                    error_code = parsed_info["error_code"]
                    st.info(f"Auto-detected error code: {error_code}")
            
            # Show a spinner while fetching data
            with st.spinner("Searching for solutions using Groq AI..."):
                # Call the API function with retry logic
                results = None
                errors = []
                
                for attempt in range(max_retries):
                    try:
                        results = query_groq_api(error_details, error_code, module, system)
                        if results:
                            break
                    except Exception as e:
                        errors.append(f"Attempt {attempt+1}: {str(e)}")
                        if debug_mode:
                            st.error(f"Error on attempt {attempt+1}: {str(e)}")
                            st.code(traceback.format_exc())
                
                # Display debug information if enabled
                if debug_mode and errors:
                    st.markdown("### Debug Information")
                    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                    for error in errors:
                        st.write(error)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Save to history if enabled
                if save_history and results and results.get('solutions'):
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': error_details,
                        'error_code': error_code,
                        'results_count': len(results.get('solutions', [])),
                    }
                    st.session_state.search_history.insert(0, history_entry)  # Add to the beginning
            
            # Display results if available
            if results and 'solutions' in results and results['solutions']:
                solutions = results['solutions']
                
                # Filter by confidence score
                filtered_solutions = [s for s in solutions if s.get('confidence', 0) >= min_confidence]
                
                if filtered_solutions:
                    st.success(f"Found {len(filtered_solutions)} relevant solutions")
                    
                    # Display the error details for reference
                    st.markdown("### Your Error")
                    st.markdown(f'<div class="error-display">{error_details}</div>', unsafe_allow_html=True)
                    
                    # Show each solution
                    st.markdown("### Suggested Solutions")
                    
                    for i, solution in enumerate(filtered_solutions):
                        with st.expander(f"Solution {i+1}: {solution.get('title', 'Unnamed Solution')}"):
                            # Confidence score
                            confidence = solution.get('confidence', 0)
                            confidence_class = get_confidence_class(confidence)
                            st.markdown(f'<p>Confidence Score: <span class="{confidence_class}">{confidence}%</span></p>', unsafe_allow_html=True)
                            
                            # Solution details
                            st.markdown("#### Description")
                            st.markdown(solution.get('description', 'No detailed description available.'))
                            
                            st.markdown("#### Steps to Resolve")
                            if 'steps' in solution and solution['steps']:
                                for step_num, step in enumerate(solution['steps'], 1):
                                    st.markdown(f"**Step {step_num}:** {step}")
                            
                            # Source information
                            if 'source' in solution and solution['source']:
                                st.markdown("#### Source")
                                source = solution['source']
                                if isinstance(source, dict) and 'url' in source and 'name' in source:
                                    st.markdown(f"[{source.get('name', 'Reference')}]({source.get('url', '#')})")
                                else:
                                    st.markdown(str(source))
                            
                            # Tags or categories
                            if 'tags' in solution and solution['tags']:
                                st.markdown("#### Related Topics")
                                tags_html = ', '.join([f'<span style="background-color: #E0E0E0; padding: 3px 8px; border-radius: 10px; margin-right: 5px;">{tag}</span>' for tag in solution['tags']])
                                st.markdown(f'<p>{tags_html}</p>', unsafe_allow_html=True)
                else:
                    st.warning(f"Found {len(solutions)} solutions, but none meet the minimum confidence threshold of {min_confidence}%.")
            else:
                st.error("No solutions found. Try adjusting your search terms or providing more details about the error.")

# Tab 2: Search History
with tab2:
    st.header("Search History")
    
    if not st.session_state.search_history:
        st.info("Your search history will appear here")
    else:
        # Convert history to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.search_history)
        
        # Add buttons to clear history
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.experimental_rerun()
        
        # Display history
        st.dataframe(history_df, use_container_width=True)
        
        # Option to export history
        if st.download_button(
            "Export History as CSV",
            data=history_df.to_csv(index=False).encode('utf-8'),
            file_name=f"sap_error_search_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        ):
            st.success("History exported successfully")

# Tab 3: About the app
with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### SAP Error Resolution Assistant
    
    This application is designed to help SAP users and administrators quickly find solutions 
    to errors they encounter within their SAP systems. By leveraging the powerful Groq API, 
    the application can generate contextually relevant solutions based on the error information 
    you provide.
    
    ### Features
    
    - **AI-Powered Solutions**: Generate solutions for SAP errors using Groq's advanced language models
    - **Context-Aware Solutions**: Include SAP module, error codes, and system information for more targeted results
    - **Confidence Scoring**: Each solution has a confidence score to indicate relevance
    - **Search History**: Keep track of previous errors and solutions
    - **Robust Error Handling**: Recover useful information even when API responses have issues
    - **Secure API Handling**: Your API keys are stored securely
    
    ### About the Groq API
    
    Groq provides powerful large language model APIs that can understand and generate text, code, and solutions. 
    To use this application, you'll need to obtain an API key from Groq's service.
    
    You can sign up for a Groq API key at [groq.com](https://www.groq.com).
    
    ### Privacy & Data Handling
    
    This application does not store your SAP error data beyond your current session unless 
    you enable the history feature. All API requests are made securely, and your API keys are 
    not shared with any third parties.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 SAP Error Resolution Assistant | Powered by Groq API")