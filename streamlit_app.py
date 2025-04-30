import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import time
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from typing import Dict, List, Any, Optional

# Import custom modules
from extractor.pdf_extractor import PDFExtractor
from extractor.llm_extractor import LLMExtractor
from mapper.entity_mapper import EntityMapper

# Set page configuration
st.set_page_config(
    page_title="Boon AI Hackathon - Document Processing & Entity Mapping",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #306998;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #5A5A5A;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }
    .info-box {
        background-color: #f0f2f6;
        color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d1e7dd;
        color: #0F5132;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #664D03;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        color: #842029;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4B8BBE;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'extracted_markdown' not in st.session_state:
    st.session_state.extracted_markdown = None
if 'extracted_json' not in st.session_state:
    st.session_state.extracted_json = None
if 'mapped_entities' not in st.session_state:
    st.session_state.mapped_entities = None
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = "pymupdf"
if 'extracted_tables' not in st.session_state:
    st.session_state.extracted_tables = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'extracted_structure' not in st.session_state:
    st.session_state.extracted_structure = None
if 'has_error' not in st.session_state:
    st.session_state.has_error = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = {}
if 'manual_edits' not in st.session_state:
    st.session_state.manual_edits = {}
if 'data_displayed' not in st.session_state:
    st.session_state.data_displayed = False

def main():
    """Main function for the Streamlit app"""
    st.markdown("<h1 class='main-header'>Boon AI Hackathon: Document Processing & Entity Mapping</h1>", unsafe_allow_html=True)

    create_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Extract", 
        "🔍 Entity Mapping", 
        "📊 Evaluation & Results",
        "⚙️ Manual Editing",
        "📁 Export Results"
    ])

    with tab1:
        upload_and_extract_section()
    
    with tab2:
        entity_mapping_section()
    
    with tab3:
        evaluation_section()
    
    with tab4:
        manual_editing_section()
    
    with tab5:
        export_results_section()

def create_sidebar():
    """Create the sidebar with settings and controls including schema type selection"""
    st.sidebar.header("⚙️ Settings")
    
    # Extraction method selection
    st.sidebar.markdown("#### Extraction Method")
    extraction_method = st.sidebar.radio(
        "Select text extraction method:",
        ["PyMuPDF", "OCR", "Use both and compare"],
        index=0
    )
    
    if extraction_method == "PyMuPDF":
        st.session_state.extraction_method = "pymupdf"
    elif extraction_method == "OCR":
        st.session_state.extraction_method = "ocr"
    else:
        st.session_state.extraction_method = "both"
    
    # Schema type selection
    st.sidebar.markdown("#### Schema Type")
    schema_type = st.sidebar.radio(
        "Select schema type:",
        ["Predefined Schema", "Dynamic Schema (from JSON)"],
        index=0
    )
    
    st.session_state.schema_type = "predefined" if schema_type == "Predefined Schema" else "dynamic"
    
    # LLM selection
    st.sidebar.markdown("#### LLM Provider")
    llm_provider = st.sidebar.radio(
        "Select LLM provider:",
        ["OpenAI", "Anthropic"],
        index=0
    )
    
    st.session_state.llm_provider = llm_provider.lower()
    
    if llm_provider == "OpenAI":
        st.session_state.model_name = st.sidebar.selectbox(
            "Select OpenAI model:",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
    else:
        st.session_state.model_name = st.sidebar.selectbox(
            "Select Anthropic model:",
            ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            index=0
        )
    
    # Extraction optimization settings
    st.sidebar.markdown("#### Extraction Optimization")
    st.session_state.use_caching = st.sidebar.checkbox(
        "Enable result caching",
        value=True
    )
    
    st.session_state.use_chunking = st.sidebar.checkbox(
        "Enable chunking for large documents",
        value=True
    )
    
    # Entity mapping settings
    st.sidebar.markdown("#### Entity Mapping")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Entity matching threshold:",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05
    )
    
    st.session_state.use_embeddings = st.sidebar.checkbox(
        "Use sentence embeddings for matching",
        value=True
    )

def upload_and_extract_section():
    """Function for the upload and extract section with persistent data display"""
    st.markdown("<h2 class='section-header'>📤 Document Upload & Data Extraction</h2>", unsafe_allow_html=True)
    
    # Display schema type info
    schema_type = "Predefined" if st.session_state.schema_type == "predefined" else "Dynamic"
    optimization_info = []
    if st.session_state.use_caching:
        optimization_info.append("Caching")
    if st.session_state.use_chunking:
        optimization_info.append("Chunking")
    
    optimizations = ", ".join(optimization_info) if optimization_info else "None"
    
    st.markdown(f"""
    <div class='info-box'>
        Upload your PDF document, markdown file, and JSON structure to extract information.<br>
        <b>Schema:</b> {schema_type}<br>
        <b>LLM:</b> {st.session_state.model_name}<br>
        <b>Optimizations:</b> {optimizations}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Upload PDF</h3>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Upload PDF document", type=["pdf"])
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Upload Markdown (Optional)</h3>", unsafe_allow_html=True)
        markdown_file = st.file_uploader("Upload markdown file (optional)", type=["md", "txt"])
    
    with col3:
        st.markdown("<h3 class='subsection-header'>Upload JSON Structure</h3>", unsafe_allow_html=True)
        json_file = st.file_uploader("Upload JSON structure", type=["json"])
    
    # Display schema type details for information
    if st.session_state.schema_type == "predefined":
        with st.expander("View Predefined Schema Fields"):
            schema = get_predefined_schema()
            st.write("The predefined schema includes these fields:")
            st.write(", ".join(schema.get("properties", {}).keys()))
            
            st.write("Required fields:")
            st.write(", ".join(schema.get("required", [])))
    
    # Process button
    if st.button("Process Documents"):
        if pdf_file is None or json_file is None:
            st.error("Please upload PDF and JSON files to proceed.")
            return
        
        try:
            st.session_state.has_error = False
            
            with st.spinner("Processing documents..."):
                # Process JSON structure first
                process_json(json_file)
                
                # Generate a hash of the PDF content for caching
                pdf_bytes = pdf_file.getvalue()
                import hashlib
                st.session_state.pdf_hash = hashlib.md5(pdf_bytes[:5000]).hexdigest()
                
                # Extract text from PDF
                process_pdf(pdf_file)
                
                # Process Markdown if provided
                if markdown_file is not None:
                    process_markdown(markdown_file)
            
            # Set the flag to indicate data has been processed and should be displayed
            st.session_state.data_displayed = True
            
            st.success("Processing complete!")
        
        except Exception as e:
            st.session_state.has_error = True
            st.session_state.error_message = str(e)
            st.error(f"Error processing documents: {str(e)}")
    
    # Always display data if it's available and has been processed before
    if st.session_state.data_displayed and st.session_state.extracted_text is not None:
        display_extracted_data()
        
def process_pdf(pdf_file):
    """Process the uploaded PDF file with better error handling"""
    start_time = time.time()
    
    try:
        # Create PDF extractor with the selected method
        if st.session_state.extraction_method == "pymupdf":
            extractor = PDFExtractor(use_ocr=False)
            st.session_state.extracted_text = extractor.extract_from_bytes(pdf_file.getvalue())
            
            # Verify that extracted_text is not None
            if st.session_state.extracted_text is None:
                st.session_state.extracted_text = {0: "No text could be extracted from the PDF."}
                
        elif st.session_state.extraction_method == "ocr":
            extractor = PDFExtractor(use_ocr=True)
            st.session_state.extracted_text = extractor.extract_from_bytes(pdf_file.getvalue())
            
            # Verify that extracted_text is not None
            if st.session_state.extracted_text is None:
                st.session_state.extracted_text = {0: "No text could be extracted from the PDF using OCR."}
                
        else:
            # Use both methods and compare
            extractor = PDFExtractor(use_ocr=False)
            pymupdf_text = extractor.extract_from_bytes(pdf_file.getvalue())
            
            ocr_text = extractor.extract_text_with_ocr(pdf_file.getvalue())
            
            # Compare methods and select the best one
            comparison = extractor.compare_extraction_methods(pdf_file.getvalue())
            
            # Initialize extracted_text if not already
            st.session_state.extracted_text = {}
            
            # Ensure we have valid data
            if not comparison or not pymupdf_text or not ocr_text:
                st.session_state.extracted_text = {0: "Error comparing extraction methods."}
            else:
                # Use the better method for each page
                for page_num in comparison:
                    if comparison[page_num]["pymupdf_length"] >= comparison[page_num]["ocr_length"]:
                        st.session_state.extracted_text[page_num] = pymupdf_text[page_num]
                    else:
                        st.session_state.extracted_text[page_num] = ocr_text[page_num]
            
            st.session_state.extraction_comparison = comparison
        
        # Extract tables from PDF
        try:
            extractor = PDFExtractor(use_ocr=False)
            st.session_state.extracted_tables = extractor.extract_tables_from_pdf(pdf_file.getvalue())
        except Exception as e:
            st.warning(f"Error extracting tables: {str(e)}")
            st.session_state.extracted_tables = []
        
        # Calculate processing time
        st.session_state.processing_time["pdf_extraction"] = time.time() - start_time
        
    except Exception as e:
        st.error(f"Error in PDF extraction: {str(e)}")
        st.session_state.extracted_text = {0: f"Error extracting text: {str(e)}"}
        st.session_state.processing_time["pdf_extraction"] = time.time() - start_time

def process_json(json_file):
    """Process the uploaded JSON structure with predefined or dynamic schema"""
    start_time = time.time()
    
    try:
        json_content = json.load(json_file)
        st.session_state.extracted_json = json_content
        
        # Generate a hash of the JSON content for caching
        json_str = json.dumps(json_content, sort_keys=True)
        import hashlib
        st.session_state.json_hash = hashlib.md5(json_str.encode()).hexdigest()
        
        # Use predefined schema if selected, otherwise generate dynamic schema
        if st.session_state.schema_type == "predefined":
            st.session_state.json_schema = get_predefined_schema()
            
            # Validate the predefined schema against the actual JSON
            # This ensures the predefined schema works with the data
            missing_fields = []
            for field in st.session_state.json_schema.get("required", []):
                if field not in json_content:
                    missing_fields.append(field)
            
            if missing_fields:
                st.warning(f"The predefined schema requires fields that are not in the JSON: {', '.join(missing_fields)}")
                
                # Adjust the required fields to match what's available
                adjusted_required = [field for field in st.session_state.json_schema.get("required", []) 
                                    if field in json_content]
                st.session_state.json_schema["required"] = adjusted_required
        else:
            # Generate dynamic JSON schema based on the actual content
            properties = {}
            for key, value in json_content.items():
                if isinstance(value, str):
                    properties[key] = {
                        "type": "string",
                        "description": f"The {key} field"
                    }
                elif isinstance(value, (int, float)):
                    properties[key] = {
                        "type": "number",
                        "description": f"The {key} field"
                    }
                elif isinstance(value, bool):
                    properties[key] = {
                        "type": "boolean",
                        "description": f"The {key} field"
                    }
                elif isinstance(value, list):
                    properties[key] = {
                        "type": "array",
                        "description": f"The {key} field which contains a list of items"
                    }
                elif isinstance(value, dict):
                    properties[key] = {
                        "type": "object",
                        "description": f"The {key} field which contains nested properties"
                    }
            
            # Set priority fields that should be extracted first
            priority_fields = ["id", "blnum", "customer_id", "commodity", "status", "ordered_date"]
            required_fields = [field for field in priority_fields if field in json_content]
            
            # Add other fields up to a reasonable limit
            other_fields = [field for field in json_content.keys() 
                           if field not in required_fields and field not in ["stops", "movements", "freightGroup"]]
            
            # Limit to 20 total fields to keep extraction focused
            max_fields = 20
            if len(required_fields) + len(other_fields) > max_fields:
                other_fields = other_fields[:max_fields - len(required_fields)]
            
            required_fields.extend(other_fields)
            
            st.session_state.json_schema = {
                "type": "object",
                "properties": properties,
                "required": required_fields
            }
        
        # Calculate processing time
        st.session_state.processing_time["json_processing"] = time.time() - start_time
    
    except json.JSONDecodeError:
        raise Exception("Invalid JSON file. Please upload a valid JSON file.")
    
def process_markdown(markdown_file):
    """Process the uploaded markdown file"""
    start_time = time.time()
    
    try:
        # Read markdown content
        markdown_content = markdown_file.getvalue().decode("utf-8")
        
        # Extract text from markdown
        extractor = PDFExtractor()
        extracted_text = extractor.extract_text_from_markdown(markdown_content)
        
        st.session_state.extracted_markdown = extracted_text
        
        # Calculate processing time
        st.session_state.processing_time["markdown_processing"] = time.time() - start_time
    
    except Exception as e:
        raise Exception(f"Error processing markdown: {str(e)}")

def display_extracted_data():
    """Display the extracted data with better error handling"""
    st.markdown("<h3 class='subsection-header'>Extracted Data</h3>", unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Text View", "Tables", "Comparison"])
    
    with tab1:
        # Display extracted text with error handling
        st.markdown("#### Text Extracted from PDF")
        
        if st.session_state.extracted_text is None:
            st.warning("No text has been extracted from the PDF yet.")
        else:
            try:
                combined_text = ""
                for page_num in sorted(st.session_state.extracted_text.keys()):
                    combined_text += f"Page {page_num + 1}:\n{st.session_state.extracted_text[page_num]}\n\n"
                
                st.text_area("Extracted Text", combined_text, height=300)
            except Exception as e:
                st.error(f"Error displaying extracted text: {str(e)}")
                st.info("The PDF text might not have been extracted correctly.")
        
        # Display markdown text if available
        if st.session_state.extracted_markdown is not None:
            st.markdown("#### Text Extracted from Markdown")
            st.text_area("Markdown Text", st.session_state.extracted_markdown, height=200)
        
        # Display JSON structure with error handling
        st.markdown("#### JSON Structure")
        if st.session_state.extracted_json is None:
            st.warning("No JSON structure has been processed yet.")
        else:
            try:
                st.json(st.session_state.extracted_json)
            except Exception as e:
                st.error(f"Error displaying JSON structure: {str(e)}")
                st.info("The JSON might not have been processed correctly.")
    
    with tab2:
        # Display extracted tables
        st.markdown("#### Tables Extracted from PDF")
        
        if not st.session_state.extracted_tables:
            st.info("No tables found in the document.")
        else:
            for i, table in enumerate(st.session_state.extracted_tables):
                st.markdown(f"##### Table {i+1}")
                
                # Convert table to pandas DataFrame
                try:
                    df = pd.DataFrame(table)
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error displaying table {i+1}: {str(e)}")
    
    with tab3:
        # Display comparison of extraction methods if available
        if "extraction_comparison" in st.session_state:
            st.markdown("#### Comparison of Extraction Methods")
            
            try:
                comparison_data = []
                for page_num, data in st.session_state.extraction_comparison.items():
                    comparison_data.append({
                        "Page": page_num + 1,
                        "PyMuPDF Characters": data["pymupdf_length"],
                        "OCR Characters": data["ocr_length"],
                        "Difference (%)": data["difference_percentage"]
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Create visualization
                    fig = px.bar(
                        comparison_df,
                        x="Page",
                        y=["PyMuPDF Characters", "OCR Characters"],
                        barmode="group",
                        title="Character Count Comparison by Page"
                    )
                    
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error displaying comparison data: {str(e)}")
        else:
            st.info("Comparison only available when 'Use both and compare' extraction method is selected.")

def get_predefined_schema():
    """
    Return the predefined schema for TMS orders with additional fields
    
    This schema is expanded to include fields from logistics load confirmations,
    with proper field descriptions and types.
    """
    return {
        "type": "object",
        "properties": {
            # Primary order identifiers
            "id": {"type": "string", "description": "Order ID that uniquely identifies the order in the system"},
            "blnum": {"type": "string", "description": "Bill of lading number for the shipment"},
            "shipment_id": {"type": "string", "description": "Shipment identifier if different from order ID"},
            "work_order": {"type": "string", "description": "Work order number (e.g., W/O: 79575)"},
            "purchase_order": {"type": "string", "description": "Purchase order number for the shipment"},
            
            # Company information
            "company_id": {"type": "string", "description": "Company identifier in the system"},
            "company_address": {"type": "string", "description": "Address of the logistics company"},
            "company_phone": {"type": "string", "description": "Phone number of the logistics company"},
            "company_fax": {"type": "string", "description": "Fax number of the logistics company"},
            
            # Customer and basic order information
            "customer_id": {"type": "string", "description": "Customer ID that identifies the customer in the system"},
            "ordered_date": {"type": "string", "description": "Date and time when the order was entered into the system"},
            "status": {"type": "string", "description": "Current status code of the order (e.g., A for Available, Open)"},
            "operational_status": {"type": "string", "description": "Operational status code (e.g., CLIN for Client)"},
            "order_mode": {"type": "string", "description": "Mode of the order (e.g., T for Transport)"},
            "delivery_date": {"type": "string", "description": "Date when the shipment is scheduled to be delivered"},
            
            # Contact information
            "dispatcher_phone": {"type": "string", "description": "Phone number of the dispatcher"},
            "dispatcher_email": {"type": "string", "description": "Email address of the dispatcher"},
            
            # Carrier information
            "carrier_phone": {"type": "string", "description": "Phone number of the carrier"},
            "carrier_fax": {"type": "string", "description": "Fax number of the carrier"},
            "driver_name": {"type": "string", "description": "Name of the driver assigned to the shipment"},
            "driver_phone": {"type": "string", "description": "Phone number of the driver"},
            "truck_number": {"type": "string", "description": "Identification number of the truck"},
            "trailer_number": {"type": "string", "description": "Identification number of the trailer"},
            
            # Commodity information
            "commodity": {"type": "string", "description": "Description of the commodities being transported"},
            "commodity_id": {"type": "string", "description": "ID code for the commodity type"},
            "hazmat": {"type": "boolean", "description": "Indicates if the load contains hazardous materials"},
            "weight": {"type": "number", "description": "Weight of the shipment in pounds or other unit"},
            "weight_um": {"type": "string", "description": "Weight unit of measure (e.g., LB for pounds)"},
            "pieces": {"type": "number", "description": "Number of pieces in the shipment"},
            "load_type": {"type": "string", "description": "Type of load (e.g., TL for Truckload, LTL for Less Than Truckload)"},
            "loading_type": {"type": "string", "description": "How the cargo is loaded (e.g., Floor Loaded, Palletized)"},
            
            # Equipment information
            "equipment_type_id": {"type": "string", "description": "Equipment type identifier (e.g., V for Van)"},
            "ltl": {"type": "boolean", "description": "Indicates if this is a less-than-truckload shipment"},
            
            # Shipper information
            "shipper_address": {"type": "string", "description": "Address of the shipper/origin location"},
            "shipper_hours": {"type": "string", "description": "Operating hours of the shipper/origin location"},
            "shipper_appointment": {"type": "boolean", "description": "Indicates if appointment is required at shipper"},
            "shipper_notes": {"type": "string", "description": "Special notes or instructions for the shipper location"},
            
            # Consignee information
            "consignee_address": {"type": "string", "description": "Address of the consignee/destination location"},
            "consignee_hours": {"type": "string", "description": "Operating hours of the consignee/destination location"},
            "consignee_appointment": {"type": "boolean", "description": "Indicates if appointment is required at consignee"},
            "consignee_notes": {"type": "string", "description": "Special notes or instructions for the consignee location"},
            
            # Distance and route information
            "bill_distance": {"type": "number", "description": "Billing distance for the shipment"},
            "bill_distance_um": {"type": "string", "description": "Distance unit of measure (e.g., MI for miles)"},
            
            # Financial information
            "collection_method": {"type": "string", "description": "Collection method code (e.g., P for Prepaid)"},
            "freight_charge": {"type": "number", "description": "Freight charge amount"},
            "total_charge": {"type": "number", "description": "Total charge amount for the shipment"},
            "rate": {"type": "number", "description": "Rate amount"},
            "rate_type": {"type": "string", "description": "Type of rate (e.g., F for Flat)"},
            "rate_units": {"type": "number", "description": "Number of rate units"},
            "revenue_code_id": {"type": "string", "description": "Revenue code identifier"},
            "line_haul": {"type": "string", "description": "Line haul charge amount"},
            "payment_terms": {"type": "string", "description": "Terms and conditions for payment"},
            "currency": {"type": "string", "description": "Currency for payment (e.g., USD)"},
            
            # Stop information
            "shipper_stop_id": {"type": "string", "description": "ID of the shipper/origin stop"},
            "consignee_stop_id": {"type": "string", "description": "ID of the consignee/destination stop"},
            
            # User information
            "entered_user_id": {"type": "string", "description": "ID of the user who entered the order"},
            "operations_user": {"type": "string", "description": "ID of the operations user assigned to the order"},
            
            # Movement information
            "curr_movement_id": {"type": "string", "description": "Current movement ID"},
            "def_move_type": {"type": "string", "description": "Default movement type"},
            
            # Additional notes
            "dispatch_notes": {"type": "string", "description": "Notes and instructions for the dispatcher"},
            "special_instructions": {"type": "string", "description": "Special instructions for handling the shipment"}
        },
        "required": [
            "id", "customer_id", "commodity", "status", 
            "equipment_type_id", "bill_distance", "freight_charge", "total_charge", 
            "rate", "rate_type", "operational_status", "operations_user", 
            "shipper_stop_id", "consignee_stop_id", "collection_method"
        ]
    }

def entity_mapping_section():
    """Function for the entity mapping section"""
    st.markdown("<h2 class='section-header'>🔍 Entity Mapping</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Extract fields from the document and map them to the database schema.</div>", unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.extracted_text is None or st.session_state.extracted_json is None:
        st.warning("Please upload and process documents first.")
        return
    
    # Extract and map button
    if st.button("Extract Fields and Map Entities"):
        with st.spinner("Extracting fields and mapping entities..."):
            extract_and_map_entities()
    
    # Display results if available
    if st.session_state.mapped_entities is not None and not st.session_state.has_error:
        display_mapping_results()

def extract_and_map_entities():
    """Extract fields from text and map entities with caching and optimizations"""
    try:
        st.session_state.has_error = False
        
        # Generate a cache key based on the content
        if "extraction_cache" not in st.session_state:
            st.session_state.extraction_cache = {}
        
        # Create a simple cache key based on basic document properties
        cache_key = None
        if st.session_state.extracted_text and 0 in st.session_state.extracted_text:
            # Create key from first 100 chars of first page, schema type, and provider
            text_sample = st.session_state.extracted_text[0][:100]
            schema_type = st.session_state.schema_type
            provider = st.session_state.llm_provider
            model = st.session_state.model_name
            cache_key = f"{text_sample}_{schema_type}_{provider}_{model}"
            
            # Convert to hash to ensure it's a valid filename
            import hashlib
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check if we have cached results
        if cache_key and cache_key in st.session_state.extraction_cache:
            st.session_state.extracted_structure = st.session_state.extraction_cache[cache_key]
            st.session_state.validation_results = {"status": "Using cached validation"}
            st.info("Using cached extraction results")
        else:
            # Extract fields using LLM
            extract_fields_using_llm()
            
            # Cache the results if extraction was successful
            if cache_key and st.session_state.extracted_structure:
                st.session_state.extraction_cache[cache_key] = st.session_state.extracted_structure
        
        # Map entities
        map_entities()
        
        st.success("Extraction and mapping complete!")
    
    except Exception as e:
        st.session_state.has_error = True
        st.session_state.error_message = str(e)
        st.error(f"Error in extraction and mapping: {str(e)}")

def extract_fields_using_llm():
    """Extract fields from text using LLM with optimizations"""
    start_time = time.time()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Preparing document for extraction...")
    progress_bar.progress(10)
    
    # Combine text from all pages (up to a reasonable limit)
    combined_text = ""
    total_chars = 0
    max_chars = 10000  # Set a reasonable limit for text length
    
    for page_num in sorted(st.session_state.extracted_text.keys()):
        page_text = st.session_state.extracted_text[page_num]
        if total_chars + len(page_text) <= max_chars:
            combined_text += page_text + "\n\n"
            total_chars += len(page_text)
        else:
            # Add partial text to fill up to the limit
            remaining = max_chars - total_chars
            if remaining > 0:
                combined_text += page_text[:remaining] + "..."
            break
    
    # Add markdown text if available and within limit
    if st.session_state.extracted_markdown is not None and total_chars < max_chars:
        remaining = max_chars - total_chars
        markdown_text = st.session_state.extracted_markdown
        if len(markdown_text) <= remaining:
            combined_text += "\n\nMarkdown Content:\n" + markdown_text
        else:
            combined_text += "\n\nMarkdown Content:\n" + markdown_text[:remaining] + "..."
    
    status_text.text("Initializing LLM extractor...")
    progress_bar.progress(20)
    
    # Initialize LLM extractor
    llm_extractor = LLMExtractor(
        model_provider=st.session_state.llm_provider,
        model_name=st.session_state.model_name
    )
    
    # Decide on extraction approach based on text size and schema complexity
    status_text.text("Determining optimal extraction strategy...")
    progress_bar.progress(30)
    
    # Get schema size
    schema_field_count = len(st.session_state.json_schema.get("properties", {}))
    
    if schema_field_count > 15 and len(combined_text) > 5000:
        # For large text and many fields, use chunking approach
        status_text.text("Processing large document with chunking approach...")
        progress_bar.progress(40)
        
        extracted_fields = llm_extractor.extract_fields_with_chunking(
            combined_text,
            st.session_state.json_schema
        )
    elif schema_field_count > 15:
        # For many fields but smaller text, use tiered approach
        status_text.text("Using tiered extraction for complex schema...")
        progress_bar.progress(40)
        
        extracted_fields = llm_extractor.extract_with_tiered_approach(
            combined_text,
            st.session_state.json_schema
        )
    else:
        # For simpler cases, use standard extraction
        status_text.text("Extracting fields from document...")
        progress_bar.progress(40)
        
        extracted_fields = llm_extractor.extract_fields_from_text(
            combined_text,
            st.session_state.json_schema
        )
    
    st.session_state.extracted_structure = extracted_fields
    
    # Quick validation
    status_text.text("Validating extraction results...")
    progress_bar.progress(80)
    
    validation_results = llm_extractor.validate_extraction_quick(
        extracted_fields,
        st.session_state.json_schema
    )
    
    st.session_state.validation_results = validation_results
    
    # Complete
    progress_bar.progress(100)
    status_text.text("Field extraction complete!")
    
    # Calculate processing time
    st.session_state.processing_time["field_extraction"] = time.time() - start_time

def map_entities():
    """Map extracted entities to database entities"""
    start_time = time.time()
    
    # Initialize entity mapper
    entity_mapper = EntityMapper(use_embeddings=st.session_state.use_embeddings)
    
    # Map entities
    mapped_entities = entity_mapper.map_entities(
        st.session_state.extracted_structure,
        st.session_state.extracted_json,
        threshold=st.session_state.similarity_threshold
    )
    
    st.session_state.mapped_entities = mapped_entities
    
    true_mappings = {}
    predicted_mappings = mapped_entities["mappings"]
    
    # Create ground truth based on all extracted fields
    for extracted_key in st.session_state.extracted_structure.keys():
        # If same key exists in JSON, it should map to it
        if extracted_key in st.session_state.extracted_json:
            true_mappings[extracted_key] = extracted_key
    
    metrics = entity_mapper.evaluate_mapping_performance(true_mappings, predicted_mappings)
    st.session_state.metrics = metrics
    
    # Calculate processing time
    st.session_state.processing_time["entity_mapping"] = time.time() - start_time
    st.session_state.final_mapped_output = {
        key: st.session_state.extracted_structure.get(key)
        for key in st.session_state.mapped_entities["mappings"]
    }

def display_mapping_results():
    """Display entity mapping results with merged JSON at the bottom"""
    st.markdown("<h3 class='subsection-header'>Mapping Results</h3>", unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Mapped Entities", "Confidence Scores", "Unmapped Entities"])
    
    with tab1:
        st.markdown("#### Successfully Mapped Entities")
        
        # Create a DataFrame for mapped entities
        if st.session_state.mapped_entities["mappings"]:
            mapping_data = []
            
            for extracted_key, db_key in st.session_state.mapped_entities["mappings"].items():
                extracted_value = st.session_state.extracted_structure.get(extracted_key, "N/A")
                db_value = st.session_state.extracted_json.get(db_key, "N/A")
                
                # Convert complex objects to strings for display
                if not isinstance(extracted_value, (str, int, float, bool, type(None))):
                    extracted_value = json.dumps(extracted_value)
                if not isinstance(db_value, (str, int, float, bool, type(None))):
                    db_value = json.dumps(db_value)
                
                mapping_data.append({
                    "Extracted Field": extracted_key,
                    "Extracted Value": extracted_value,
                    "Database Field": db_key,
                    "Database Value": db_value,
                    "Confidence": st.session_state.mapped_entities["confidence_scores"].get(extracted_key, 0) * 100
                })
            
            if mapping_data:
                mapping_df = pd.DataFrame(mapping_data)
                st.dataframe(mapping_df)
            else:
                st.info("No entities were successfully mapped.")
        else:
            st.info("No entities were successfully mapped.")
    
    with tab2:
        st.markdown("#### Confidence Scores")
        
        # Create visualization for confidence scores
        if st.session_state.mapped_entities["confidence_scores"]:
            confidence_data = []
            
            for key, score in st.session_state.mapped_entities["confidence_scores"].items():
                confidence_data.append({
                    "Field": key,
                    "Confidence": score * 100
                })
            
            confidence_df = pd.DataFrame(confidence_data)
            
            # Create bar chart
            fig = px.bar(
                confidence_df,
                x="Field",
                y="Confidence",
                title="Mapping Confidence Scores",
                labels={"Confidence": "Confidence (%)"},
                color="Confidence",
                color_continuous_scale="RdYlGn",
                height=400
            )
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(confidence_df) - 0.5,
                y0=st.session_state.similarity_threshold * 100,
                y1=st.session_state.similarity_threshold * 100,
                line=dict(color="red", width=2, dash="dash")
            )
            
            # Add annotation for threshold
            fig.add_annotation(
                x=len(confidence_df) / 2,
                y=st.session_state.similarity_threshold * 100 + 5,
                text=f"Threshold: {st.session_state.similarity_threshold * 100:.0f}%",
                showarrow=False,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence scores available.")
    
    with tab3:
        st.markdown("#### Unmapped Entities")
        
        # Display unmapped entities
        if st.session_state.mapped_entities["unmapped_entities"]:
            st.markdown("The following entities could not be mapped to the database schema:")
            
            unmapped_data = []
            
            for field in st.session_state.mapped_entities["unmapped_entities"]:
                value = st.session_state.extracted_structure.get(field, "N/A")
                
                # Convert complex objects to strings for display
                if not isinstance(value, (str, int, float, bool, type(None))):
                    value = json.dumps(value)
                
                unmapped_data.append({
                    "Field": field,
                    "Extracted Value": value
                })
            
            if unmapped_data:
                unmapped_df = pd.DataFrame(unmapped_data)
                st.dataframe(unmapped_df)
            
            # Add an interactive component to suggest possible matches
            if st.button("Suggest Possible Matches for Unmapped Entities"):
                with st.spinner("Analyzing possible matches..."):
                    # Initialize entity mapper
                    entity_mapper = EntityMapper(use_embeddings=st.session_state.use_embeddings)
                    
                    suggestions = {}
                    for field in st.session_state.mapped_entities["unmapped_entities"]:
                        value = st.session_state.extracted_structure.get(field, "")
                        if value and isinstance(value, str):
                            # Get database entity names
                            db_entities = list(st.session_state.extracted_json.keys())
                            
                            # Get suggestions
                            field_suggestions = entity_mapper.suggest_entity_corrections(
                                field, db_entities
                            )
                            
                            if field_suggestions:
                                suggestions[field] = field_suggestions
                    
                    if suggestions:
                        st.markdown("#### Suggested Matches")
                        
                        for field, field_suggestions in suggestions.items():
                            st.markdown(f"**{field}**")
                            
                            sugg_data = []
                            for sugg in field_suggestions:
                                sugg_data.append({
                                    "Suggested Field": sugg["entity"],
                                    "Database Value": st.session_state.extracted_json.get(sugg["entity"], "N/A"),
                                    "Confidence": sugg["confidence"] * 100
                                })
                            
                            if sugg_data:
                                sugg_df = pd.DataFrame(sugg_data)
                                st.dataframe(sugg_df)
                    else:
                        st.info("No suggestions found for unmapped entities.")
        else:
            st.success("All entities were successfully mapped!")
    
    # Display the merged JSON at the bottom after mapping
    st.markdown("<h3 class='subsection-header'>Merged JSON Result</h3>", unsafe_allow_html=True)
    
    # Create a toggle for showing/hiding the JSON
    if st.checkbox("Show Merged JSON"):
        # Create the merged JSON
        if (st.session_state.get("extracted_structure") and 
            st.session_state.get("extracted_json") and 
            st.session_state.get("mapped_entities")):
            
            # Create a deep copy of the original JSON to avoid modifying it
            merged_json = dict(st.session_state.extracted_json)
            
            # For each mapped entity, update the corresponding field in the original JSON
            for extracted_key, db_key in st.session_state.mapped_entities["mappings"].items():
                extracted_value = st.session_state.extracted_structure.get(extracted_key)
                
                # Only merge non-null values
                if extracted_value is not None:
                    # Check if the database already has a value
                    db_value = st.session_state.extracted_json.get(db_key)
                    
                    # Only replace if the original value is null or the types match
                    if db_value is None or isinstance(db_value, type(extracted_value)):
                        merged_json[db_key] = extracted_value
            
            # Display the merged JSON
            st.json(merged_json)
        else:
            st.info("JSON data not available for merging.")
        
        # Also add a button to download the merged JSON
        if (st.session_state.get("extracted_structure") and 
            st.session_state.get("extracted_json") and 
            st.session_state.get("mapped_entities")):
            
            merged_json_str = json.dumps(merged_json, indent=2)
            st.download_button(
                label="📥 Download Merged JSON",
                data=merged_json_str,
                file_name="merged_output.json",
                mime="application/json"
            )

# Also need to update the update_final_mapped_output function to create the merged JSON
def update_final_mapped_output():
    """Update the final mapped output with merged JSON"""
    if (not st.session_state.get("extracted_structure") or 
        not st.session_state.get("extracted_json") or 
        not st.session_state.get("mapped_entities")):
        return
        
    # Create a deep copy of the original JSON
    merged_json = dict(st.session_state.extracted_json)
    
    # For each mapped entity, update the corresponding field in the original JSON
    for extracted_key, db_key in st.session_state.mapped_entities["mappings"].items():
        extracted_value = st.session_state.extracted_structure.get(extracted_key)
        
        # Only merge non-null values
        if extracted_value is not None:
            # Check if the database already has a value
            db_value = st.session_state.extracted_json.get(db_key)
            
            # Only replace if the original value is null or the types match
            if db_value is None or isinstance(db_value, type(extracted_value)):
                merged_json[db_key] = extracted_value
    
    st.session_state.final_mapped_output = merged_json

def evaluation_section():
    """Function for the evaluation and results section"""
    st.markdown("<h2 class='section-header'>📊 Evaluation & Results</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>View performance metrics and validation results.</div>", unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.mapped_entities is None:
        st.warning("Please extract and map entities first.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Validation Results", "Processing Time"])
    
    with tab1:
        st.markdown("#### Entity Mapping Performance")
        
        if st.session_state.metrics is not None and "error" not in st.session_state.metrics:
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{st.session_state.metrics['precision']:.2%}")
            
            with col2:
                st.metric("Recall", f"{st.session_state.metrics['recall']:.2%}")
            
            with col3:
                st.metric("F1 Score", f"{st.session_state.metrics['f1_score']:.2%}")
            
            with col4:
                st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.2%}")
            
            # Create visualization
            metrics_data = [
                {"Metric": "Precision", "Value": st.session_state.metrics["precision"] * 100},
                {"Metric": "Recall", "Value": st.session_state.metrics["recall"] * 100},
                {"Metric": "F1 Score", "Value": st.session_state.metrics["f1_score"] * 100},
                {"Metric": "Accuracy", "Value": st.session_state.metrics["accuracy"] * 100}
            ]
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = px.bar(
                metrics_df,
                x="Metric",
                y="Value",
                title="Entity Mapping Performance Metrics",
                labels={"Value": "Score (%)"},
                color="Value",
                color_continuous_scale="Blues",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display counts
            st.markdown("#### Entity Mapping Counts")
            
            counts_data = [
                {"Category": "Correct Mappings", "Count": st.session_state.metrics["correct"]},
                {"Category": "Incorrect Mappings", "Count": st.session_state.metrics["incorrect"]},
                {"Category": "Missed Mappings", "Count": st.session_state.metrics["missed"]}
            ]
            
            counts_df = pd.DataFrame(counts_data)
            
            fig = px.pie(
                counts_df,
                names="Category",
                values="Count",
                title="Entity Mapping Results",
                color="Category",
                color_discrete_map={
                    "Correct Mappings": "#4CAF50",
                    "Incorrect Mappings": "#F44336",
                    "Missed Mappings": "#FFC107"
                },
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics available yet.")
    
    with tab2:
        st.markdown("#### Field Extraction Validation")
        
        if st.session_state.validation_results is not None:
            # Display validation results
            st.json(st.session_state.validation_results)
            
            # Try to extract confidence scores from validation results
            confidence_data = []
            
            try:
                for field, data in st.session_state.validation_results.items():
                    if isinstance(data, dict) and "confidence" in data:
                        confidence_data.append({
                            "Field": field,
                            "Confidence": data["confidence"]
                        })
            except:
                pass
            
            if confidence_data:
                confidence_df = pd.DataFrame(confidence_data)
                
                fig = px.bar(
                    confidence_df,
                    x="Field",
                    y="Confidence",
                    title="Field Extraction Confidence",
                    labels={"Confidence": "Confidence (%)"},
                    color="Confidence",
                    color_continuous_scale="Viridis",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No validation results available yet.")
    
    with tab3:
        st.markdown("#### Processing Time Breakdown")
        
        if st.session_state.processing_time:
            # Create processing time display
            time_data = []
            
            for stage, duration in st.session_state.processing_time.items():
                time_data.append({
                    "Stage": stage.replace("_", " ").title(),
                    "Time (seconds)": duration
                })
            
            time_df = pd.DataFrame(time_data)
            
            fig = px.bar(
                time_df,
                x="Stage",
                y="Time (seconds)",
                title="Processing Time by Stage",
                color="Time (seconds)",
                color_continuous_scale="Thermal",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display total processing time
            total_time = sum(st.session_state.processing_time.values())
            st.metric("Total Processing Time", f"{total_time:.2f} seconds")
        else:
            st.info("No processing time data available yet.")

def manual_editing_section():
    """Function for the manual editing section"""
    st.markdown("<h2 class='section-header'>⚙️ Manual Editing</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>Manually edit extracted fields and mappings for improved accuracy.</div>", unsafe_allow_html=True)
    
    # Check if data is available
    if st.session_state.extracted_structure is None or st.session_state.mapped_entities is None:
        st.warning("Please extract and map entities first.")
        return
    
    # Create tabs for different editing options
    tab1, tab2 = st.tabs(["Edit Extracted Fields", "Edit Mappings"])
    
    with tab1:
        st.markdown("#### Edit Extracted Fields")
        
        # Display and allow editing of extracted fields
        if st.session_state.extracted_structure:
            st.markdown("Review and edit the extracted fields below:")
            
            # Initialize manual edits if needed
            if "field_edits" not in st.session_state.manual_edits:
                st.session_state.manual_edits["field_edits"] = {}
                for field, value in st.session_state.extracted_structure.items():

                    st.session_state.manual_edits["field_edits"][field] = value
            
            # Create a form for editing
            edited_fields = {}
            
            for field, value in st.session_state.manual_edits["field_edits"].items():
                if isinstance(value, str):
                    edited_value = st.text_input(f"**{field}**", value, key=f"edit_{field}")
                elif isinstance(value, (int, float)):
                    edited_value = st.number_input(f"**{field}**", value=value, key=f"edit_{field}")
                elif isinstance(value, bool):
                    edited_value = st.checkbox(f"**{field}**", value, key=f"edit_{field}")
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    st.markdown(f"**{field}** (List of Strings)")
                    edited_list = []
                    for i, item in enumerate(value):
                        edited_item = st.text_input(f"{field}[{i}]", item, key=f"{field}_{i}")
                        edited_list.append(edited_item)
                    edited_value = edited_list
                else:
                    # Fallback to editable JSON
                    json_str = json.dumps(value, indent=2)
                    edited_str = st.text_area(f"**{field}** (JSON)", json_str, height=150, key=f"edit_{field}")
                    try:
                        edited_value = json.loads(edited_str)
                    except json.JSONDecodeError:
                        edited_value = value

                edited_fields[field] = edited_value
            
            if st.button("Apply Field Edits"):
                # Update the extracted structure with edited values
                for field, value in edited_fields.items():
                    st.session_state.extracted_structure[field] = value
                
                # Update session state
                st.session_state.manual_edits["field_edits"] = edited_fields
                
                st.success("Field edits applied successfully!")
                update_final_mapped_output()

                # Option to re-run mapping with edited fields
                if st.button("Re-run Entity Mapping with Edited Fields"):
                    with st.spinner("Mapping entities with edited fields..."):
                        map_entities()
                    
                    st.success("Entity mapping updated!")
        else:
            st.info("No extracted fields to edit.")
    
    with tab2:
        st.markdown("#### Edit Entity Mappings")
        
        # Display and allow editing of mappings
        if st.session_state.mapped_entities and st.session_state.mapped_entities["mappings"]:
            st.markdown("Review and edit the entity mappings below:")
            
            # Initialize mapping edits if needed
            if "mapping_edits" not in st.session_state.manual_edits:
                st.session_state.manual_edits["mapping_edits"] = {}
                for extracted_key, db_key in st.session_state.mapped_entities["mappings"].items():
                    st.session_state.manual_edits["mapping_edits"][extracted_key] = db_key
            
            # Get all possible database fields
            db_fields = list(st.session_state.extracted_json.keys())
            
            # Create a form for editing mappings
            edited_mappings = {}
            
            for extracted_key, db_key in st.session_state.manual_edits["mapping_edits"].items():
                # Create a selectbox for each mapping
                selected_db_key = st.selectbox(
                    f"Map **{extracted_key}** to:",
                    options=[""] + db_fields,
                    index=db_fields.index(db_key) + 1 if db_key in db_fields else 0,
                    key=f"map_{extracted_key}"
                )
                
                if selected_db_key:
                    edited_mappings[extracted_key] = selected_db_key
            
            # Option to add new mappings for unmapped entities
            if st.session_state.mapped_entities["unmapped_entities"]:
                st.markdown("#### Map Unmapped Entities")
                
                for field in st.session_state.mapped_entities["unmapped_entities"]:
                    # Create a selectbox for each unmapped entity
                    selected_db_key = st.selectbox(
                        f"Map **{field}** to:",
                        options=[""] + db_fields,
                        index=0,
                        key=f"unmap_{field}"
                    )
                    
                    if selected_db_key:
                        edited_mappings[field] = selected_db_key
            
            if st.button("Apply Mapping Edits"):
                # Update the mappings with edited values
                st.session_state.mapped_entities["mappings"].update(edited_mappings)
                
                # Update unmapped entities list
                all_fields = list(st.session_state.extracted_structure.keys())
                mapped_fields = list(st.session_state.mapped_entities["mappings"].keys())
                unmapped_fields = [field for field in all_fields if field not in mapped_fields]
                
                st.session_state.mapped_entities["unmapped_entities"] = unmapped_fields
                
                # Update session state
                st.session_state.manual_edits["mapping_edits"] = edited_mappings
                
                st.success("Mapping edits applied successfully!")
                update_final_mapped_output()
                
                # Update metrics
                entity_mapper = EntityMapper(use_embeddings=st.session_state.use_embeddings)
                
                true_mappings = {}
                for key in st.session_state.mapped_entities["mappings"]:
                    # Assume the key from extracted structure should map to same key in JSON
                    true_mappings[key] = key
                
                metrics = entity_mapper.evaluate_mapping_performance(
                    true_mappings, 
                    st.session_state.mapped_entities["mappings"]
                )
                
                st.session_state.metrics = metrics
        else:
            st.info("No mappings to edit.")

def update_final_mapped_output():
    st.session_state.final_mapped_output = {
        key: st.session_state.extracted_structure.get(key)
        for key in st.session_state.mapped_entities["mappings"]
    }

def export_results_section():
    st.markdown("## 📁 Export Results")
    st.markdown("Download the final mapped JSON with all extracted and aligned fields.")

    if (
        st.session_state.get("mapped_entities") 
        and st.session_state.get("extracted_structure")
        and st.session_state.mapped_entities.get("mappings")
    ):
        st.session_state.final_mapped_output = {
            key: st.session_state.extracted_structure.get(key)
            for key in st.session_state.mapped_entities["mappings"]
        }

    if st.session_state.get("final_mapped_output"):
        mapped_json = json.dumps(st.session_state['final_mapped_output'], indent=2)
        st.download_button(
            label="📥 Download Mapped JSON",
            data=mapped_json,
            file_name="mapped_output.json",
            mime="application/json"
        )
    else:
        st.info("No mapped data available to export yet.")

if __name__ == "__main__":
    main()