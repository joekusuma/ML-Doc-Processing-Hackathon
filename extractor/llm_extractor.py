import json
import os
from typing import Dict, List, Any, Optional, Union
import openai
from anthropic import Anthropic
from dotenv import load_dotenv
import time
import hashlib

# Load environment variables
load_dotenv()

class LLMExtractor:
    """
    Class for extracting structured information from text using Large Language Models
    with optimizations for performance and efficiency
    """
    
    def __init__(self, model_provider="openai", model_name=None, cache_dir="./cache"):
        """
        Initialize the LLM extractor
        
        Args:
            model_provider (str): Provider of the LLM ("openai" or "anthropic")
            model_name (str): Name of the model to use (optional)
            cache_dir (str): Directory for caching extraction results
        """
        self.model_provider = model_provider.lower()
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize API keys and clients
        if self.model_provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model_name = model_name or "gpt-4o"
        elif self.model_provider == "anthropic":
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = model_name or "claude-3-opus-20240229"
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
    
    def extract_fields_from_text(self, text: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured information from text based on a JSON schema with optimization
        
        Args:
            text (str): Text to extract information from
            json_schema (Dict): JSON schema defining the fields to extract
            
        Returns:
            Dict: Extracted fields
        """
        # Check if result is in cache
        cache_key = self._generate_cache_key(text[:1000], json_schema)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create a more focused prompt
        fields = list(json_schema.get('properties', {}).keys())
        fields_str = ", ".join(fields)
        
        # Create descriptions for required fields
        field_descriptions = []
        for field in json_schema.get('required', []):
            if field in json_schema.get('properties', {}):
                desc = json_schema.get('properties', {}).get(field, {}).get('description', f'The {field} field')
                field_descriptions.append(f"- {field}: {desc}")
        
        field_desc_str = "\n".join(field_descriptions)
        
        # Create a more concise prompt
        prompt = f"""
Extract the following fields from the provided text:
{field_desc_str}

The fields to extract are: {fields_str}

Return the extracted data in JSON format. If a field cannot be found, set it to null.

Text to extract from:
{text[:10000]}  # Limiting text length to avoid token limit issues

Return ONLY the JSON output.
        """
        
        try:
            if self.model_provider == "openai":
                result = self._extract_with_openai(prompt, json_schema)
            elif self.model_provider == "anthropic":
                result = self._extract_with_anthropic(prompt, json_schema)
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            return result
        except Exception as e:
            raise Exception(f"Error extracting fields with LLM: {str(e)}")
    
    def _extract_with_openai(self, prompt: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information using OpenAI models with optimization
        
        Args:
            prompt (str): Prompt for the model
            json_schema (Dict): JSON schema defining the fields to extract
            
        Returns:
            Dict: Extracted fields
        """
        # Create a more concise system message
        system_message = "Extract the requested fields from the provided text and return only the JSON output."
        
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Lower temperature for more consistent results
        )
        
        extracted_json = json.loads(response.choices[0].message.content)
        
        # Validate against schema (basic validation)
        for field in json_schema.get('required', []):
            if field not in extracted_json:
                extracted_json[field] = None
        
        return extracted_json
    
    def _extract_with_anthropic(self, prompt: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information using Anthropic models with optimization
        
        Args:
            prompt (str): Prompt for the model
            json_schema (Dict): JSON schema defining the fields to extract
            
        Returns:
            Dict: Extracted fields
        """
        response = self.anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        # Extract JSON from response
        content = response.content[0].text
        
        # Try to parse JSON from the response
        try:
            # Find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                extracted_json = json.loads(json_str)
            else:
                # Fallback if no JSON found
                extracted_json = {}
        except json.JSONDecodeError:
            # Handle parsing errors
            extracted_json = {}
        
        # Validate against schema (basic validation)
        for field in json_schema.get('required', []):
            if field not in extracted_json:
                extracted_json[field] = None
        
        return extracted_json
    
    def extract_fields_with_chunking(self, text: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from large text by processing in batches of fields
        
        Args:
            text (str): Text to extract information from
            json_schema (Dict): JSON schema defining the fields to extract
            
        Returns:
            Dict: Extracted fields
        """
        # Check if result is in cache
        cache_key = self._generate_cache_key(text[:1000], json_schema)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        # Split required fields into groups to process separately
        required_fields = json_schema.get("required", [])
        
        # If there are more than 10 fields, process in batches
        if len(required_fields) > 10:
            field_batches = [required_fields[i:i+10] for i in range(0, len(required_fields), 10)]
        else:
            field_batches = [required_fields]
        
        # Process each batch with the full text
        results = {}
        
        for batch in field_batches:
            # Create a simplified schema with only the fields in this batch
            batch_schema = {
                "type": "object",
                "properties": {field: json_schema["properties"][field] for field in batch if field in json_schema["properties"]},
                "required": batch
            }
            
            # Extract fields for this batch
            batch_results = self.extract_fields_from_text(text, batch_schema)
            
            # Merge results
            results.update(batch_results)
        
        # Cache the result
        self._cache_result(cache_key, results)
        
        return results
    
    def extract_with_tiered_approach(self, text: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use a tiered approach with faster models for simple extraction
        and more powerful models for complex fields
        
        Args:
            text (str): Text to extract information from
            json_schema (Dict): JSON schema defining the fields to extract
            
        Returns:
            Dict: Extracted fields
        """
        # Check if result is in cache
        cache_key = self._generate_cache_key(text[:1000], json_schema)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        # Categorize fields by complexity (simple vs complex)
        # Simple fields: strings, numbers, booleans
        # Complex fields: objects, arrays, or fields with specific validation requirements
        simple_fields = []
        complex_fields = []
        
        for field, schema in json_schema.get('properties', {}).items():
            field_type = schema.get('type', '')
            
            if field_type in ['string', 'number', 'integer', 'boolean']:
                simple_fields.append(field)
            else:
                complex_fields.append(field)
        
        # Process simple fields with faster model
        original_model = self.model_name
        
        # Switch to faster model for simple fields
        if self.model_provider == "openai":
            self.model_name = "gpt-3.5-turbo"
        elif self.model_provider == "anthropic":
            self.model_name = "claude-3-haiku-20240307"
        
        # Create schema for simple fields
        simple_schema = {
            "type": "object",
            "properties": {field: json_schema["properties"][field] for field in simple_fields if field in json_schema["properties"]},
            "required": [field for field in simple_fields if field in json_schema.get("required", [])]
        }
        
        # Extract simple fields
        if simple_fields:
            simple_results = self.extract_fields_from_text(text, simple_schema)
        else:
            simple_results = {}
        
        # Switch back to powerful model for complex fields
        self.model_name = original_model
        
        # Create schema for complex fields
        complex_schema = {
            "type": "object",
            "properties": {field: json_schema["properties"][field] for field in complex_fields if field in json_schema["properties"]},
            "required": [field for field in complex_fields if field in json_schema.get("required", [])]
        }
        
        # Extract complex fields
        if complex_fields:
            complex_results = self.extract_fields_from_text(text, complex_schema)
        else:
            complex_results = {}
        
        # Combine results
        combined_results = {**simple_results, **complex_results}
        
        # Cache the result
        self._cache_result(cache_key, combined_results)
        
        return combined_results
    
    def validate_extraction_quick(self, extracted_data: Dict[str, Any], json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a quick validation of extracted data against schema
        
        Args:
            extracted_data (Dict): Extracted field data
            json_schema (Dict): JSON schema to validate against
            
        Returns:
            Dict: Validation results
        """
        validation_results = {}
        
        # Check required fields
        for field in json_schema.get("required", []):
            if field in json_schema.get("properties", {}):
                field_present = field in extracted_data and extracted_data[field] is not None
                field_type = json_schema["properties"][field].get("type", "string")
                
                # Check type match
                type_match = True
                if field_present and extracted_data[field] is not None:
                    if field_type == "string" and not isinstance(extracted_data[field], str):
                        type_match = False
                    elif field_type == "number" and not isinstance(extracted_data[field], (int, float)):
                        type_match = False
                    elif field_type == "boolean" and not isinstance(extracted_data[field], bool):
                        type_match = False
                    elif field_type == "array" and not isinstance(extracted_data[field], list):
                        type_match = False
                    elif field_type == "object" and not isinstance(extracted_data[field], dict):
                        type_match = False
                
                validation_results[field] = {
                    "present": field_present,
                    "type_match": type_match,
                    "confidence": 90 if (field_present and type_match) else 50 if field_present else 0
                }
        
        # Check optional fields
        for field in json_schema.get("properties", {}):
            if field not in json_schema.get("required", []) and field in extracted_data:
                field_present = extracted_data[field] is not None
                field_type = json_schema["properties"][field].get("type", "string")
                
                # Check type match
                type_match = True
                if field_present and extracted_data[field] is not None:
                    if field_type == "string" and not isinstance(extracted_data[field], str):
                        type_match = False
                    elif field_type == "number" and not isinstance(extracted_data[field], (int, float)):
                        type_match = False
                    elif field_type == "boolean" and not isinstance(extracted_data[field], bool):
                        type_match = False
                    elif field_type == "array" and not isinstance(extracted_data[field], list):
                        type_match = False
                    elif field_type == "object" and not isinstance(extracted_data[field], dict):
                        type_match = False
                
                validation_results[field] = {
                    "present": field_present,
                    "type_match": type_match,
                    "confidence": 70 if (field_present and type_match) else 40 if field_present else 0
                }
        
        return validation_results
    
    def _generate_cache_key(self, text_sample: str, json_schema: Dict[str, Any]) -> str:
        """
        Generate a cache key based on text sample and schema
        
        Args:
            text_sample (str): Sample of text to include in the key
            json_schema (Dict): JSON schema
            
        Returns:
            str: Cache key
        """
        # Create a hash from the text sample and schema
        schema_str = json.dumps(json_schema.get("required", []), sort_keys=True)
        combined = f"{text_sample}_{schema_str}_{self.model_provider}_{self.model_name}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if results are available in cache
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            Optional[Dict]: Cached result or None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
                
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Save result to cache
        
        Args:
            cache_key (str): Cache key
            result (Dict): Result to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except:
            # If caching fails, just continue without caching
            pass