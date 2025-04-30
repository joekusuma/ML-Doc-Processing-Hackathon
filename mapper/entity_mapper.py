import json
import difflib
import re
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from dateutil import parser
import datetime

# Try to import sentence-transformers for better similarity matching
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

class EntityMapper:
    """
    Class for mapping extracted entities to database entities, 
    handling variations in entity names
    """
    
    def __init__(self, use_embeddings=True):
        """
        Initialize the entity mapper
        
        Args:
            use_embeddings (bool): Whether to use sentence embeddings for matching
        """
        self.use_embeddings = use_embeddings and HAVE_SENTENCE_TRANSFORMERS
        
        # Initialize sentence transformer model if available
        if self.use_embeddings:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.use_embeddings = False
                print("Warning: Failed to load sentence transformer model. Falling back to TF-IDF.")
        
        # TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=1
        )
        
        # Date format patterns for detection
        self.date_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',  # ISO format: 2023-03-02
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # US format: 3/2/23 or 3/2/2023
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # Dashed format: 3-2-23 or 3-2-2023
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}',  # 2 March 2023
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}',  # March 2, 2023
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}'  # Mar 2, 2023
        ]
    
    def map_entities(self, 
                     extracted_entities: Dict[str, Any], 
                     database_entities: Dict[str, Any], 
                     threshold: float = 0.7) -> Dict[str, Any]:
        """
        Map extracted entities to database entities
        
        Args:
            extracted_entities (Dict): Entities extracted from documents
            database_entities (Dict): Entities from the database
            threshold (float): Similarity threshold for matching
            
        Returns:
            Dict: Mapping between extracted and database entities
        """
        mappings = {}
        confidence_scores = {}
        unmapped_entities = []
        
        # Process each extracted entity
        for extracted_key, extracted_value in extracted_entities.items():
            if extracted_value is None:
                continue
                
            # Skip nested objects - map them separately if needed
            if isinstance(extracted_value, dict):
                continue
                
            # Find best match in database keys
            best_match, name_score, value_score, combined_score = self._find_best_match(
                extracted_key, extracted_value, database_entities, threshold
            )
            
            if best_match:
                mappings[extracted_key] = best_match
                confidence_scores[extracted_key] = combined_score
            else:
                unmapped_entities.append(extracted_key)
        
        return {
            "mappings": mappings,
            "confidence_scores": confidence_scores,
            "unmapped_entities": unmapped_entities
        }
    
    def _find_best_match(self, 
                         extracted_key: str,
                         extracted_value: Any, 
                         database_entities: Dict[str, Any], 
                         threshold: float) -> Tuple[Optional[str], float, float, float]:
        """
        Find the best match for an entity in the database, considering both key and value
        
        Args:
            extracted_key (str): Key of the extracted entity
            extracted_value (Any): Value of the extracted entity
            database_entities (Dict): Database entities
            threshold (float): Similarity threshold
            
        Returns:
            Tuple: (best_match, name_score, value_score, combined_score)
        """
        best_match = None
        best_name_score = 0.0
        best_value_score = 0.0
        best_combined_score = 0.0
        
        # Convert extracted value to string if needed
        extracted_value_str = str(extracted_value) if not isinstance(extracted_value, str) else extracted_value
        
        # Check if the extracted value looks like a date
        extracted_date = None
        if self._looks_like_date(extracted_value_str):
            try:
                extracted_date = self._normalize_date(extracted_value_str)
            except:
                extracted_date = None
        
        # Process each database entity
        for db_key, db_value in database_entities.items():
            # Skip nested objects
            if isinstance(db_value, dict):
                continue
                
            # Convert to string for comparison
            db_value_str = str(db_value) if not isinstance(db_value, str) else db_value
            
            # Calculate name similarity
            name_similarity = self._compute_key_similarity(extracted_key, db_key)
            
            # Calculate value similarity
            value_similarity = 0.0
            
            # Special handling for dates
            if extracted_date is not None and self._looks_like_date(db_value_str):
                try:
                    db_date = self._normalize_date(db_value_str)
                    # If dates are exactly the same, perfect match
                    if extracted_date == db_date:
                        value_similarity = 1.0
                    else:
                        # Calculate closeness of dates
                        date_diff = abs((extracted_date - db_date).days)
                        if date_diff <= 30:  # Within one month
                            value_similarity = max(0, 1 - (date_diff / 30))
                        else:
                            value_similarity = 0.0
                except:
                    # If date parsing fails, fall back to string similarity
                    value_similarity = self._compute_value_similarity(extracted_value_str, db_value_str)
            else:
                # Regular string similarity for non-date values
                value_similarity = self._compute_value_similarity(extracted_value_str, db_value_str)
            
            # Combined score - weighted average of name and value similarity
            # Give more weight to name similarity (70%) than value similarity (30%)
            combined_similarity = (0.7 * name_similarity) + (0.3 * value_similarity)
            
            # Update best match if this one is better
            if combined_similarity > best_combined_score and combined_similarity >= threshold:
                best_match = db_key
                best_name_score = name_similarity
                best_value_score = value_similarity
                best_combined_score = combined_similarity
        
        return best_match, best_name_score, best_value_score, best_combined_score
    
    def _compute_key_similarity(self, key1: str, key2: str) -> float:
        """
        Compute similarity between entity keys (field names)
        
        Args:
            key1 (str): First key
            key2 (str): Second key
            
        Returns:
            float: Similarity score (0-1)
        """
        # Normalize keys
        key1 = self._normalize_key(key1)
        key2 = self._normalize_key(key2)
        
        # If keys are identical after normalization, return perfect match
        if key1 == key2:
            return 1.0
        
        # Use embeddings for semantic similarity if available
        if self.use_embeddings:
            return self._compute_embedding_similarity(key1, key2)
        else:
            return self._compute_string_similarity(key1, key2)
    
    def _compute_value_similarity(self, value1: str, value2: str) -> float:
        """
        Compute similarity between entity values
        
        Args:
            value1 (str): First value
            value2 (str): Second value
            
        Returns:
            float: Similarity score (0-1)
        """
        # If values are identical, return perfect match
        if value1 == value2:
            return 1.0
        
        # Use embeddings for semantic similarity if available
        if self.use_embeddings:
            return self._compute_embedding_similarity(value1, value2)
        else:
            return self._compute_string_similarity(value1, value2)
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize field key for better comparison
        
        Args:
            key (str): Field key to normalize
            
        Returns:
            str: Normalized field key
        """
        # Convert to lowercase
        normalized = key.lower()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Replace underscores and hyphens with spaces
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _normalize_entity_name(self, entity: str) -> str:
        """
        Normalize entity name for better comparison
        
        Args:
            entity (str): Entity name to normalize
            
        Returns:
            str: Normalized entity name
        """
        # Convert to lowercase
        normalized = entity.lower()
        
        # Remove common legal suffixes
        suffixes = [' inc', ' corp', ' llc', ' ltd', ' limited', ' co', ' company', ' group']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _looks_like_date(self, text: str) -> bool:
        """
        Check if a string looks like a date
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text looks like a date
        """
        # Check against date patterns
        for pattern in self.date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Try to parse with dateutil as a fallback
        try:
            parser.parse(text, fuzzy=True)
            return True
        except:
            return False
    
    def _normalize_date(self, date_str: str) -> datetime.datetime:
        """
        Normalize date string to datetime object
        
        Args:
            date_str (str): Date string to normalize
            
        Returns:
            datetime.datetime: Normalized date
        """
        return parser.parse(date_str, fuzzy=True)
    
    def _compute_embedding_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute similarity between entities using embeddings
        
        Args:
            entity1 (str): First entity
            entity2 (str): Second entity
            
        Returns:
            float: Similarity score
        """
        # Handle empty strings
        if not entity1 or not entity2:
            return 0.0
            
        # Encode entities
        embedding1 = self.model.encode([entity1])[0]
        embedding2 = self.model.encode([entity2])[0]
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        return float(similarity)
    
    def _compute_string_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute similarity between entities using TF-IDF and string metrics
        
        Args:
            entity1 (str): First entity
            entity2 (str): Second entity
            
        Returns:
            float: Similarity score
        """
        # Handle empty strings
        if not entity1 or not entity2:
            return 0.0
            
        # Use a combination of different string similarity metrics
        
        # 1. Sequence matcher
        seq_similarity = difflib.SequenceMatcher(None, entity1, entity2).ratio()
        
        # 2. TF-IDF with character n-grams
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([entity1, entity2])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # 3. Word overlap
        words1 = set(entity1.split())
        words2 = set(entity2.split())
        if not words1 or not words2:
            overlap_similarity = 0.0
        else:
            overlap_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))
        
        # Combine similarities with weights
        combined_similarity = (0.4 * seq_similarity + 0.4 * tfidf_similarity + 0.2 * overlap_similarity)
        
        return float(combined_similarity)
    
    def generate_entity_aliases(self, entities: List[str]) -> Dict[str, List[str]]:
        """
        Generate potential aliases for entities
        
        Args:
            entities (List[str]): List of entity names
            
        Returns:
            Dict: Mapping of entities to their potential aliases
        """
        aliases = {}
        
        for entity in entities:
            entity_aliases = []
            
            # Normalize the entity name
            normalized = self._normalize_entity_name(entity)
            if normalized != entity.lower():
                entity_aliases.append(normalized)
            
            # Add abbreviation if entity has multiple words
            words = entity.split()
            if len(words) > 1:
                abbreviation = ''.join(word[0] for word in words if word)
                if len(abbreviation) > 1:
                    entity_aliases.append(abbreviation.upper())
            
            # Add variants without common suffixes
            if any(suffix in entity.lower() for suffix in [' inc', ' corp', ' llc', ' ltd']):
                base_name = re.sub(r'\s+(inc|corp|llc|ltd|limited|co|company|group)\.?$', '', entity.lower(), flags=re.IGNORECASE)
                if base_name and base_name != entity.lower():
                    entity_aliases.append(base_name)
            
            aliases[entity] = entity_aliases
        
        return aliases
    
    def evaluate_mapping_performance(self, 
                                true_mappings: Dict[str, str], 
                                predicted_mappings: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate entity mapping performance with improved accuracy
        
        Args:
            true_mappings (Dict): Ground truth mappings
            predicted_mappings (Dict): Predicted mappings
            
        Returns:
            Dict: Evaluation metrics
        """
        if not true_mappings:
            return {"error": "No ground truth mappings provided"}
        
        total_true = len(true_mappings)
        total_predicted = len(predicted_mappings)
        
        correct = 0
        incorrect = 0
        missed = 0
        
        # Count correct predictions and misses
        for entity, true_match in true_mappings.items():
            if entity in predicted_mappings:
                if predicted_mappings[entity] == true_match:
                    correct += 1
                else:
                    incorrect += 1
            else:
                missed += 1
        
        # Count any additional incorrect predictions 
        # (fields that were mapped but weren't in true_mappings)
        extra_incorrect = sum(1 for entity in predicted_mappings if entity not in true_mappings)
        incorrect += extra_incorrect
        
        # Calculate metrics using proper denominators
        precision = correct / total_predicted if total_predicted > 0 else 0
        recall = correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": correct / total_true if total_true > 0 else 0,
            "correct": correct,
            "incorrect": incorrect,
            "missed": missed,
            "total_true": total_true,
            "total_predicted": total_predicted
        }

    def _values_match(self, val1: Any, val2: Any) -> bool:
        """
        Check if two values match, with type conversion for numeric values
        
        Args:
            val1: First value
            val2: Second value
            
        Returns:
            bool: True if values match
        """
        # Handle None values
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False
        
        # Convert to strings for comparison
        str_val1 = str(val1).strip()
        str_val2 = str(val2).strip()
        
        # Try strict equality first
        if str_val1 == str_val2:
            return True
        
        # Try numeric comparison for numeric values
        try:
            # Handle common currency formatting
            num_val1 = float(str_val1.replace(',', '').replace('$', ''))
            num_val2 = float(str_val2.replace(',', '').replace('$', ''))
            
            # Allow small differences for floating point values
            return abs(num_val1 - num_val2) < 0.01
        except:
            # If conversion fails, they're not numeric or not in expected format
            pass
        
        # Try case-insensitive comparison for strings
        if str_val1.lower() == str_val2.lower():
            return True
        
        return False
    
    def handle_ambiguous_matches(self, 
                               entity: str, 
                               candidates: List[str], 
                               context: str = None) -> Dict[str, Any]:
        """
        Handle cases where multiple database entities match an extracted entity
        
        Args:
            entity (str): Extracted entity
            candidates (List[str]): List of candidate matches
            context (str, optional): Context text to help disambiguate
            
        Returns:
            Dict: Disambiguated results with confidence scores
        """
        # If only one candidate, return it directly
        if len(candidates) == 1:
            return {
                "best_match": candidates[0],
                "confidence": 1.0,
                "alternatives": []
            }
        
        # Initialize result structure
        results = {
            "best_match": None,
            "confidence": 0.0,
            "alternatives": []
        }
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            if self.use_embeddings:
                score = self._compute_embedding_similarity(entity, candidate)
            else:
                score = self._compute_string_similarity(entity, candidate)
                
            # If context is provided, use it to adjust the score
            if context:
                context_score = self._evaluate_with_context(candidate, context)
                # Weight the scores (0.7 for string similarity, 0.3 for context)
                score = 0.7 * score + 0.3 * context_score
                
            scored_candidates.append((candidate, score))
        
        # Sort candidates by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Set best match and alternatives
        if scored_candidates:
            best_candidate, best_score = scored_candidates[0]
            results["best_match"] = best_candidate
            results["confidence"] = best_score
            
            # Add alternatives with their scores
            results["alternatives"] = [
                {"entity": candidate, "score": score}
                for candidate, score in scored_candidates[1:]
            ]
        
        return results
    
    def _evaluate_with_context(self, entity: str, context: str) -> float:
        """
        Evaluate how well an entity fits in a given context
        
        Args:
            entity (str): Entity to evaluate
            context (str): Context text
            
        Returns:
            float: Context relevance score
        """
        # Simple implementation: check if entity appears in context
        if entity.lower() in context.lower():
            return 1.0
            
        # Check for word-by-word appearance
        words = entity.lower().split()
        words_in_context = sum(1 for word in words if word in context.lower())
        
        if words:
            return words_in_context / len(words)
        else:
            return 0.0
    
    def suggest_entity_corrections(self, entity: str, database_entities: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest corrections for an entity that couldn't be mapped
        
        Args:
            entity (str): Unmapped entity
            database_entities (List[str]): List of database entities
            
        Returns:
            List: Suggested corrections with confidence scores
        """
        suggestions = []
        
        for db_entity in database_entities:
            if self.use_embeddings:
                similarity = self._compute_embedding_similarity(entity, db_entity)
            else:
                similarity = self._compute_string_similarity(entity, db_entity)
                
            if similarity > 0.5:  # Threshold for suggestions
                suggestions.append({
                    "entity": db_entity,
                    "confidence": similarity
                })
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions