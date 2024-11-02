import torch
import spacy
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import nltk
from transformers import pipeline
import os
from .pipeline import ProcessingStatus, GPUOptimizer

class ContentProcessor:
    """Processes and analyzes video content using NLP models"""
    
    def __init__(self, models_dir: str = "models", device: Optional[str] = None):
        """
        Initialize content processor with CUDA support
        
        Args:
            models_dir: Directory for model files
            device: Optional device override
        """
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger('ContentProcessor')
        
        # Initialize GPU optimizer
        self.gpu_optimizer = GPUOptimizer(memory_fraction=0.3)
        self.device = device if device else ('cuda' if self.gpu_optimizer.setup() else 'cpu')
        
        # Download required NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            self.logger.error(f"Error downloading NLTK data: {str(e)}")
            
        # Initialize models
        self.setup_models()

    def setup_models(self):
        """Initialize NLP models with error handling and GPU support"""
        try:
            # Load spaCy model (CPU only)
            self.logger.info("Loading spaCy model...")
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.info("Downloading spaCy model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load('en_core_web_sm')

            # Load sentence transformer with GPU support
            self.logger.info("Loading sentence transformer model...")
            def load_sentence_transformer():
                return SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            self.sentence_model = load_sentence_transformer()

            # Initialize keyword extraction
            self.logger.info("Initializing keyword extraction...")
            try:
                def init_keybert():
                    return KeyBERT(model=self.sentence_model)
                
                self.key_phrase_model = init_keybert()
            except Exception as e:
                self.logger.warning(f"Error initializing KeyBERT with custom model: {str(e)}")
                self.key_phrase_model = KeyBERT()

            # Load sentiment analysis with GPU support
            self.logger.info("Loading sentiment analysis model...")
            def load_sentiment_analyzer():
                return pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if self.device == 'cuda' else -1
                )
            
            self.sentiment_analyzer = load_sentiment_analyzer()

            # Initialize content type patterns
            self.setup_content_patterns()
            
            # Clean up any GPU memory after model loading
            if self.device == 'cuda':
                self.gpu_optimizer.cleanup()
                
            self.logger.info("All models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise

    def setup_content_patterns(self):
        """Initialize content type patterns and weights"""
        self.content_patterns = {
            'technical': {
                'patterns': ['technique', 'method', 'algorithm', 'implementation', 'system'],
                'weight': 1.2
            },
            'educational': {
                'patterns': ['learn', 'understand', 'concept', 'explain', 'tutorial'],
                'weight': 1.0
            },
            'narrative': {
                'patterns': ['story', 'experience', 'journey', 'happened', 'when'],
                'weight': 0.8
            },
            'descriptive': {
                'patterns': ['looks', 'appears', 'seems', 'shows', 'demonstrates'],
                'weight': 0.9
            },
            'instructional': {
                'patterns': ['step', 'guide', 'how to', 'follow', 'instructions'],
                'weight': 1.1
            },
            'analytical': {
                'patterns': ['analysis', 'research', 'study', 'findings', 'results'],
                'weight': 1.3
            }
        }

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content with GPU optimization"""
        try:
            with torch.no_grad():
                content_types = self.identify_content_type(text)
                key_concepts = self.extract_key_concepts(text)
                
                result = {
                    'content_types': content_types,
                    'key_concepts': key_concepts,
                    'status': ProcessingStatus.COMPLETED.value
                }
                
                # Cleanup GPU memory if needed
                if self.device == 'cuda':
                    self.gpu_optimizer.cleanup()
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Error in content analysis: {str(e)}")
            return {
                'content_types': {'general': 100.0},
                'key_concepts': [],
                'status': ProcessingStatus.FAILED.value,
                'error': str(e)
            }

    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment with GPU optimization"""
        try:
            # Limit chunk size for transformer efficiency
            text = text[:512]
            
            with torch.no_grad():
                result = self.sentiment_analyzer(text)[0]
                
                sentiment_result = {
                    'label': result['label'],
                    'score': result['score'],
                    'status': ProcessingStatus.COMPLETED.value
                }
                
                # Cleanup GPU memory if needed
                if self.device == 'cuda':
                    self.gpu_optimizer.cleanup()
                    
                return sentiment_result
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'status': ProcessingStatus.FAILED.value,
                'error': str(e)
            }

    def identify_content_type(self, chunk: str) -> Dict[str, float]:
        """Identify content type with improved pattern matching"""
        try:
            doc = self.nlp(chunk.lower())
            scores = defaultdict(float)

            # Pattern matching with weights
            for content_type, info in self.content_patterns.items():
                for pattern in info['patterns']:
                    if pattern in chunk.lower():
                        scores[content_type] += info['weight']

            # Analyze linguistic features
            self._analyze_linguistic_features(doc, scores)
            
            # Normalize scores
            total_score = sum(scores.values()) or 1
            normalized_scores = {k: (v / total_score) * 100 for k, v in scores.items()}
            
            return normalized_scores or {'general': 100.0}

        except Exception as e:
            self.logger.error(f"Error identifying content type: {str(e)}")
            return {'general': 100.0}

    def _analyze_linguistic_features(self, doc: spacy.tokens.Doc, scores: Dict[str, float]):
        """Analyze linguistic features for content type identification"""
        try:
            # Entity analysis
            entity_types = [ent.label_ for ent in doc.ents]
            if any(ent in ['DATE', 'TIME'] for ent in entity_types):
                scores['instructional'] += 0.5
            if any(ent in ['PERSON', 'ORG'] for ent in entity_types):
                scores['narrative'] += 0.5

            # Syntactic analysis
            if any(token.dep_ == 'nummod' for token in doc):
                scores['instructional'] += 0.3
            if len([token for token in doc if token.pos_ == 'VERB' and token.dep_ == 'ROOT']):
                scores['instructional'] += 0.4

            # Technical terminology
            technical_pos = ['NOUN', 'PROPN']
            if len([token for token in doc if token.pos_ in technical_pos]) / len(doc) > 0.3:
                scores['technical'] += 0.5

        except Exception as e:
            self.logger.error(f"Error analyzing linguistic features: {str(e)}")

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts with GPU optimization"""
        try:
            with torch.no_grad():
                # KeyBERT keyword extraction
                keywords = self.key_phrase_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=15,
                    top_n=8
                )

                # Extract named entities
                doc = self.nlp(text)
                entities = [ent.text for ent in doc.ents]

                # Combine and deduplicate
                concepts = list(set([kw[0] for kw in keywords] + entities))
                
                return concepts[:10]  # Return top 10 concepts

        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {str(e)}")
            return []

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.device == 'cuda':
                self.gpu_optimizer.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")