#!/usr/bin/env python3
import torch
import spacy
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import nltk
from transformers import pipeline
import os
from .cuda_setup import configure_cuda, safe_cuda_operation, get_optimal_device

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
        self.device = device if device else get_optimal_device()
        
        # Download required NLTK data
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('stopwords')
            nltk.download('wordnet')
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
            
            self.sentence_model = self.safe_cuda_operation(
                load_sentence_transformer,
                lambda: SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
            )

            # Initialize keyword extraction
            self.logger.info("Initializing keyword extraction...")
            try:
                def init_keybert():
                    return KeyBERT(model=self.sentence_model)
                
                self.key_phrase_model = self.safe_cuda_operation(
                    init_keybert,
                    lambda: KeyBERT()  # Fallback to default model
                )
            except Exception as e:
                self.logger.warning(f"Error initializing KeyBERT with custom model: {str(e)}")
                self.key_phrase_model = KeyBERT()

            # Load sentiment analysis with GPU support
            self.logger.info("Loading sentiment analysis model...")
            def load_sentiment_analyzer():
                return pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if self.device == "cuda" else -1
                )
            
            self.sentiment_analyzer = self.safe_cuda_operation(
                load_sentiment_analyzer,
                lambda: pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=-1
                )
            )

            # Initialize content type patterns
            self.setup_content_patterns()
            
            # Clean up any GPU memory after model loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
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

    def preprocess_transcript(self, transcription: str) -> List[str]:
        """
        Preprocess and segment transcript into coherent chunks
        
        Args:
            transcription: Raw transcript text
        
        Returns:
            List of processed text chunks
        """
        try:
            # Verify NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                self.logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt')
            
            # Segment the transcription into sentences
            sentences = nltk.tokenize.sent_tokenize(transcription)
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(transcription)
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error in preprocessing transcription: {str(e)}")
            return []

    def _create_semantic_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Create semantic chunks using GPU when available
        
        Args:
            text: Input text
            max_chunk_size: Maximum size of each chunk
        """
        try:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_length = 0

            # Calculate sentence embeddings
            embeddings = self.sentence_model.encode(
                sentences,
                convert_to_tensor=True,
                device=self.device
            )

            for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
                if current_length + len(sentence) > max_chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                if current_chunk and i > 0:
                    prev_embedding = embeddings[i-1]
                    similarity = torch.cosine_similarity(
                        embedding.unsqueeze(0),
                        prev_embedding.unsqueeze(0)
                    ).item()

                    if similarity < 0.7:  # Topic change threshold
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_length = 0

                current_chunk.append(sentence)
                current_length += len(sentence)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        except Exception as e:
            self.logger.error(f"Error creating semantic chunks: {str(e)}")
            return self._create_semantic_chunks_cpu(text)

    def _create_semantic_chunks_cpu(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        CPU fallback for semantic chunking
        
        Args:
            text: Input text
            max_chunk_size: Maximum size of each chunk
        """
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sentence)
            current_length += len(sentence)
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def process_content(self, chunk: str, summarizer) -> Dict:
        """
        Process content chunk with GPU acceleration when available
        
        Args:
            chunk: Text chunk to process
            summarizer: Summarization model
        """
        try:
            return safe_cuda_operation(
                lambda: self._process_content_gpu(chunk, summarizer),
                lambda: self.process_content_cpu(chunk, summarizer)
            )
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return self._fallback_content_processing(chunk)

    def _process_content_gpu(self, chunk: str, summarizer) -> Dict:
        """
        GPU-accelerated content processing
        
        Args:
            chunk: Text chunk to process
            summarizer: Summarization model
        """
        content_type = self.identify_content_type(chunk)
        key_concepts = self.extract_key_concepts(chunk)
        sentiment = self.analyze_sentiment(chunk)
        
        summary = self.generate_focused_summary(
            chunk, 
            content_type, 
            key_concepts,
            sentiment,
            summarizer
        )

        return {
            'summary': summary,
            'content_type': content_type,
            'key_concepts': key_concepts,
            'sentiment': sentiment
        }
    def safe_cuda_operation(self, operation: callable, fallback: Optional[callable] = None) -> any:
        """
        Safely execute CUDA operations with fallback
        
        Args:
            operation: Function to execute
            fallback: Optional fallback function for CPU execution
            
        Returns:
            Result of operation or fallback
        """
        if not torch.cuda.is_available():
            return fallback() if fallback else operation()
            
        try:
            result = operation()
            torch.cuda.empty_cache()
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.warning(f"CUDA out of memory: {str(e)}")
                torch.cuda.empty_cache()
                if fallback:
                    return fallback()
            raise
        except Exception as e:
            self.logger.error(f"CUDA operation error: {str(e)}")
            if fallback:
                return fallback()
            raise

    def process_content_cpu(self, chunk: str, summarizer) -> Dict:
        """
        CPU fallback for content processing
        
        Args:
            chunk: Text chunk to process
            summarizer: Summarization model
        """
        try:
            # Basic content type identification
            content_type = self.identify_content_type(chunk)
            
            # Simple keyword extraction
            words = chunk.split()
            word_freq = defaultdict(int)
            for word in words:
                if len(word) > 3:
                    word_freq[word.lower()] += 1
            key_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            key_concepts = [word for word, _ in key_concepts]
            
            # Generate summary
            summary = summarizer(
                chunk,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']

            return {
                'summary': summary,
                'content_type': content_type,
                'key_concepts': key_concepts,
                'sentiment': {'label': 'NEUTRAL', 'score': 0.5}
            }
        except Exception as e:
            self.logger.error(f"CPU processing failed: {str(e)}")
            return self._fallback_content_processing(chunk)

    def _fallback_content_processing(self, chunk: str) -> Dict:
        """
        Minimal fallback processing when both GPU and CPU methods fail
        
        Args:
            chunk: Text chunk to process
        """
        return {
            'summary': chunk[:200] + "..." if len(chunk) > 200 else chunk,
            'content_type': 'general',
            'key_concepts': [],
            'sentiment': {'label': 'NEUTRAL', 'score': 0.5}
        }

    def identify_content_type(self, chunk: str) -> str:
        """
        Identify content type using multiple features
        
        Args:
            chunk: Text chunk to analyze
        """
        try:
            doc = self.nlp(chunk.lower())
            scores = defaultdict(float)

            # Pattern matching with weights
            for content_type, info in self.content_patterns.items():
                for pattern in info['patterns']:
                    if pattern in chunk.lower():
                        scores[content_type] += info['weight']

            # Linguistic feature analysis
            self._analyze_linguistic_features(doc, scores)
            
            # Get highest scoring type
            if scores:
                return max(scores.items(), key=lambda x: x[1])[0]
            return 'general'

        except Exception as e:
            self.logger.error(f"Error identifying content type: {str(e)}")
            return 'general'

    def _analyze_linguistic_features(self, doc: spacy.tokens.Doc, scores: Dict[str, float]):
        """
        Analyze linguistic features for content type identification
        
        Args:
            doc: spaCy document
            scores: Score dictionary to update
        """
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

    def extract_key_concepts(self, chunk: str) -> List[str]:
        """
        Extract key concepts using multiple methods
        
        Args:
            chunk: Text chunk to analyze
            
        Returns:
            List of key concepts
        """
        try:
            # KeyBERT keyword extraction
            keywords = self.key_phrase_model.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=15,
                top_n=8
            )

            # Extract named entities
            doc = self.nlp(chunk)
            entities = [ent.text for ent in doc.ents]

            # Combine and deduplicate
            concepts = list(set([kw[0] for kw in keywords] + entities))
            
            return concepts[:10]  # Return top 10 concepts

        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {str(e)}")
            return []

    def analyze_sentiment(self, chunk: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of content
        
        Args:
            chunk: Text chunk to analyze
            
        Returns:
            Dictionary containing sentiment label and score
        """
        try:
            result = self.sentiment_analyzer(chunk[:512])[0]  # Limit chunk size
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def generate_focused_summary(
        self, 
        chunk: str, 
        content_type: str, 
        key_concepts: List[str],
        sentiment: Dict[str, Union[str, float]],
        summarizer
    ) -> str:
        """
        Generate context-aware summary
        
        Args:
            chunk: Text chunk to summarize
            content_type: Type of content
            key_concepts: List of key concepts
            sentiment: Sentiment analysis results
            summarizer: Summarization model
            
        Returns:
            Generated summary text
        """
        try:
            # Create focused prompt based on content type
            prompts = {
                'technical': f"Explain this technical content about {', '.join(key_concepts)}: ",
                'educational': f"Summarize these learning points about {', '.join(key_concepts)}: ",
                'narrative': f"Summarize this story involving {', '.join(key_concepts)}: ",
                'descriptive': f"Describe this content about {', '.join(key_concepts)}: ",
                'instructional': f"Outline these steps involving {', '.join(key_concepts)}: ",
                'analytical': f"Summarize these findings about {', '.join(key_concepts)}: ",
                'general': f"Summarize this content about {', '.join(key_concepts)}: "
            }

            prompt = prompts.get(content_type, prompts['general']) + chunk

            # Parameters based on content type
            params = {
                'technical': {'max_length': 150, 'min_length': 50, 'num_beams': 4},
                'educational': {'max_length': 200, 'min_length': 75, 'num_beams': 4},
                'narrative': {'max_length': 175, 'min_length': 50, 'num_beams': 3},
                'descriptive': {'max_length': 150, 'min_length': 50, 'num_beams': 3},
                'instructional': {'max_length': 200, 'min_length': 75, 'num_beams': 4},
                'analytical': {'max_length': 175, 'min_length': 50, 'num_beams': 4},
                'general': {'max_length': 150, 'min_length': 50, 'num_beams': 3}
            }

            try:
                # Generate summary with error handling
                summary = summarizer(
                    prompt,
                    **params.get(content_type, params['general']),
                    do_sample=False
                )[0]['summary_text']
                return summary
            except Exception as e:
                self.logger.error(f"Error in summarizer: {str(e)}")
                return chunk[:200] + "..."  # Fallback to truncated text

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return chunk[:200] + "..."  # Fallback to truncated text

if __name__ == "__main__":
    # Test the content processor
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize processor
        processor = ContentProcessor()
        print(f"Initialized with device: {processor.device}")
        
        # Test text
        test_text = """
        This is a test text for content processing. It demonstrates the basic functionality
        of the content processor including sentiment analysis, key concept extraction,
        and content type identification. The processor can handle technical content,
        educational material, and various other types of content.
        """
        
        # Create a mock summarizer for testing
        mock_summarizer = lambda text, **kwargs: [{'summary_text': text[:100] + "..."}]
        
        # Process the test text
        result = processor.process_content(test_text, mock_summarizer)
        
        print("\nTest Results:")
        print("=" * 40)
        for key, value in result.items():
            print(f"\n{key}:")
            print("-" * 20)
            print(value)
            
    except Exception as e:
        print(f"Test failed: {str(e)}")