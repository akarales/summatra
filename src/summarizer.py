#!/usr/bin/env python3
import os
import gc
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
import time
import json
from collections import defaultdict
from rich.console import Console
import whisper
import torch
import numpy as np
from transformers import pipeline
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.logging import RichHandler
from tqdm import tqdm
from .content_processor import ContentProcessor
from .model_manager import ModelManager
from .video_handler import VideoHandler
from .cuda_setup import (
    configure_cuda,
    safe_cuda_operation,
    cleanup_cuda_memory,
    get_memory_status
)
from .utils import (
    create_visualizations,
    add_visualization_to_summary,
    setup_logging,
    create_directory_structure,
    verify_ffmpeg_installation
)

console = Console()

class VideoSummarizer:
    def __init__(self, device: str = None, models_dir: str = "models"):
        """Initialize VideoSummarizer with optimized device settings"""
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger('VideoSummarizer')
        
        try:
            self.logger.info("Initializing video summarizer components...")
            
            # Setup device
            if device is None:
                cuda_config = configure_cuda(memory_fraction=0.3)
                self.device = cuda_config['device']
                if self.device == 'cuda':
                    self.logger.info(f"Using GPU: {cuda_config['name']} with {cuda_config['memory_fraction']*100}% memory allocation")
                else:
                    self.logger.info("Using CPU")
            else:
                self.device = device
            
            # Initialize components
            self._initialize_components()
            self._initialize_spacy()
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def _initialize_spacy(self):
        """Initialize spaCy model with proper error handling"""
        try:
            # Add explicit spaCy initialization in VideoSummarizer.__init__
            self.logger.info("Loading spaCy model...")
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.info("Downloading spaCy model...")
                import os
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            self.logger.error(f"Error initializing spaCy: {str(e)}")
            self.nlp = None

    def _initialize_components(self):
        """Initialize all components with proper error handling"""
        try:
            # Initialize basic components
            self.model_manager = ModelManager(models_dir=self.models_dir)
            self.content_processor = ContentProcessor(
                models_dir=self.models_dir,
                device=self.device
            )
            self.video_handler = VideoHandler()

            # Initialize Whisper
            self._initialize_whisper()

            # Initialize summarization pipeline
            self._initialize_summarizer()

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            raise

    def _initialize_whisper(self):
        """Initialize Whisper model with proper error handling"""
        console.print("\n🔄 Initializing Whisper model...", style="blue")
        
        try:
            # Determine model size based on available GPU memory
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
                
                if gpu_memory > 10:
                    model_size = "base"
                elif gpu_memory > 6:
                    model_size = "small"
                else:
                    model_size = "tiny"
            else:
                model_size = "tiny"
                
            self.logger.info(f"Selected Whisper model size: {model_size}")
            
            # Clear GPU memory before loading
            if self.device == "cuda":
                cleanup_cuda_memory()
            
            # Load the model
            self.whisper_model = whisper.load_model(
                model_size,
                device=self.device if self.device == "cuda" else "cpu"
            )
            
            # Verify model loaded correctly
            if self.whisper_model is None:
                raise RuntimeError("Whisper model failed to load")
                
            console.print(f"✅ Whisper model ({model_size}) loaded successfully", style="green")
            
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {str(e)}")
            console.print("⚠️ Attempting to fall back to tiny model on CPU...", style="yellow")
            try:
                self.device = "cpu"
                self.whisper_model = whisper.load_model("tiny", device="cpu")
                console.print("✅ Fallback to tiny model successful", style="green")
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed: {str(fallback_error)}")
                raise RuntimeError("Could not initialize Whisper model") from fallback_error

    def _initialize_summarizer(self):
        """Initialize summarization pipeline with improved memory management"""
        console.print("\n🔄 Initializing summarization model...", style="blue")
        
        try:
            if self.device == "cuda":
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                free_memory = gpu_memory - allocated_memory
                
                self.logger.info(f"GPU Total Memory: {gpu_memory:.2f} GB")
                self.logger.info(f"Free Memory: {free_memory:.2f} GB")
                
                # Clear cache before loading
                torch.cuda.empty_cache()
                
                try:
                    # Try loading with GPU
                    self.summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-base",  # Use smaller base model
                        device=0,
                        torch_dtype=torch.float16,  # Use half precision
                    )
                    console.print("✅ Summarization model loaded successfully with GPU", 
                                style="green")
                    return
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load model with GPU: {str(e)}")
                    torch.cuda.empty_cache()
                    self.device = "cpu"  # Fall back to CPU
            
            # CPU initialization
            if self.device == "cpu":
                console.print("⚠️ Using CPU for summarization...", style="yellow")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-base",  # Use smaller model
                    device=-1,
                    torch_dtype=torch.float32
                )
                console.print("✅ CPU initialization successful", style="green")
                
        except Exception as e:
            self.logger.error(f"Summarizer initialization failed: {str(e)}")
            raise RuntimeError("Could not initialize summarizer") from e

    def transcribe_audio(self, audio_path: Union[str, Path]) -> str:
        """Transcribe audio using Whisper with better error handling"""
        try:
            console.print("\n🎙️ Starting transcription...", style="blue")
            start_time = time.time()
            
            # Verify whisper model
            if not hasattr(self, 'whisper_model'):
                raise AttributeError("Whisper model not initialized properly")
            
            # Clean up memory before transcription
            if self.device == "cuda":
                cleanup_cuda_memory()
            
            # Convert path to string if it's a Path object
            audio_file = str(audio_path)
            self.logger.debug(f"Transcribing audio file: {audio_file}")
            
            # Create progress bar without callback
            pbar = tqdm(total=100, desc="Transcribing", unit="%")
            pbar.update(10)  # Show initial progress
            
            # Transcribe with error handling
            try:
                result = self.whisper_model.transcribe(
                    audio_file,
                    fp16=torch.cuda.is_available() and self.device == "cuda",
                    language='en',
                    task='transcribe'
                )
                pbar.update(90)  # Complete the progress bar
                
            except RuntimeError as e:
                if "out of memory" in str(e) and self.device == "cuda":
                    self.logger.warning("GPU OOM during transcription, falling back to CPU")
                    cleanup_cuda_memory()
                    # Temporarily move model to CPU
                    self.whisper_model = self.whisper_model.to("cpu")
                    pbar.set_description("Transcribing (CPU)")
                    
                    result = self.whisper_model.transcribe(
                        audio_file,
                        language='en'
                    )
                    
                    # Move back to GPU if possible
                    if self.device == "cuda":
                        self.whisper_model = self.whisper_model.to("cuda")
                else:
                    raise
            finally:
                pbar.close()
            
            # Extract transcription
            transcript = result.get('text', '').strip()
            
            if not transcript:
                raise ValueError("Transcription produced empty result")
            
            processing_time = time.time() - start_time
            console.print(f"✅ Transcription completed in {processing_time:.2f}s!", style="green")
            
            # Save transcription immediately
            self._save_transcription(transcript, Path(audio_file).stem)
            
            return transcript
            
        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
    def _save_transcription(self, transcript: str, identifier: str) -> Path:
            """Save transcription to file with improved error handling"""
            try:
                results_dir = Path('results')
                results_dir.mkdir(exist_ok=True)
                
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                transcript_file = results_dir / f"transcript_{identifier}_{timestamp}.txt"
                
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                    
                self.logger.info(f"Transcription saved to: {transcript_file}")
                console.print(f"📝 Transcription saved to: {transcript_file}", style="green")
                
                return transcript_file
                
            except Exception as e:
                self.logger.error(f"Error saving transcription: {str(e)}")
                raise

    def process_video(self, url: str, cleanup: bool = False) -> Optional[Dict]:
        """Process video URL and generate summary"""
        try:
            start_time = time.time()
            self.logger.info(f"Starting video processing for URL: {url}")
            
            # Step 1: Download video
            console.print("\n📥 Downloading video...", style="blue")
            video_info = self.video_handler.download_video(url)
            
            if not video_info:
                raise Exception("Video download failed")
            
            download_time = time.time() - start_time
            self.logger.info(f"Download completed in {download_time:.2f}s")
            
            # Step 2: Generate transcription
            self.logger.info("Starting audio transcription...")
            audio_file = Path(video_info['audio_file'])
            transcript = self.transcribe_audio(audio_file)
            
            if not transcript:
                raise Exception("Transcription failed")
                
            # Step 3: Generate summary
            console.print("\n📝 Generating summary from transcription...", style="blue")
            summary_info = self._generate_final_summary(transcript, video_info)
            
            # Add processing metadata
            summary_info.update({
                'video_info': video_info,
                'processing_time': time.time() - start_time,
                'device_used': self.device,
                'memory_usage': get_memory_status()
            })
            
            # Save results
            console.print("\n💾 Saving results...", style="blue")
            output_file = self.save_results(summary_info)
            summary_info['output_file'] = str(output_file)
            
            # Cleanup if requested
            if cleanup:
                console.print("\n🧹 Cleaning up downloaded files...", style="blue")
                self.video_handler.cleanup_files(video_info)
            
            total_time = time.time() - start_time
            console.print(f"\n✅ Processing completed in {total_time:.2f} seconds", style="green")
            console.print(f"📁 Results saved to: {output_file}", style="blue")
            
            return summary_info
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return None

    def _generate_final_summary(self, transcript: str, video_info: Dict) -> Dict:
        """
        Internal method to generate final summary with enhanced memory management and error handling
        
        Args:
            transcript: Text transcript to summarize
            video_info: Dictionary containing video metadata
            
        Returns:
            Dictionary containing summary and analysis results
        """
        try:
            self.logger.info("Starting final summary generation")
            start_time = time.time()
            
            # Clear GPU memory before processing
            if self.device == "cuda":
                self._cleanup_cuda_memory()
            
            # Split transcript into manageable chunks
            words = transcript.split()
            total_words = len(words)
            self.logger.debug(f"Transcript length: {total_words} words")
            
            # Create smaller chunks for better memory management
            chunk_size = 300  # Reduced for stability
            chunks = []
            for i in range(0, total_words, chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            
            # Process chunks
            chunk_summaries = []
            chunk_sentiments = []
            
            for i, chunk in enumerate(chunks, 1):
                console.print(f"\nProcessing chunk {i}/{len(chunks)}...", style="yellow")
                
                try:
                    # Calculate appropriate lengths - reduced for safety
                    chunk_words = len(chunk.split())
                    max_length = min(chunk_words // 3, 100)  # Reduced max length
                    min_length = max(30, max_length // 2)    # Adjusted min length
                    
                    # Generate summary with safer configuration
                    generation_config = {
                        'max_length': max_length,
                        'min_length': min_length,
                        'do_sample': False,  # Disabled sampling for stability
                        'num_beams': 2,      # Reduced beam size
                        'early_stopping': True,
                        'no_repeat_ngram_size': 3,
                        'truncation': True
                    }
                    
                    # Use autocast for better memory efficiency
                    with torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                        summary = self.summarizer(chunk, **generation_config)
                    
                    if summary and summary[0]['summary_text']:
                        chunk_summaries.append(summary[0]['summary_text'])
                    
                    # Clear GPU cache after each chunk
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Process sentiment
                    try:
                        sentiment = self.content_processor.analyze_sentiment(chunk)
                        chunk_sentiments.append({
                            'segment': i,
                            'score': sentiment['score'],
                            'label': sentiment['label']
                        })
                    except Exception as e:
                        self.logger.warning(f"Error analyzing sentiment for chunk {i}: {str(e)}")
                        chunk_sentiments.append({
                            'segment': i,
                            'score': 0.5,
                            'label': 'NEUTRAL'
                        })
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.device == "cuda":
                        self.logger.warning(f"GPU OOM for chunk {i}, attempting recovery...")
                        self._cleanup_cuda_memory()
                        # Retry with smaller size
                        try:
                            generation_config['max_length'] = max(max_length // 2, 40)
                            generation_config['min_length'] = max(min_length // 2, 20)
                            with torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                                summary = self.summarizer(chunk, **generation_config)
                            if summary and summary[0]['summary_text']:
                                chunk_summaries.append(summary[0]['summary_text'])
                        except Exception as retry_error:
                            self.logger.error(f"Retry failed for chunk {i}: {str(retry_error)}")
                            chunk_summaries.append(chunk[:100] + "...")
                    else:
                        self.logger.warning(f"Error processing chunk {i}: {str(e)}")
                        chunk_summaries.append(chunk[:100] + "...")
            
            # Combine summaries with safe concatenation
            combined_summary = ' '.join(chunk_summaries)
            final_summary = combined_summary
            
            # Only attempt final summarization if combined summary isn't too long
            if len(combined_summary.split()) > 200:
                try:
                    final_summary_config = {
                        'max_length': 150,
                        'min_length': 50,
                        'do_sample': False,
                        'num_beams': 2,
                        'early_stopping': True,
                        'truncation': True
                    }
                    
                    with torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                        final_summary = self.summarizer(
                            combined_summary[:1000],  # Limit input length
                            **final_summary_config
                        )[0]['summary_text']
                except Exception as e:
                    self.logger.error(f"Error in final summarization: {str(e)}")
                    final_summary = combined_summary[:500] + "..."
            
            # Generate content analysis safely
            try:
                content_analysis = {
                    'content_types': self._analyze_content_types(transcript),
                    'key_concepts': self._extract_key_concepts(transcript),
                    'sentiment_timeline': chunk_sentiments
                }
            except Exception as e:
                self.logger.error(f"Error in content analysis: {str(e)}")
                content_analysis = {
                    'content_types': {'general': 100.0},
                    'key_concepts': [],
                    'sentiment_timeline': chunk_sentiments
                }
            
            # Calculate sentiment analysis
            sentiment_scores = [s['score'] for s in chunk_sentiments]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
            # Prepare final result
            result = {
                'summary': final_summary,
                'content_analysis': content_analysis,
                'sentiment_analysis': {
                    'average_score': avg_sentiment,
                    'detailed_scores': chunk_sentiments
                },
                'video_metadata': {
                    'title': video_info.get('title', 'Unknown'),
                    'duration': video_info.get('duration', 0),
                    'url': video_info.get('url', '')
                },
                'transcript': transcript,
                'processing_stats': {
                    'chunk_count': len(chunks),
                    'original_length': total_words,
                    'summary_length': len(final_summary.split()),
                    'compression_ratio': len(final_summary.split()) / total_words if total_words > 0 else 0,
                    'processing_time': time.time() - start_time
                }
            }
            
            self.logger.info(f"Summary generation completed in {time.time() - start_time:.2f} seconds")
            return result
                
        except Exception as e:
            self.logger.error(f"Error generating final summary: {str(e)}")
            # Return basic fallback summary
            return {
                'summary': transcript[:500] + "...",
                'content_analysis': {
                    'content_types': {'general': 100.0},
                    'key_concepts': [],
                    'sentiment_timeline': []
                },
                'sentiment_analysis': {
                    'average_score': 0.5,
                    'detailed_scores': []
                },
                'video_metadata': video_info,
                'transcript': transcript,
                'processing_stats': {
                    'chunk_count': 1,
                    'original_length': len(transcript.split()),
                    'summary_length': 500,
                    'compression_ratio': 0,
                    'processing_time': time.time() - start_time
                }
            }

    def _cleanup_cuda_memory(self):
        """
        Internal method to perform comprehensive CUDA memory cleanup
        Cleans up all GPU memory allocations and caches with proper error handling
        """
        if not torch.cuda.is_available():
            return
            
        try:
            self.logger.debug("Starting CUDA memory cleanup")
            
            # Basic CUDA cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Clear any accumulated memory stats if available
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                torch.cuda.reset_accumulated_memory_stats()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Move models to CPU temporarily if they exist
            try:
                if hasattr(self, 'summarizer'):
                    if hasattr(self.summarizer, 'model'):
                        self.summarizer.model.cpu()
                        torch.cuda.empty_cache()
                        if self.device == "cuda":
                            self.summarizer.model.cuda()
            except Exception as model_e:
                self.logger.warning(f"Error moving summarizer model: {str(model_e)}")
                
            try:
                if hasattr(self, 'whisper_model'):
                    self.whisper_model.cpu()
                    torch.cuda.empty_cache()
                    if self.device == "cuda":
                        self.whisper_model.cuda()
            except Exception as model_e:
                self.logger.warning(f"Error moving whisper model: {str(model_e)}")
                
            # Clear all gradients and tensors
            try:
                for obj in gc.get_objects():
                    if torch.is_tensor(obj):
                        if obj.is_cuda:
                            obj.detach_()
                            if hasattr(obj, 'cpu'):
                                obj.cpu()
                    elif hasattr(obj, 'cuda') and hasattr(obj, 'cpu'):
                        try:
                            if next(obj.parameters(), None) is not None:
                                obj.cpu()
                        except Exception:
                            pass
            except Exception as tensor_e:
                self.logger.warning(f"Error cleaning tensor objects: {str(tensor_e)}")
                
            # Additional CUDA optimizations
            try:
                # Reset memory fraction to default
                torch.cuda.set_per_process_memory_fraction(0.5)
                
                # Set gradients to None instead of zeroing them
                for param in self.parameters() if hasattr(self, 'parameters') else []:
                    if param.grad is not None:
                        param.grad = None
                        
                # Clear CUDA caches if using newer PyTorch versions
                if hasattr(torch.cuda, 'clear_memory_allocated'):
                    torch.cuda.clear_memory_allocated()
                    
                # Synchronize CUDA stream
                torch.cuda.synchronize()
                
            except Exception as opt_e:
                self.logger.warning(f"Error during CUDA optimizations: {str(opt_e)}")
                
            # Check and log memory status
            try:
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    self.logger.debug(
                        f"Memory Status after cleanup:\n"
                        f"  Total: {total_memory:.2f} GB\n"
                        f"  Allocated: {allocated_memory:.2f} GB\n"
                        f"  Reserved: {reserved_memory:.2f} GB"
                    )
                    
                    # Additional cleanup if too much memory is still allocated
                    if allocated_memory > 0.5 * total_memory:  # More than 50% still allocated
                        self.logger.warning("High memory usage after cleanup, attempting aggressive cleanup")
                        self._aggressive_cuda_cleanup()
                        
            except Exception as mem_e:
                self.logger.warning(f"Error getting memory status: {str(mem_e)}")
                
            self.logger.debug("CUDA memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during CUDA cleanup: {str(e)}")
            # Attempt basic fallback cleanup
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as fallback_e:
                self.logger.error(f"Fallback cleanup failed: {str(fallback_e)}")

    def _aggressive_cuda_cleanup(self):
        """
        Internal method for aggressive CUDA memory cleanup when normal cleanup isn't sufficient
        """
        try:
            # Force release of all CUDA memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                torch.cuda.reset_accumulated_memory_stats()
                
            # Clear model caches
            model_attributes = ['summarizer', 'whisper_model', 'sentiment_analyzer', 'content_processor']
            for attr in model_attributes:
                if hasattr(self, attr):
                    try:
                        model = getattr(self, attr)
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        if hasattr(model, 'clear_cache'):
                            model.clear_cache()
                    except Exception:
                        pass
                        
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                
            # Reset CUDA device
            if torch.cuda.is_available():
                try:
                    device_id = torch.cuda.current_device()
                    torch.cuda.device(device_id).empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Aggressive cleanup failed: {str(e)}")

    def _analyze_content_types(self, text: str) -> Dict[str, float]:
        """Analyze content types in text with improved pattern matching"""
        try:
            types = defaultdict(float)
            patterns = {
                'technical': {
                    'patterns': ['code', 'algorithm', 'system', 'technical', 'software', 'data', 'process'],
                    'weight': 1.2
                },
                'educational': {
                    'patterns': ['learn', 'understand', 'explain', 'tutorial', 'guide', 'lesson', 'teach'],
                    'weight': 1.0
                },
                'entertainment': {
                    'patterns': ['fun', 'amazing', 'awesome', 'cool', 'interesting', 'exciting', 'enjoy'],
                    'weight': 0.9
                },
                'news': {
                    'patterns': ['report', 'announcement', 'update', 'recent', 'latest', 'breaking', 'current'],
                    'weight': 1.1
                },
                'discussion': {
                    'patterns': ['opinion', 'think', 'believe', 'perspective', 'view', 'debate', 'discuss'],
                    'weight': 1.0
                }
            }
            
            # Count pattern matches with context
            text_lower = text.lower()
            words = text_lower.split()
            total_matches = 0
            
            # Analyze with context window
            window_size = 5
            for i in range(len(words)):
                window = ' '.join(words[max(0, i-window_size):min(len(words), i+window_size+1)])
                
                for content_type, info in patterns.items():
                    for pattern in info['patterns']:
                        if pattern in window:
                            # Add weighted score based on pattern location
                            distance_to_center = abs(window_size - i % (window_size * 2))
                            weight_factor = 1 - (distance_to_center / (window_size * 2))
                            types[content_type] += info['weight'] * weight_factor
                            total_matches += 1
            
            # Convert to percentages and normalize
            if total_matches > 0:
                types = {k: (v / total_matches) * 100 for k, v in types.items()}
                
                # Normalize to ensure total is 100%
                total = sum(types.values())
                if total > 0:
                    types = {k: (v / total) * 100 for k, v in types.items()}
            else:
                # Equal distribution if no matches
                num_types = len(patterns)
                types = {k: 100.0 / num_types for k in patterns.keys()}
            
            return dict(types)
            
        except Exception as e:
            self.logger.error(f"Error analyzing content types: {str(e)}")
            return {'general': 100.0}

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text with improved relevance scoring"""
        try:
            # Use content processor if available
            if hasattr(self, 'content_processor'):
                concepts = self.content_processor.extract_key_concepts(text)
                
                # Additional NLP processing for better concept extraction
                doc = self.nlp(text)
                
                # Extract named entities with scores
                entities = {}
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'EVENT']:
                        score = 1.0
                        # Boost score based on frequency and position
                        freq = text.lower().count(ent.text.lower())
                        position_factor = 1 - (ent.start / len(doc))
                        score = score * (1 + freq/10) * (1 + position_factor)
                        entities[ent.text] = score
                
                # Combine concepts and entities
                combined_concepts = set(concepts)
                for entity, score in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]:
                    combined_concepts.add(entity)
                
                return list(combined_concepts)[:15]  # Return top 15 concepts
            
            # Fallback to basic extraction
            words = text.lower().split()
            word_freq = defaultdict(int)
            for word in words:
                if len(word) > 3 and word not in nltk.corpus.stopwords.words('english'):
                    word_freq[word] += 1
            
            # Get top concepts
            concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return [word for word, _ in concepts]
            
        except Exception as e:
            self.logger.error(f"Error extracting key concepts: {str(e)}")
            return []

    def _create_visualizations(self, summary_info: Dict, output_dir: Path) -> Optional[Path]:
        """
        Internal method to create comprehensive visualizations of the content analysis
        
        Args:
            summary_info: Dictionary containing analysis results
            output_dir: Directory to save visualizations
            
        Returns:
            Path to saved visualization file or None if visualization fails
        """
        try:
            # Reset style and set theme
            plt.style.use('default')
            sns.set_theme(style='whitegrid')
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
            
            # Set color palette
            colors = sns.color_palette("husl", 8)
            
            # Update plot styling
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9
            })

            try:
                # 1. Content Type Distribution (Top Left)
                ax1 = fig.add_subplot(gs[0, 0])
                content_types = summary_info['content_analysis']['content_types']
                content_df = pd.DataFrame(list(content_types.items()),
                                        columns=['Type', 'Count'])
                
                sns.barplot(data=content_df,
                        x='Count',
                        y='Type',
                        palette=colors[:len(content_df)],
                        ax=ax1)
                ax1.set_title('Content Type Distribution')
                ax1.set_xlabel('Percentage')
                
                # Add percentage labels
                for i, v in enumerate(content_df['Count']):
                    ax1.text(v, i, f'{v:.1f}%', va='center')

            except Exception as e:
                self.logger.warning(f"Error creating content type plot: {str(e)}")
                ax1.text(0.5, 0.5, 'Content Type Plot Unavailable',
                        ha='center', va='center')

            try:
                # 2. Sentiment Timeline (Top Middle)
                ax2 = fig.add_subplot(gs[0, 1])
                sentiment_data = pd.DataFrame(summary_info['sentiment_analysis']['detailed_scores'])
                
                sns.lineplot(data=sentiment_data,
                            x='segment',
                            y='score',
                            marker='o',
                            color=colors[1],
                            ax=ax2)
                ax2.set_title('Sentiment Timeline')
                ax2.set_xlabel('Segment')
                ax2.set_ylabel('Sentiment Score')
                ax2.grid(True, alpha=0.3)

            except Exception as e:
                self.logger.warning(f"Error creating sentiment timeline: {str(e)}")
                ax2.text(0.5, 0.5, 'Sentiment Timeline Unavailable',
                        ha='center', va='center')

            try:
                # 3. Key Concepts (Top Right)
                ax3 = fig.add_subplot(gs[0, 2])
                key_concepts = summary_info['content_analysis']['key_concepts'][:10]
                concepts_df = pd.DataFrame({
                    'Concept': key_concepts,
                    'Frequency': range(len(key_concepts), 0, -1)
                })
                
                sns.barplot(data=concepts_df,
                        x='Frequency',
                        y='Concept',
                        palette=sns.color_palette("rocket", len(concepts_df)),
                        ax=ax3)
                ax3.set_title('Top Key Concepts')
                
                # Add frequency labels
                for i, v in enumerate(concepts_df['Frequency']):
                    ax3.text(v, i, str(v), va='center')

            except Exception as e:
                self.logger.warning(f"Error creating key concepts plot: {str(e)}")
                ax3.text(0.5, 0.5, 'Key Concepts Plot Unavailable',
                        ha='center', va='center')

            try:
                # 4. Processing Stats (Middle Row)
                ax4 = fig.add_subplot(gs[1, :])
                stats = summary_info['processing_stats']
                stats_df = pd.DataFrame([
                    {'Metric': key, 'Value': value}
                    for key, value in stats.items()
                    if isinstance(value, (int, float))  # Ensure numeric values
                ])
                
                sns.barplot(data=stats_df,
                        x='Metric',
                        y='Value',
                        palette='viridis',
                        ax=ax4)
                ax4.set_title('Processing Statistics')
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for i, v in enumerate(stats_df['Value']):
                    ax4.text(i, v, f'{v:.2f}', ha='center', va='bottom')

            except Exception as e:
                self.logger.warning(f"Error creating processing stats plot: {str(e)}")
                ax4.text(0.5, 0.5, 'Processing Stats Plot Unavailable',
                        ha='center', va='center')

            try:
                # 5. Sentiment Distribution (Bottom Left)
                ax5 = fig.add_subplot(gs[2, 0])
                sentiment_df = pd.DataFrame(summary_info['sentiment_analysis']['detailed_scores'])
                
                sns.histplot(data=sentiment_df,
                            x='score',
                            bins=20,
                            color=colors[2],
                            ax=ax5)
                ax5.set_title('Sentiment Distribution')
                ax5.set_xlabel('Sentiment Score')
                ax5.set_ylabel('Count')

            except Exception as e:
                self.logger.warning(f"Error creating sentiment distribution: {str(e)}")
                ax5.text(0.5, 0.5, 'Sentiment Distribution Unavailable',
                        ha='center', va='center')

            try:
                # 6. Word Count Analysis (Bottom Middle)
                ax6 = fig.add_subplot(gs[2, 1])
                word_counts = {
                    'Original': stats['original_length'],
                    'Summary': stats['summary_length']
                }
                
                count_df = pd.DataFrame(list(word_counts.items()),
                                    columns=['Type', 'Count'])
                
                sns.barplot(data=count_df,
                        x='Type',
                        y='Count',
                        palette=colors[3:5],
                        ax=ax6)
                ax6.set_title('Word Count Comparison')
                
                # Add count labels
                for i, v in enumerate(count_df['Count']):
                    ax6.text(i, v, str(v), ha='center', va='bottom')

            except Exception as e:
                self.logger.warning(f"Error creating word count plot: {str(e)}")
                ax6.text(0.5, 0.5, 'Word Count Plot Unavailable',
                        ha='center', va='center')

            try:
                # 7. Compression Ratio (Bottom Right)
                ax7 = fig.add_subplot(gs[2, 2])
                compression = stats['compression_ratio'] * 100
                
                sns.barplot(x=['Compression'],
                        y=[compression],
                        color=colors[5],
                        ax=ax7)
                ax7.set_title('Compression Ratio')
                ax7.set_ylabel('Percentage')
                ax7.set_ylim(0, 100)
                
                # Add percentage label
                ax7.text(0, compression, f'{compression:.1f}%',
                        ha='center', va='bottom')

            except Exception as e:
                self.logger.warning(f"Error creating compression ratio plot: {str(e)}")
                ax7.text(0.5, 0.5, 'Compression Ratio Plot Unavailable',
                        ha='center', va='center')

            # Add title and metadata
            try:
                title = summary_info['video_metadata']['title']
                title = (title[:50] + '...') if len(title) > 50 else title
                plt.suptitle(f"Video Analysis: {title}",
                            fontsize=16, y=0.95)
            except Exception as e:
                self.logger.warning(f"Error adding title: {str(e)}")
                plt.suptitle("Video Analysis", fontsize=16, y=0.95)

            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save visualizations
            vis_path = output_dir / 'visualizations'
            vis_path.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            plot_file = vis_path / f"analysis_{timestamp}.png"
            
            # Save with high quality settings
            plt.savefig(plot_file,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.5,
                    facecolor='white',
                    edgecolor='none',
                    metadata={'Title': 'Video Analysis Visualization'})
            
            plt.close()
            
            self.logger.info(f"Visualization saved to: {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            return None

    def save_results(self, summary_info: Dict) -> Path:
        """Save results and generate visualizations"""
        try:
            # Create results directory
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Generate visualizations
            try:
                console.print("\n📊 Generating visualizations...", style="blue")
                visualization_path = self._create_visualizations(summary_info, results_dir)
                
                if visualization_path:
                    console.print("✅ Visualizations generated", style="green")
                    summary_info = add_visualization_to_summary(summary_info, visualization_path)
                else:
                    console.print("⚠️ Visualization generation failed", style="yellow")
            except Exception as e:
                self.logger.warning(f"Error generating visualizations: {str(e)}")
                console.print("⚠️ Visualization generation failed", style="yellow")
            
            # Generate filename from video metadata
            video_id = summary_info['video_info']['video_id']
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = results_dir / f"summary_{video_id}_{timestamp}.json"
            
            # Save results with proper formatting
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_info, f, indent=4, ensure_ascii=False)
            
            return output_file
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            # Clean up CUDA memory
            if self.device == "cuda":
                cleanup_cuda_memory()
            
            # Clean up downloaded models
            if hasattr(self, 'model_manager'):
                self.model_manager.cleanup_models()
            
            # Clean up temporary files
            if hasattr(self, 'video_handler'):
                self.video_handler.cleanup_temp_files()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error(f"Error during execution: {exc_type.__name__}: {exc_val}")
            return False
        return True


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = setup_logging()
    
    try:
        # Test the summarizer
        with VideoSummarizer() as summarizer:
            print(f"Initialized with device: {summarizer.device}")
            
            # Test URL
            test_url = "https://www.youtube.com/watch?v=example"
            result = summarizer.process_video(test_url)
            
            if result:
                print("\nTest Results:")
                print("=" * 40)
                print(f"Summary file: {result['output_file']}")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
                print(f"Device used: {result['device_used']}")
                print("\nSummary:")
                print("-" * 40)
                print(result['summary'])
                
                if 'processing_stats' in result:
                    print("\nProcessing Statistics:")
                    print("-" * 40)
                    for key, value in result['processing_stats'].items():
                        print(f"{key}: {value}")
            else:
                print("❌ Processing failed")
                
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}", exc_info=True)