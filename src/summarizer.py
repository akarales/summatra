import logging
import time
import json
import whisper
import gc
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import torch
from transformers import AutoTokenizer, pipeline
from rich.console import Console

from .pipeline import (
    ChunkManager,
    BatchProcessor,
    GPUOptimizer,
    ProcessingStatus
)
from .content_processor import ContentProcessor
from .model_manager import ModelManager
from .video_handler import VideoHandler
from .pdf_generator import PDFGenerator
from .utils import (
    create_visualizations,
    add_visualization_to_summary,
    setup_logging,
    create_directory_structure,
    verify_ffmpeg_installation
)

console = Console()

class VideoSummarizer:
    """Main video summarization class with optimized processing"""
    
    def __init__(self, device: str = None, models_dir: str = "models", results_dir: str = "results"):
        """Initialize video summarizer with optimized components"""
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger('VideoSummarizer')
        
        try:
            self.logger.info("Initializing video summarizer components...")
            # Create necessary directories
            self.models_dir.mkdir(exist_ok=True)
            self.results_dir.mkdir(exist_ok=True)
            # Setup device and GPU optimization
            self.gpu_optimizer = GPUOptimizer(memory_fraction=0.3)
            self.device = device if device else ('cuda' if self.gpu_optimizer.setup() else 'cpu')
            
            if self.device == 'cuda':
                gpu_info = self.gpu_optimizer.get_gpu_info()
                self.logger.info(f"Using GPU: {gpu_info['name']} with "
                               f"{gpu_info['free_memory']:.2f}GB free memory")
            else:
                self.logger.info("Using CPU")
            
            # Initialize components
            self._initialize_components()
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def _initialize_components(self):
        """Initialize processing components with error handling"""
        try:
            # Initialize basic components
            self.model_manager = ModelManager(models_dir=self.models_dir)
            self.content_processor = ContentProcessor(
                models_dir=self.models_dir,
                device=self.device
            )
            self.video_handler = VideoHandler()

            # Initialize Whisper with optimal settings
            self._initialize_whisper()

            # Initialize summarization pipeline
            self._initialize_summarizer()

            # Initialize chunk manager
            self.chunk_manager = ChunkManager(chunk_size=300)
            
            # Initialize batch processor
            sample_size = 512  # Base size for transformer models
            batch_size = self.gpu_optimizer.get_optimal_batch_size(sample_size)
            
            self.batch_processor = BatchProcessor(
                model=self.summarizer,
                tokenizer=self.tokenizer,
                device=self.device,
                batch_size=batch_size
            )

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def _initialize_whisper(self):
        """Initialize Whisper model with optimal settings"""
        self.logger.info("Initializing Whisper model...")
        
        try:
            # Determine model size based on available GPU memory
            if self.device == "cuda":
                gpu_info = self.gpu_optimizer.get_gpu_info()
                available_memory = gpu_info['free_memory']
                
                if available_memory > 10:
                    model_size = "base"
                elif available_memory > 6:
                    model_size = "small"
                else:
                    model_size = "tiny"
            else:
                model_size = "tiny"
                
            self.logger.info(f"Selected Whisper model size: {model_size}")
            
            # Clean GPU memory before loading model
            if self.device == "cuda":
                self.gpu_optimizer.cleanup()
            
            # Load the model
            self.whisper_model = whisper.load_model(
                model_size,
                device=self.device
            )
            
            if self.whisper_model is None:
                raise RuntimeError("Whisper model failed to load")
                
            self.logger.info(f"Whisper model ({model_size}) loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading Whisper model: {str(e)}")
            self.logger.info("Attempting to fall back to tiny model on CPU...")
            
            try:
                self.whisper_model = whisper.load_model("tiny", device="cpu")
                self.logger.info("Fallback to tiny model successful")
            except Exception as fallback_error:
                self.logger.error(f"Whisper fallback failed: {str(fallback_error)}")
                raise RuntimeError("Could not initialize Whisper model") from fallback_error

    def _initialize_summarizer(self):
        """Initialize summarization pipeline with GPU optimization"""
        self.logger.info("Initializing summarization model...")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            
            # Configure GPU settings
            if self.device == "cuda":
                gpu_info = self.gpu_optimizer.get_gpu_info()
                self.logger.info(f"GPU Memory Available: {gpu_info['free_memory']:.2f}GB")
                
                # Clear cache before loading
                self.gpu_optimizer.cleanup()
                
                try:
                    # Load model with GPU optimization
                    self.summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-base",
                        tokenizer=self.tokenizer,
                        device=0,
                        torch_dtype=torch.float16,  # Use half precision for GPU
                        framework="pt"
                    )
                    self.logger.info("Summarization model loaded successfully with GPU")
                    return
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load model with GPU: {str(e)}")
                    self.gpu_optimizer.cleanup()
                    self.device = "cpu"  # Fall back to CPU
            
            # CPU initialization
            self.logger.info("Using CPU for summarization...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-base",
                tokenizer=self.tokenizer,
                device=-1,
                framework="pt"
            )
            self.logger.info("Summarization model loaded successfully on CPU")
                
        except Exception as e:
            error_msg = f"Summarizer initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def transcribe_audio(self, audio_file: Path) -> Optional[str]:
        """
        Transcribe audio file using Whisper model
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            self.logger.info("\n🎙️ Transcribing audio...")
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
                
            # Load audio using whisper's internal loading
            # This handles various formats and resampling automatically
            audio = whisper.load_audio(str(audio_file))
            
            # Get optimal chunk size based on available memory
            gpu_info = self.gpu_optimizer.get_gpu_info() if self.device == 'cuda' else None
            max_chunk_duration = 30 * 60  # 30 minutes default
            
            if gpu_info and gpu_info.get('free_memory', 0) < 4:  # Less than 4GB free
                max_chunk_duration = 10 * 60  # 10 minutes for low memory
                
            # Process audio in chunks if needed
            audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
            if audio_duration > max_chunk_duration:
                return self._transcribe_long_audio(audio, max_chunk_duration)
            
            # Transcribe with optimal settings
            transcription_options = {
                'language': 'en',  # Can be made configurable
                'task': 'transcribe',
                'fp16': self.device == 'cuda',
                'verbose': False
            }
            
            # Clear GPU memory before transcription
            if self.device == 'cuda':
                self.gpu_optimizer.cleanup()
                
            # Run transcription
            result = self.whisper_model.transcribe(
                audio,
                **transcription_options
            )
            
            if not result or 'text' not in result:
                raise ValueError("Transcription failed to produce text")
                
            # Post-process transcription
            transcript = self._post_process_transcript(result['text'])
            
            self.logger.info("✅ Transcription completed successfully")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            return None

    def _transcribe_long_audio(self, audio: np.ndarray, chunk_duration: int) -> Optional[str]:
        """
        Transcribe long audio files by processing in chunks
        
        Args:
            audio: Audio data as numpy array
            chunk_duration: Maximum duration of each chunk in seconds
            
        Returns:
            Combined transcription text
        """
        try:
            sample_rate = whisper.audio.SAMPLE_RATE
            chunk_size = chunk_duration * sample_rate
            chunks = [
                audio[i:i + chunk_size] 
                for i in range(0, len(audio), chunk_size)
            ]
            
            transcriptions = []
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"\nTranscribing chunk {i}/{len(chunks)}...")
                
                # Clear GPU memory before each chunk
                if self.device == 'cuda':
                    self.gpu_optimizer.cleanup()
                
                result = self.whisper_model.transcribe(
                    chunk,
                    language='en',
                    task='transcribe',
                    fp16=self.device == 'cuda',
                    verbose=False
                )
                
                if result and 'text' in result:
                    transcriptions.append(result['text'])
                
            # Combine and post-process transcriptions
            combined_text = ' '.join(transcriptions)
            return self._post_process_transcript(combined_text)
            
        except Exception as e:
            self.logger.error(f"Error transcribing long audio: {str(e)}")
            return None

    def _post_process_transcript(self, text: str) -> str:
        """Clean and format transcribed text"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Fix common transcription artifacts
            text = text.replace(' .', '.')
            text = text.replace(' ,', ',')
            text = text.replace(' ?', '?')
            text = text.replace(' !', '!')
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error post-processing transcript: {str(e)}")
            return text

    def process_video(self, url: str, cleanup: bool = False) -> Optional[Dict]:
        """Process video URL with optimized pipeline"""
        try:
            start_time = time.time()
            self.logger.info(f"Starting video processing for URL: {url}")
            
            # Download video
            console.print("\n📥 Downloading video...", style="blue")
            video_info = self.video_handler.download_video(url)
            
            if not video_info:
                raise Exception("Video download failed")
            
            download_time = time.time() - start_time
            self.logger.info(f"Download completed in {download_time:.2f}s")
            
            # Generate transcription
            self.logger.info("Starting audio transcription...")
            audio_file = Path(video_info['audio_file'])
            transcript = self.transcribe_audio(audio_file)
            
            if not transcript:
                raise Exception("Transcription failed")
                
            # Generate summary
            console.print("\n📝 Generating summary from transcription...", style="blue")
            summary_info = self._generate_final_summary(transcript, video_info)
            
            # Add processing metadata
            summary_info.update({
                'video_info': video_info,
                'processing_time': time.time() - start_time,
                'device_used': self.device,
                'gpu_info': self.gpu_optimizer.get_gpu_info() if self.device == 'cuda' else None,
                'transcript': transcript
            })
            
            # Save results
            console.print("\n💾 Saving results...", style="blue")
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Save transcript to separate file
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            video_id = video_info['video_id']
            
            transcript_file = results_dir / f"transcript_{video_id}_{timestamp}.txt"
            output_file = results_dir / f"summary_{video_id}_{timestamp}.json"
            
            # Save transcript
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Save summary
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_info, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Results saved to: {output_file}")
            self.logger.info(f"Transcript saved to: {transcript_file}")
            
            # Add file paths to result
            summary_info['output_file'] = str(output_file)
            summary_info['transcript_file'] = str(transcript_file)
            
            # Generate visualizations
            console.print("\n📊 Generating visualizations...", style="blue")
            visualization_path = create_visualizations(summary_info, results_dir)
            
            if visualization_path:
                console.print("✅ Visualizations generated", style="green")
                summary_info = add_visualization_to_summary(summary_info, visualization_path)
            else:
                console.print("⚠️ Visualization generation failed", style="yellow")

            # Generate PDF documents
            console.print("\n📄 Generating PDF documents...", style="blue")
            pdf_files = self.generate_pdf_documents(summary_info)
            
            if pdf_files:
                console.print("\n✅ PDF documents generated successfully:", style="green")
                for doc_type, path in pdf_files.items():
                    console.print(f"  • {doc_type}: {path}", style="green")
                
                # Update summary info with PDF paths
                summary_info['pdf_documents'] = {
                    name: str(path) for name, path in pdf_files.items()
                }
                
                # Update the saved JSON with PDF information
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_info, f, indent=4, ensure_ascii=False)
            else:
                console.print("⚠️ PDF generation failed", style="yellow")
            
            # Cleanup if requested
            if cleanup:
                console.print("\n🧹 Cleaning up downloaded files...", style="blue")
                self.video_handler.cleanup_files(video_info)
            
            total_time = time.time() - start_time
            console.print(f"\n✅ Processing completed in {total_time:.2f} seconds", style="green")
            
            # Print summary of generated files
            console.print("\n📁 Generated Files:", style="blue")
            console.print(f"  • Summary JSON: {output_file}", style="green")
            console.print(f"  • Transcript: {transcript_file}", style="green")
            if visualization_path:
                console.print(f"  • Visualizations: {visualization_path}", style="green")
            if pdf_files:
                for doc_type, path in pdf_files.items():
                    console.print(f"  • {doc_type}: {path}", style="green")
            
            return summary_info
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return None
        finally:
            # Always cleanup GPU memory
            if self.device == 'cuda':
                self.gpu_optimizer.cleanup()

    def generate_pdf_documents(self, summary_info: Dict) -> Dict[str, Path]:
        """
        Generate PDF documents for summary, transcript, and visualizations
        
        Args:
            summary_info: Dictionary containing analysis results
            
        Returns:
            Dictionary containing paths to generated PDF files
        """
        try:
            from .pdf_generator import PDFGenerator
            
            self.logger.info("\n📄 Generating PDF documents...")
            pdf_generator = PDFGenerator(output_dir=self.results_dir)
            
            generated_files = {}
            
            # Generate summary PDF
            console.print("\nGenerating summary PDF...", style="blue")
            summary_pdf = pdf_generator.generate_summary_pdf(summary_info)
            if summary_pdf:
                generated_files['summary_pdf'] = summary_pdf
                console.print(f"✅ Summary PDF generated: {summary_pdf}", style="green")
            
            # Generate transcript PDF
            console.print("\nGenerating transcript PDF...", style="blue")
            transcript_pdf = pdf_generator.generate_transcript_pdf(
                summary_info['transcript'],
                summary_info['video_info']
            )
            if transcript_pdf:
                generated_files['transcript_pdf'] = transcript_pdf
                console.print(f"✅ Transcript PDF generated: {transcript_pdf}", style="green")
            
            # Generate visualization report
            console.print("\nGenerating visualization report...", style="blue")
            vis_pdf = pdf_generator.generate_visualization_report(summary_info)
            if vis_pdf:
                generated_files['visualization_pdf'] = vis_pdf
                console.print(f"✅ Visualization report generated: {vis_pdf}", style="green")
            
            # Add PDF paths to summary info
            summary_info['pdf_documents'] = {
                name: str(path) for name, path in generated_files.items()
            }
            
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error generating PDF documents: {str(e)}")
            return {}

    def _generate_final_summary(self, transcript: str, video_info: Dict) -> Dict:
        """Generate summary using optimized batch processing"""
        try:
            start_time = time.time()
            self.logger.info("Starting summary generation")
            
            # Prepare chunks
            chunks = self.chunk_manager.prepare_chunks(
                transcript, 
                duration=video_info.get('duration', 0)
            )
            
            # Process chunks with progress tracking
            def progress_callback(progress):
                self.logger.info(f"Processing progress: {progress:.2f}%")
            
            results = self.batch_processor.process_chunks(chunks, progress_callback)
            
            # Update chunks with results
            self.chunk_manager.update_chunk_results(results)
            
            # Get summaries in order
            summaries = []
            for chunk in sorted(self.chunk_manager.chunks, key=lambda x: x.chunk_id):
                if chunk.status == ProcessingStatus.COMPLETED and chunk.results:
                    summaries.append(chunk.results['summary'])
            
            # Combine summaries
            final_summary = ' '.join(summaries)
            
            # Analyze content and sentiment
            content_analysis = self.content_processor.analyze_content(transcript)
            sentiment_results = [
                self.content_processor.analyze_sentiment(chunk.text)
                for chunk in self.chunk_manager.chunks
                if chunk.status == ProcessingStatus.COMPLETED
            ]
            
            # Calculate average sentiment
            sentiment_scores = [result['score'] for result in sentiment_results]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
            # Create final result
            result = {
                'summary': final_summary,
                'content_analysis': content_analysis,
                'sentiment_analysis': {
                    'average_score': avg_sentiment,
                    'detailed_scores': sentiment_results
                },
                'video_metadata': video_info,
                'transcript': transcript,
                'processing_stats': {
                    'chunk_count': len(chunks),
                    'processed_chunks': len([c for c in chunks if c.status == ProcessingStatus.COMPLETED]),
                    'failed_chunks': len([c for c in chunks if c.status == ProcessingStatus.FAILED]),
                    'original_length': len(transcript.split()),
                    'summary_length': len(final_summary.split()),
                    'compression_ratio': len(final_summary.split()) / len(transcript.split()),
                    'processing_time': time.time() - start_time
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up GPU memory
            if self.device == 'cuda':
                self.gpu_optimizer.cleanup()
            
            # Clean up models
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