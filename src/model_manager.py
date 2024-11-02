from pathlib import Path
import sys
import logging
import subprocess
import shutil
import json
import torch
from typing import Dict, Optional, List
from tqdm import tqdm
import spacy
import nltk
from transformers import AutoTokenizer, AutoModel
from .utils import load_json, save_json
from .pipeline import ProcessingStatus, GPUOptimizer

class ModelManager:
    """Manages model downloads, updates, and version control for ML models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager with enhanced GPU support
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.version_file = self.models_dir / 'versions.json'
        self.logger = logging.getLogger('ModelManager')
        
        # Initialize version tracking
        self.load_version_info()
        
        # Setup GPU optimization
        self.gpu_optimizer = GPUOptimizer(memory_fraction=0.5)
        try:
            if self.gpu_optimizer.setup():
                self.device = "cuda"
                gpu_info = self.gpu_optimizer.get_gpu_info()
                self.logger.info(f"Using GPU: {gpu_info['name']}")
                self.logger.info(f"Available Memory: {gpu_info['free_memory']:.2f}GB")
            else:
                self.device = "cpu"
                self.logger.info("Using CPU")
        except Exception as e:
            self.logger.error(f"Error during CUDA setup: {str(e)}")
            self.device = "cpu"
            self.logger.info("Falling back to CPU")

    def load_version_info(self):
        """Load model version information with status tracking"""
        self.versions = load_json(self.version_file)
        if not self.versions:
            self.versions = {
                'spacy': {
                    'version': None, 
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                },
                'nltk': {
                    'version': None, 
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                },
                'whisper': {
                    'version': None, 
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                },
                'transformers': {
                    'version': None, 
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                },
                'sentence_transformer': {
                    'version': None, 
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                }
            }
            self.save_version_info()

    def save_version_info(self):
        """Save current model version information"""
        save_json(self.versions, self.version_file)

    def download_spacy_model(self, force: bool = False) -> str:
        """
        Download and setup spaCy model with status tracking
        
        Args:
            force: Whether to force download even if model exists
        """
        try:
            model_name = "en_core_web_sm"
            self.versions['spacy']['status'] = ProcessingStatus.PROCESSING.value
            
            # Check if we need to download
            if not force:
                try:
                    nlp = spacy.load(model_name)
                    self.logger.info(f"Using existing spaCy model: {model_name}")
                    self.versions['spacy']['status'] = ProcessingStatus.COMPLETED.value
                    return model_name
                except:
                    pass

            # Download model
            self.logger.info(f"Downloading spaCy model: {model_name}")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True
            )
            
            # Verify installation
            nlp = spacy.load(model_name)
            current_version = nlp.meta['version']
            
            # Update version info
            self.versions['spacy'] = {
                'version': current_version,
                'path': model_name,
                'status': ProcessingStatus.COMPLETED.value
            }
            self.save_version_info()
            
            self.logger.info(f"Successfully installed spaCy model {model_name} v{current_version}")
            return model_name

        except Exception as e:
            self.versions['spacy']['status'] = ProcessingStatus.FAILED.value
            self.save_version_info()
            self.logger.error(f"Error downloading spaCy model: {str(e)}")
            raise

    def download_nltk_data(self, force: bool = False) -> Path:
        """
        Download required NLTK data with status tracking
        
        Args:
            force: Whether to force download even if data exists
        """
        nltk_dir = self.models_dir / 'nltk_data'
        nltk_dir.mkdir(parents=True, exist_ok=True)
        
        # Add custom directory to NLTK's search path
        nltk.data.path.insert(0, str(nltk_dir))
        
        required_packages = {
            'punkt': 'tokenizers/punkt',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet'
        }
        
        self.versions['nltk']['status'] = ProcessingStatus.PROCESSING.value
        self.logger.info("\nSetting up NLTK data...")
        
        try:
            for package, data_path in required_packages.items():
                self.logger.info(f"Setting up NLTK package: {package}")
                
                try:
                    if not force:
                        # Check if package exists
                        nltk.data.find(data_path)
                        self.logger.info(f"Package {package} already exists")
                        continue
                except LookupError:
                    pass
                
                # Download the package
                self.logger.info(f"Downloading {package}...")
                success = nltk.download(
                    package,
                    download_dir=str(nltk_dir),
                    quiet=True
                )
                
                if not success:
                    raise Exception(f"Failed to download {package}")
                
                # Special handling for wordnet
                if package == 'wordnet':
                    # Download omw-1.4
                    nltk.download('omw-1.4', download_dir=str(nltk_dir), quiet=True)
                    
                    # Copy from user's home directory if needed
                    home_nltk_dir = Path.home() / 'nltk_data'
                    if home_nltk_dir.exists():
                        home_wordnet = home_nltk_dir / 'corpora' / 'wordnet'
                        custom_wordnet = nltk_dir / 'corpora' / 'wordnet'
                        
                        if home_wordnet.exists() and not custom_wordnet.exists():
                            self.logger.info("Copying WordNet data from home directory...")
                            shutil.copytree(home_wordnet, custom_wordnet)
                
                # Verify the download
                try:
                    nltk.data.find(data_path)
                except LookupError as e:
                    raise Exception(f"Failed to verify {package} after download")
            
            # Update version info
            self.versions['nltk'] = {
                'version': nltk.__version__,
                'path': str(nltk_dir),
                'status': ProcessingStatus.COMPLETED.value
            }
            self.save_version_info()
                
            return nltk_dir
            
        except Exception as e:
            self.versions['nltk']['status'] = ProcessingStatus.FAILED.value
            self.save_version_info()
            self.logger.error(f"Error downloading NLTK data: {str(e)}")
            raise

    def download_transformers_models(self, force: bool = False) -> Dict[str, str]:
        """
        Download and cache required transformer models with GPU optimization
        
        Args:
            force: Whether to force download even if models exist
        """
        try:
            self.versions['transformers']['status'] = ProcessingStatus.PROCESSING.value
            
            models = {
                'summarizer': 'facebook/bart-large-cnn',
                'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment'
            }
            
            cached_models = {}
            
            for model_type, model_name in tqdm(models.items(), 
                                             desc="Downloading transformer models"):
                self.logger.info(f"Setting up {model_type} model: {model_name}")
                
                try:
                    # Clean GPU memory before loading new model
                    if self.device == "cuda":
                        self.gpu_optimizer.cleanup()
                    
                    # Download tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).to(self.device)
                    
                    # Verify the downloads
                    if tokenizer is None or model is None:
                        raise Exception(f"Failed to download {model_type} model")
                    
                    cached_models[model_type] = {
                        'name': model_name,
                        'path': tokenizer.name_or_path,
                        'status': ProcessingStatus.COMPLETED.value
                    }
                    
                    self.logger.info(f"✓ Successfully loaded {model_type} model")
                    
                    # Clean up GPU memory after each model
                    if self.device == "cuda":
                        del model
                        self.gpu_optimizer.cleanup()
                        
                except Exception as e:
                    self.logger.error(f"Error loading {model_type} model: {str(e)}")
                    cached_models[model_type] = {
                        'name': model_name,
                        'path': None,
                        'status': ProcessingStatus.FAILED.value,
                        'error': str(e)
                    }
            
            # Update version info
            self.versions['transformers'] = {
                'version': None,
                'models': cached_models,
                'status': ProcessingStatus.COMPLETED.value
            }
            self.save_version_info()
            
            return cached_models

        except Exception as e:
            self.versions['transformers']['status'] = ProcessingStatus.FAILED.value
            self.save_version_info()
            self.logger.error(f"Error downloading transformer models: {str(e)}")
            raise

    def setup_all_models(self, force: bool = False) -> Dict[str, str]:
        """
        Download and setup all required models with status tracking
        
        Args:
            force: Whether to force download even if models exist
        """
        model_paths = {}

        try:
            # Setup spaCy
            self.logger.info("\nSetting up spaCy model...")
            model_paths['spacy'] = self.download_spacy_model(force)
            
            # Setup NLTK
            self.logger.info("\nSetting up NLTK data...")
            model_paths['nltk'] = self.download_nltk_data(force)
            
            # Setup transformers
            self.logger.info("\nSetting up transformer models...")
            model_paths.update(self.download_transformers_models(force))

            self.logger.info("\n✓ All models setup successfully")
            return model_paths

        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise

    def check_updates_available(self) -> Dict[str, bool]:
        """Check for available updates for all models"""
        updates = {}

        try:
            # Check spaCy
            if 'spacy' in self.versions:
                current_version = self.versions['spacy'].get('version')
                if current_version:
                    meta = spacy.cli.info()
                    if meta.get('spacy_version') != current_version:
                        updates['spacy'] = True

            # Check NLTK
            if 'nltk' in self.versions:
                current_nltk = self.versions['nltk'].get('version')
                if current_nltk and nltk.__version__ != current_nltk:
                    updates['nltk'] = True

            return updates

        except Exception as e:
            self.logger.error(f"Error checking for updates: {str(e)}")
            return {}

    def cleanup_models(self, models: Optional[List[str]] = None):
        """
        Clean up downloaded models with GPU optimization
        
        Args:
            models: Optional list of models to clean up. If None, clean all.
        """
        try:
            if models is None:
                models = list(self.versions.keys())

            for model in models:
                if model in self.versions and self.versions[model].get('path'):
                    path = Path(self.versions[model]['path'])
                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        self.logger.info(f"Cleaned up {model} model")

            # Reset version info for cleaned models
            for model in models:
                self.versions[model] = {
                    'version': None,
                    'path': None,
                    'status': ProcessingStatus.PENDING.value
                }
            self.save_version_info()

            # Clean up GPU memory
            if hasattr(self, 'gpu_optimizer'):
                self.gpu_optimizer.cleanup()

        except Exception as e:
            self.logger.error(f"Error cleaning up models: {str(e)}")
            raise

    def verify_model_setup(self) -> bool:
        """Verify all models are properly setup"""
        try:
            # Verify spaCy
            self.logger.info("Verifying spaCy model...")
            model_name = "en_core_web_sm"
            nlp = spacy.load(model_name)
            self.logger.info("✓ spaCy model verified")

            # Verify NLTK
            self.logger.info("Verifying NLTK data...")
            required_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']
            for package in required_packages:
                path = f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}'
                nltk.data.find(path)
            self.logger.info("✓ NLTK data verified")

            # Verify transformers
            self.logger.info("Verifying transformer models...")
            transformer_models = self.versions.get('transformers', {}).get('models', {})
            for model_info in transformer_models.values():
                if model_info.get('status') == ProcessingStatus.COMPLETED.value:
                    tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                    model = AutoModel.from_pretrained(model_info['name']).to(self.device)
                    del model  # Clean up GPU memory
                    if self.device == "cuda":
                        self.gpu_optimizer.cleanup()
            
            self.logger.info("✓ Transformer models verified")
            return True

        except Exception as e:
            self.logger.error(f"Model verification failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the model manager
    logging.basicConfig(level=logging.INFO)
    try:
        manager = ModelManager()
        print(f"Initialized with device: {manager.device}")
        
        # Test GPU info if available
        if manager.device == "cuda":
            gpu_info = manager.gpu_optimizer.get_gpu_info()
            print("\nGPU Information:")
            print(f"Name: {gpu_info['name']}")
            print(f"Total Memory: {gpu_info['total_memory']:.2f}GB")
            print(f"Free Memory: {gpu_info['free_memory']:.2f}GB")
            print(f"Compute Capability: {gpu_info['compute_capability']}")
        
        # Verify existing models
        print("\nVerifying model setup...")
        if manager.verify_model_setup():
            print("✅ All models verified successfully")
        else:
            print("⚠️ Model verification failed")
            
        # Check for updates
        print("\nChecking for updates...")
        updates = manager.check_updates_available()
        if updates:
            print("Updates available for:", ", ".join(updates.keys()))
        else:
            print("All models are up to date")
            
        # Display model status
        print("\nModel Status:")
        for model_name, info in manager.versions.items():
            status = info.get('status', ProcessingStatus.PENDING.value)
            version = info.get('version', 'Unknown')
            print(f"{model_name}: Status={status}, Version={version}")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        logging.error(f"Test failed: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        if 'manager' in locals():
            manager.cleanup_models()