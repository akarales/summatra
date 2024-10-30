#!/usr/bin/env python3
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
from .cuda_setup import configure_cuda, safe_cuda_operation, get_optimal_device

class ModelManager:
    """Manages model downloads, updates, and version control for ML models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager with CUDA support
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.version_file = self.models_dir / 'versions.json'
        self.logger = logging.getLogger('ModelManager')
        
        # Initialize version tracking
        self.load_version_info()
        
        # Setup CUDA with lower memory fraction since this is just for model loading
        try:
            device_config = configure_cuda(memory_fraction=0.5)
            self.device = device_config['device']
            
            if self.device == "cuda":
                self.logger.info(f"Using GPU: {device_config.get('name', 'Unknown')}")
                self.logger.info(f"Memory Allocated: {device_config.get('memory_allocated', 'Unknown')}")
            else:
                self.logger.info("Using CPU")
                
        except Exception as e:
            self.logger.error(f"Error during CUDA setup: {str(e)}")
            self.device = "cpu"
            self.logger.info("Falling back to CPU")

    def load_version_info(self):
        """Load model version information from file"""
        self.versions = load_json(self.version_file)
        if not self.versions:
            self.versions = {
                'spacy': {'version': None, 'path': None},
                'nltk': {'version': None, 'path': None},
                'whisper': {'version': None, 'path': None},
                'transformers': {'version': None, 'path': None},
                'sentence_transformer': {'version': None, 'path': None}
            }
            self.save_version_info()

    def save_version_info(self):
        """Save current model version information to file"""
        save_json(self.versions, self.version_file)

    def safe_cuda_operation(self, operation: callable, fallback: Optional[callable] = None) -> any:
        """
        Safely execute CUDA operations with fallback
        
        Args:
            operation: Function to execute
            fallback: Optional fallback function for CPU execution
        """
        return safe_cuda_operation(operation, fallback)

    def download_spacy_model(self, force: bool = False) -> str:
        """
        Download and setup spaCy model
        
        Args:
            force: Whether to force download even if model exists
        """
        try:
            model_name = "en_core_web_sm"
            
            # Check if we need to download
            if not force:
                try:
                    nlp = spacy.load(model_name)
                    self.logger.info(f"Using existing spaCy model: {model_name}")
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
                'path': model_name
            }
            self.save_version_info()
            
            self.logger.info(f"Successfully installed spaCy model {model_name} v{current_version}")
            return model_name

        except Exception as e:
            self.logger.error(f"Error downloading spaCy model: {str(e)}")
            raise

    def download_nltk_data(self, force: bool = False) -> Path:
        """
        Download required NLTK data packages
        
        Args:
            force: Whether to force download even if data exists
        """
        nltk_dir = self.models_dir / 'nltk_data'
        nltk_dir.mkdir(parents=True, exist_ok=True)
        
        # Add our custom directory to NLTK's search path
        nltk.data.path.insert(0, str(nltk_dir))
        
        required_packages = {
            'punkt': 'tokenizers/punkt',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet'
        }
        
        self.logger.info("\nSetting up NLTK data...")
        
        try:
            for package, data_path in required_packages.items():
                self.logger.info(f"Setting up NLTK package: {package}")
                
                try:
                    if not force:
                        # Check if package already exists
                        nltk.data.find(data_path)
                        self.logger.info(f"Package {package} already exists")
                        continue
                except LookupError:
                    pass
                
                # Download the package to our custom directory
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
                    # Also download omw-1.4 which is needed by wordnet
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
                'path': str(nltk_dir)
            }
            self.save_version_info()
                
            return nltk_dir
            
        except Exception as e:
            self.logger.error(f"Error downloading NLTK data: {str(e)}")
            raise

    def download_transformers_models(self, force: bool = False) -> Dict[str, str]:
        """
        Download and cache required transformer models
        
        Args:
            force: Whether to force download even if models exist
        """
        try:
            models = {
                'summarizer': 'facebook/bart-large-cnn',
                'sentiment': 'nlptown/bert-base-multilingual-uncased-sentiment'
            }
            
            cached_models = {}
            
            for model_type, model_name in tqdm(models.items(), 
                                             desc="Downloading transformer models"):
                self.logger.info(f"Setting up {model_type} model: {model_name}")
                
                def download_model():
                    # Download tokenizer and model to device
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).to(self.device)
                    return tokenizer, model
                
                # Use safe CUDA operation for model loading
                tokenizer, model = self.safe_cuda_operation(
                    download_model,
                    lambda: (
                        AutoTokenizer.from_pretrained(model_name),
                        AutoModel.from_pretrained(model_name).to("cpu")
                    )
                )
                
                # Verify the downloads
                if tokenizer is None or model is None:
                    raise Exception(f"Failed to download {model_type} model")
                
                cached_models[model_type] = {
                    'name': model_name,
                    'path': tokenizer.name_or_path
                }
                
                self.logger.info(f"✓ Successfully loaded {model_type} model")
                
                # Clean up GPU memory after each model
                if self.device == "cuda":
                    del model
                    torch.cuda.empty_cache()
            
            # Update version info
            self.versions['transformers'] = {
                'version': None,  # Could get from transformers.__version__
                'models': cached_models
            }
            self.save_version_info()
            
            return cached_models

        except Exception as e:
            self.logger.error(f"Error downloading transformer models: {str(e)}")
            raise

    def setup_all_models(self, force: bool = False) -> Dict[str, str]:
        """
        Download and setup all required models
        
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
        """
        Check for available updates for all models
        
        Returns:
            Dictionary of models with available updates
        """
        updates = {}

        try:
            # Check spaCy
            if 'spacy' in self.versions:
                current_version = self.versions['spacy']['version']
                if current_version:
                    meta = spacy.cli.info()
                    if meta.get('spacy_version') != current_version:
                        updates['spacy'] = True

            # Check NLTK
            if 'nltk' in self.versions:
                current_nltk = self.versions['nltk']['version']
                if current_nltk and nltk.__version__ != current_nltk:
                    updates['nltk'] = True

            return updates

        except Exception as e:
            self.logger.error(f"Error checking for updates: {str(e)}")
            return {}

    def cleanup_models(self, models: Optional[List[str]] = None):
        """
        Clean up downloaded models to free space
        
        Args:
            models: Optional list of models to clean up. If None, clean all.
        """
        try:
            if models is None:
                models = list(self.versions.keys())

            for model in models:
                if model in self.versions and self.versions[model]['path']:
                    path = Path(self.versions[model]['path'])
                    if path.exists():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        self.logger.info(f"Cleaned up {model} model")

            # Reset version info for cleaned models
            for model in models:
                self.versions[model] = {'version': None, 'path': None}
            self.save_version_info()

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Error cleaning up models: {str(e)}")
            raise

    def verify_model_setup(self) -> bool:
        """
        Verify all models are properly setup
        
        Returns:
            Boolean indicating whether all models are properly setup
        """
        try:
            # Verify spaCy
            self.logger.info("Verifying spaCy model...")
            model_name = "en_core_web_sm"
            nlp = spacy.load(model_name)
            self.logger.info("✓ spaCy model verified")

            # Verify NLTK
            self.logger.info("Verifying NLTK data...")
            for package in ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
            self.logger.info("✓ NLTK data verified")

            # Verify transformers
            self.logger.info("Verifying transformer models...")
            def verify_transformers():
                for model_info in self.versions['transformers'].get('models', {}).values():
                    tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                    model = AutoModel.from_pretrained(model_info['name']).to(self.device)
                    del model  # Clean up GPU memory
                return True
            
            success = self.safe_cuda_operation(
                verify_transformers,
                lambda: verify_transformers()  # Fall back to CPU verification
            )
            
            if success:
                self.logger.info("✓ Transformer models verified")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Model verification failed: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the model manager
    logging.basicConfig(level=logging.INFO)
    try:
        manager = ModelManager()
        print(f"Initialized with device: {manager.device}")
        manager.verify_model_setup()
    except Exception as e:
        print(f"Test failed: {str(e)}")