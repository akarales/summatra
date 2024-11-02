import torch
import gc
import logging
from typing import Dict, Any
import psutil

logger = logging.getLogger(__name__)

class GPUOptimizer:
    def __init__(self, device: str = 'cuda', memory_fraction: float = 0.8):
        self.device = device
        self.memory_fraction = memory_fraction
        
    def setup(self) -> bool:
        """Configure GPU settings"""
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                # Configure CUDA settings
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                torch.backends.cudnn.benchmark = True
                
                # Clear existing cache
                self.cleanup()
                
                # Log GPU info
                gpu_info = self.get_gpu_info()
                logger.info(f"GPU Setup - Device: {gpu_info['name']}, "
                          f"Memory: {gpu_info['free_memory']:.2f}GB free")
                
                return True
            except Exception as e:
                logger.error(f"Error setting up GPU: {str(e)}")
                return False
        return False
    
    def cleanup(self):
        """Perform thorough GPU memory cleanup"""
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Get current memory usage
                gpu_info = self.get_gpu_info()
                logger.debug(f"GPU Memory - Free: {gpu_info['free_memory']:.2f}GB, "
                           f"Used: {gpu_info['used_memory']:.2f}GB")
                
            except Exception as e:
                logger.error(f"Error during GPU cleanup: {str(e)}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU status"""
        if self.device != 'cuda' or not torch.cuda.is_available():
            return {}
            
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            total_memory = props.total_memory / 1e9  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated(device) / 1e9
            reserved_memory = torch.cuda.memory_reserved(device) / 1e9
            
            return {
                'name': props.name,
                'total_memory': total_memory,
                'used_memory': allocated_memory,
                'reserved_memory': reserved_memory,
                'free_memory': total_memory - allocated_memory,
                'compute_capability': f"{props.major}.{props.minor}"
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {str(e)}")
            return {}
    
    def get_optimal_batch_size(self, sample_input_size: int) -> int:
        """Calculate optimal batch size based on available memory"""
        if self.device != 'cuda' or not torch.cuda.is_available():
            return 8  # Default CPU batch size
            
        try:
            # Get GPU memory info
            gpu_info = self.get_gpu_info()
            available_memory = gpu_info['free_memory'] * self.memory_fraction
            
            # Estimate memory per sample
            test_batch = torch.zeros((1, sample_input_size), device=self.device)
            memory_per_sample = torch.cuda.memory_allocated() / 1e9
            del test_batch
            torch.cuda.empty_cache()
            
            # Calculate optimal batch size (use 70% of available memory)
            optimal_size = int((available_memory * 0.7) / memory_per_sample)
            
            # Clamp between reasonable values
            return max(1, min(optimal_size, 32))
            
        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {str(e)}")
            return 8  # Default fallback