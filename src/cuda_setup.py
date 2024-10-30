#!/usr/bin/env python3
import torch
import logging
import psutil
import os
from typing import Dict, Optional, Callable, Any, List, Union
from pathlib import Path

logger = logging.getLogger('CUDASetup')

class CudaOutOfMemoryError(Exception):
    """Custom exception for CUDA out of memory errors"""
    pass

def configure_cuda(memory_fraction: float = 0.5) -> Dict[str, Any]:
    """
    Configure CUDA settings with optimized memory management
    
    Args:
        memory_fraction: Fraction of GPU memory to use (default 0.5)
        
    Returns:
        Dictionary containing device configuration
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return {'device': 'cpu'}
        
    try:
        # Clear any existing allocations
        torch.cuda.empty_cache()
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)  # Convert to GB
        
        # Adjust memory fraction based on available GPU memory
        if total_memory < 8:  # For GPUs with less than 8GB
            memory_fraction = 0.3
            logger.info(f"Adjusting memory fraction to {memory_fraction} for {total_memory:.1f}GB GPU")
        elif total_memory < 12:  # For GPUs with less than 12GB
            memory_fraction = 0.4
            logger.info(f"Adjusting memory fraction to {memory_fraction} for {total_memory:.1f}GB GPU")
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable memory efficient options
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
        
        # Get current memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
        
        device_info = {
            'device': 'cuda',
            'name': device_props.name,
            'compute_capability': f"{device_props.major}.{device_props.minor}",
            'total_memory': total_memory,
            'memory_allocated': memory_allocated,
            'memory_reserved': memory_reserved,
            'free_memory': total_memory - memory_allocated,
            'memory_fraction': memory_fraction,
            'cuda_version': torch.version.cuda,
            'gpu_utilization': get_gpu_utilization()
        }
        
        logger.info(f"CUDA configured successfully on {device_info['name']}")
        logger.debug(f"Device info: {device_info}")
        
        return device_info
        
    except Exception as e:
        logger.warning(f"CUDA configuration error: {str(e)}. Falling back to CPU")
        return {'device': 'cpu'}

def safe_cuda_operation(operation: Callable, fallback: Optional[Callable] = None, 
                       max_retries: int = 3, cleanup_threshold: float = 0.8) -> Any:
    """
    Safely execute CUDA operations with advanced error handling
    
    Args:
        operation: Function to execute
        fallback: Optional fallback function for CPU execution
        max_retries: Maximum number of retry attempts
        cleanup_threshold: Memory threshold to trigger cleanup
        
    Returns:
        Result of operation or fallback
    """
    if not torch.cuda.is_available():
        return fallback() if fallback else operation()
    
    for attempt in range(max_retries):
        try:
            # Check memory usage before operation
            memory_info = get_memory_status()
            if memory_info['gpu_memory_used_fraction'] > cleanup_threshold:
                cleanup_cuda_memory()
            
            # Try GPU operation
            result = operation()
            
            # Clean up after successful operation
            torch.cuda.empty_cache()
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"CUDA out of memory (attempt {attempt + 1}/{max_retries})")
                cleanup_cuda_memory()
                
                if attempt == max_retries - 1 and fallback:
                    logger.info("Maximum retries reached, falling back to CPU")
                    return fallback()
                    
                # Try to free more memory
                if attempt == max_retries - 2:
                    force_cuda_cleanup()
                    
            else:
                logger.error(f"CUDA operation error: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error in CUDA operation: {str(e)}")
            if fallback:
                logger.info("Falling back to CPU operation")
                return fallback()
            raise
    
    raise CudaOutOfMemoryError("Failed to execute operation after maximum retries")

def get_optimal_device(min_memory_gb: float = 2.0) -> str:
    """
    Get the optimal device for computation based on available resources
    
    Args:
        min_memory_gb: Minimum required free GPU memory in GB
        
    Returns:
        String indicating device ('cuda' or 'cpu')
    """
    if not torch.cuda.is_available():
        return 'cpu'
        
    try:
        # Test CUDA availability and memory
        torch.cuda.init()
        device_props = torch.cuda.get_device_properties(0)
        
        # Check system resources
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        # Calculate available GPU memory
        total_memory = device_props.total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        free_memory = total_memory - allocated_memory
        
        # Decision logic
        if free_memory < min_memory_gb:
            logger.warning(f"Insufficient GPU memory ({free_memory:.1f}GB free). Using CPU")
            return 'cpu'
        
        if cpu_percent < 70 and ram_percent < 80 and free_memory < min_memory_gb * 2:
            logger.info("System resources available, using CPU to preserve GPU memory")
            return 'cpu'
        
        # Basic CUDA test
        test_tensor = torch.zeros((1, 1), device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        
        logger.info(f"Using GPU with {free_memory:.1f}GB free memory")
        return 'cuda'
        
    except Exception as e:
        logger.warning(f"Error testing CUDA device: {str(e)}. Using CPU")
        return 'cpu'

def cleanup_cuda_memory():
    """Clean up CUDA memory and caches"""
    if torch.cuda.is_available():
        try:
            # Empty CUDA cache
            torch.cuda.empty_cache()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Clear memory allocated by current process
            torch.cuda.memory._dump_snapshot = False
            
            logger.debug("CUDA memory cleaned up")
            
        except Exception as e:
            logger.warning(f"Error cleaning up CUDA memory: {str(e)}")

def force_cuda_cleanup():
    """Force aggressive CUDA memory cleanup"""
    if torch.cuda.is_available():
        try:
            # Empty CUDA cache
            torch.cuda.empty_cache()
            
            # Reset device
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Clear all gradients
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    obj.detach_()
                    del obj
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset device again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Forced CUDA cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during forced CUDA cleanup: {str(e)}")

def get_memory_status() -> Dict[str, float]:
    """
    Get comprehensive memory status for system and GPU
    
    Returns:
        Dictionary containing memory information
    """
    memory_info = {
        'cpu_percent': psutil.cpu_percent(),
        'ram_total': psutil.virtual_memory().total / (1024**3),
        'ram_available': psutil.virtual_memory().available / (1024**3),
        'ram_percent': psutil.virtual_memory().percent,
        'swap_used': psutil.swap_memory().used / (1024**3),
        'gpu_memory_allocated': 0,
        'gpu_memory_reserved': 0,
        'gpu_memory_used_fraction': 0
    }
    
    if torch.cuda.is_available():
        try:
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            
            memory_info.update({
                'gpu_total_memory': total_memory,
                'gpu_memory_allocated': allocated_memory,
                'gpu_memory_reserved': reserved_memory,
                'gpu_memory_free': total_memory - allocated_memory,
                'gpu_memory_used_fraction': allocated_memory / total_memory
            })
        except Exception as e:
            logger.warning(f"Error getting GPU memory status: {str(e)}")
            
    return memory_info

def get_gpu_utilization() -> Optional[float]:
    """Get GPU utilization percentage"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except:
        return None

def set_cuda_device(device_id: int = 0):
    """Set CUDA device with error handling"""
    if torch.cuda.is_available():
        try:
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                logger.info(f"Set CUDA device to: {torch.cuda.get_device_name(device_id)}")
            else:
                logger.warning(f"Invalid device ID {device_id}. Using default device.")
        except Exception as e:
            logger.error(f"Error setting CUDA device: {str(e)}")

def get_available_memory() -> Dict[str, float]:
    """Get available memory for CPU and GPU"""
    memory_info = {
        'cpu_available': psutil.virtual_memory().available / (1024**3),
        'gpu_available': 0
    }
    
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            memory_info['gpu_available'] = (total - allocated) / (1024**3)
        except Exception as e:
            logger.warning(f"Error getting GPU memory info: {str(e)}")
            
    return memory_info

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test CUDA setup
    cuda_info = configure_cuda()
    logger.info(f"CUDA Configuration: {cuda_info}")
    
    # Test memory status
    memory_status = get_memory_status()
    logger.info(f"Memory Status: {memory_status}")
    
    # Test optimal device selection
    device = get_optimal_device()
    logger.info(f"Optimal Device: {device}")