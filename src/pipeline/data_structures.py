from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ChunkMetadata:
    chunk_id: int
    start_time: float
    end_time: float
    word_count: int
    
@dataclass
class ProcessingChunk:
    chunk_id: int
    text: str
    metadata: ChunkMetadata
    status: ProcessingStatus = ProcessingStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
class ChunkManager:
    def __init__(self, chunk_size: int = 300):
        self.chunks: List[ProcessingChunk] = []
        self._current_id = 0
        self.chunk_size = chunk_size
        
    def prepare_chunks(self, text: str, duration: float = 0) -> List[ProcessingChunk]:
        """Split text into chunks with timing information"""
        words = text.split()
        total_words = len(words)
        current_pos = 0
        
        for i in range(0, total_words, self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate timing for video content
            start_time = (current_pos / total_words) * duration
            end_time = ((current_pos + len(chunk_words)) / total_words) * duration
            
            metadata = ChunkMetadata(
                chunk_id=self._current_id,
                start_time=start_time,
                end_time=end_time,
                word_count=len(chunk_words)
            )
            
            chunk = ProcessingChunk(
                chunk_id=self._current_id,
                text=chunk_text,
                metadata=metadata
            )
            
            self.chunks.append(chunk)
            self._current_id += 1
            current_pos += len(chunk_words)
            
        return self.chunks
    
    def get_unprocessed_chunks(self) -> List[ProcessingChunk]:
        """Get list of chunks that haven't been processed"""
        return [chunk for chunk in self.chunks if chunk.status == ProcessingStatus.PENDING]
    
    def get_failed_chunks(self) -> List[ProcessingChunk]:
        """Get list of chunks that failed processing"""
        return [chunk for chunk in self.chunks if chunk.status == ProcessingStatus.FAILED]
    
    def update_chunk_results(self, results: List[Dict[str, Any]]):
        """Update chunks with processing results"""
        for result in results:
            chunk_id = result['chunk_id']
            try:
                chunk = next(c for c in self.chunks if c.chunk_id == chunk_id)
                chunk.results = result
                chunk.status = ProcessingStatus.COMPLETED
            except StopIteration:
                logger.error(f"Could not find chunk with id {chunk_id}")
                continue
            except Exception as e:
                logger.error(f"Error updating chunk {chunk_id}: {str(e)}")
                continue