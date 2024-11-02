import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union, NamedTuple
import logging
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from .data_structures import ProcessingChunk, ProcessingStatus, ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class BatchMetadata:
    """Metadata container for batch processing"""
    start_time: torch.Tensor
    end_time: torch.Tensor
    word_count: torch.Tensor

class ChunkDataset(Dataset):
    def __init__(self, chunks: List[ProcessingChunk], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        encodings = self.tokenizer(
            chunk.text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'chunk_id': chunk.chunk_id,
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'text': chunk.text,
            'metadata': {
                'start_time': chunk.metadata.start_time,
                'end_time': chunk.metadata.end_time,
                'word_count': chunk.metadata.word_count
            }
        }

def custom_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function using proper dataclass for metadata"""
    
    # Initialize output dictionary
    output = {
        'chunk_id': [],
        'input_ids': [],
        'attention_mask': [],
        'text': [],
        'metadata_start_time': [],
        'metadata_end_time': [],
        'metadata_word_count': []
    }
    
    # Collect items from batch
    for item in batch:
        output['chunk_id'].append(item['chunk_id'])
        output['input_ids'].append(item['input_ids'])
        output['attention_mask'].append(item['attention_mask'])
        output['text'].append(item['text'])
        output['metadata_start_time'].append(item['metadata']['start_time'])
        output['metadata_end_time'].append(item['metadata']['end_time'])
        output['metadata_word_count'].append(item['metadata']['word_count'])
    
    # Convert to appropriate tensor types
    collated = {
        'chunk_id': torch.tensor(output['chunk_id']),
        'input_ids': torch.stack(output['input_ids']),
        'attention_mask': torch.stack(output['attention_mask']),
        'text': output['text'],
        'metadata': BatchMetadata(
            start_time=torch.tensor(output['metadata_start_time']),
            end_time=torch.tensor(output['metadata_end_time']),
            word_count=torch.tensor(output['metadata_word_count'])
        )
    }
    
    return collated

class BatchProcessor:
    def __init__(
        self,
        model: Union[PreTrainedModel, Pipeline],
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda',
        batch_size: int = 8,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Check if model is a Pipeline or raw model
        self.is_pipeline = isinstance(self.model, Pipeline)
        
        # Only move to device if it's a raw model
        if not self.is_pipeline and hasattr(self.model, 'to'):
            self.model.to(device)
        
    def process_chunks(
        self,
        chunks: List[ProcessingChunk],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        dataset = ChunkDataset(chunks, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0 if self.device == 'cuda' else 2,  # Reduce workers for CUDA
            pin_memory=self.device == 'cuda',
            collate_fn=custom_collate_fn
        )
        
        results = []
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Process batch
                    batch_results = self._process_single_batch(batch)
                    results.extend(batch_results)
                    
                    # Update progress
                    if progress_callback:
                        progress = (batch_idx + 1) / total_batches * 100
                        progress_callback(progress)
                    
                    # Cleanup GPU memory periodically
                    if self.device == 'cuda' and (batch_idx + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
        
        return results
    
    def _process_single_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.is_pipeline:
            return self._process_pipeline_batch(batch)
        else:
            return self._process_model_batch(batch)
            
    def _process_pipeline_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Process using Pipeline
        texts = batch['text']
        summaries = self.model(texts, max_length=150, min_length=40, do_sample=False)
        
        batch_results = []
        for idx, chunk_id in enumerate(batch['chunk_id']):
            result = {
                'chunk_id': chunk_id.item(),
                'text': texts[idx],
                'summary': summaries[idx]['summary_text'],
                'metadata': {
                    'start_time': batch['metadata'].start_time[idx].item(),
                    'end_time': batch['metadata'].end_time[idx].item(),
                    'word_count': batch['metadata'].word_count[idx].item()
                }
            }
            batch_results.append(result)
        
        return batch_results
    
    def _process_model_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Move tensors to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Model inference
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0
        )
        
        # Process outputs
        batch_results = []
        for idx, chunk_id in enumerate(batch['chunk_id']):
            summary = self.tokenizer.decode(outputs[idx], skip_special_tokens=True)
            result = {
                'chunk_id': chunk_id.item(),
                'text': batch['text'][idx],
                'summary': summary,
                'metadata': {
                    'start_time': batch['metadata'].start_time[idx].item(),
                    'end_time': batch['metadata'].end_time[idx].item(),
                    'word_count': batch['metadata'].word_count[idx].item()
                }
            }
            batch_results.append(result)
        
        return batch_results