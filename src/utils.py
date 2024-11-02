import time
import torch
from datetime import timedelta
import re
import logging
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from wordcloud import WordCloud
import textwrap
import shutil
import os
from tqdm import tqdm
import psutil
from rich.console import Console
from rich.logging import RichHandler
import subprocess
from .pipeline import ProcessingStatus, GPUOptimizer

console = Console()
logger = logging.getLogger('Utils')

def create_visualizations(summary_info: Dict, output_dir: Path) -> Optional[Path]:
    """
    Create comprehensive visualizations of the content analysis
    
    Args:
        summary_info: Dictionary containing analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved visualization file
    """
    try:
        # Set the style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # Get processing stats
        processing_stats = summary_info.get('processing_stats', {})
        chunk_status = {
            'completed': processing_stats.get('processed_chunks', 0),
            'failed': processing_stats.get('failed_chunks', 0),
            'total': processing_stats.get('chunk_count', 0)
        }
        
        # 1. Content Type Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        content_types = summary_info.get('content_analysis', {}).get('content_types', {})
        if content_types:
            df = pd.DataFrame(list(content_types.items()), 
                            columns=['Content Type', 'Percentage'])
            sns.barplot(data=df, x='Percentage', y='Content Type', ax=ax1)
            ax1.set_title('Content Type Distribution')
            ax1.set_xlabel('Percentage')
            
            # Add percentage labels
            for i, v in enumerate(df['Percentage']):
                ax1.text(v, i, f'{v:.1f}%', va='center')
        else:
            ax1.text(0.5, 0.5, 'No content type data available', 
                    ha='center', va='center')
        
        # 2. Sentiment Analysis (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        sentiments = summary_info.get('sentiment_analysis', {}).get('detailed_scores', [])
        if sentiments:
            sentiment_scores = [s.get('score', 0) for s in sentiments if isinstance(s, dict)]
            if sentiment_scores:
                sns.histplot(sentiment_scores, bins=20, ax=ax2)
                ax2.set_title('Sentiment Distribution')
                ax2.set_xlabel('Sentiment Score')
                ax2.set_ylabel('Count')
        else:
            ax2.text(0.5, 0.5, 'No sentiment data available', 
                    ha='center', va='center')
        
        # 3. Key Concepts Word Cloud (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        key_concepts = summary_info.get('content_analysis', {}).get('key_concepts', [])
        if key_concepts:
            text = ' '.join(key_concepts)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate(text)
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')
            ax3.set_title('Key Concepts')
        else:
            ax3.text(0.5, 0.5, 'No key concepts available', 
                    ha='center', va='center')
            ax3.axis('off')
        
        # 4. Processing Timeline (Middle Row)
        ax4 = fig.add_subplot(gs[1, :])
        timeline_data = pd.DataFrame({
            'Status': ['Completed', 'Failed', 'Total'],
            'Count': [
                chunk_status['completed'],
                chunk_status['failed'],
                chunk_status['total']
            ]
        })
        sns.barplot(data=timeline_data, x='Status', y='Count', ax=ax4)
        ax4.set_title('Processing Status Distribution')
        
        # Add count labels
        for i, v in enumerate(timeline_data['Count']):
            ax4.text(i, v, str(v), ha='center', va='bottom')
        
        # Add processing time if available
        if 'processing_time' in processing_stats:
            time_text = f"Total Processing Time: {processing_stats['processing_time']:.2f}s"
            ax4.text(0.5, -0.2, time_text, ha='center', transform=ax4.transAxes)
        
        # 5. Sentiment Timeline (Bottom Left)
        ax5 = fig.add_subplot(gs[2, 0])
        if sentiments:
            sentiment_df = pd.DataFrame({
                'Segment': range(len(sentiment_scores)),
                'Score': sentiment_scores
            })
            sns.lineplot(data=sentiment_df, x='Segment', y='Score', ax=ax5)
            ax5.set_title('Sentiment Timeline')
            ax5.set_xlabel('Segment')
            ax5.set_ylabel('Sentiment Score')
        else:
            ax5.text(0.5, 0.5, 'No sentiment timeline available', 
                    ha='center', va='center')
        
        # 6. Topic Distribution (Bottom Middle)
        ax6 = fig.add_subplot(gs[2, 1])
        if key_concepts:
            # Take top 10 concepts
            top_concepts = key_concepts[:10]
            concept_counts = range(len(top_concepts), 0, -1)
            plt.pie(concept_counts, labels=top_concepts, autopct='%1.1f%%')
            ax6.set_title('Topic Distribution (Top 10)')
        else:
            ax6.text(0.5, 0.5, 'No topic distribution available', 
                    ha='center', va='center')
        
        # 7. Processing Performance (Bottom Right)
        ax7 = fig.add_subplot(gs[2, 2])
        performance_metrics = {
            'Compression Ratio': processing_stats.get('compression_ratio', 0) * 100,
            'Success Rate': (chunk_status['completed'] / chunk_status['total'] * 100) 
                          if chunk_status['total'] > 0 else 0,
            'Time/Chunk (s)': (processing_stats.get('processing_time', 0) / 
                             chunk_status['total']) if chunk_status['total'] > 0 else 0
        }
        
        perf_df = pd.DataFrame({
            'Metric': list(performance_metrics.keys()),
            'Value': list(performance_metrics.values())
        })
        sns.barplot(data=perf_df, x='Value', y='Metric', ax=ax7)
        ax7.set_title('Processing Performance')
        
        # Add value labels
        for i, v in enumerate(perf_df['Value']):
            ax7.text(v, i, f'{v:.1f}', va='center')
        
        # Add title and metadata
        video_title = summary_info.get('video_info', {}).get('title', 'Unknown Video')
        plt.suptitle(f"Video Analysis: {video_title}", fontsize=16, y=0.95)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save visualizations
        vis_path = output_dir / 'visualizations'
        vis_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        plot_file = vis_path / f"analysis_{timestamp}.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {plot_file}")
        return plot_file
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        if 'fig' in locals():
            plt.close(fig)
        return None

def setup_logging(log_dir: str = "logs", verbose: bool = False) -> logging.Logger:
    """Configure logging with rich formatting"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create log file name
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"video_summarizer_{timestamp}.log"
    
    # Set logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                         '%Y-%m-%d %H:%M:%S')
    )
    
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(rich_handler)
    
    return root_logger

def create_directory_structure(base_dir: str = ".") -> Dict[str, Path]:
    """Create and return necessary directories"""
    directories = {
        'models': 'models',
        'downloads': 'downloads',
        'results': 'results',
        'logs': 'logs',
        'cache': 'cache',
        'temp': 'temp'
    }
    
    paths = {}
    for name, path in directories.items():
        full_path = Path(base_dir) / path
        full_path.mkdir(exist_ok=True)
        paths[name] = full_path
        
    return paths

def verify_ffmpeg_installation() -> bool:
    """Verify FFmpeg installation and codecs"""
    try:
        # Check FFmpeg installation
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False

        # Check available codecs
        result = subprocess.run(
            ['ffmpeg', '-codecs'],
            capture_output=True,
            text=True
        )
        required_codecs = ['aac', 'libmp3lame', 'libx264']
        for codec in required_codecs:
            if codec not in result.stdout:
                return False
        
        return True
    except Exception:
        return False

def extract_video_id(url: str) -> str:
    """Extract video ID from URL with extended platform support"""
    patterns = {
        'youtube': [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ],
        'vimeo': [
            r'(?:video\/)([0-9]+)',
            r'(?:vimeo\.com\/)([0-9]+)'
        ]
    }
    
    for platform, platform_patterns in patterns.items():
        for pattern in platform_patterns:
            match = re.search(pattern, url)
            if match:
                return f"{platform}_{match.group(1)}"
    
    raise ValueError("Could not extract video ID from URL")

def ensure_sufficient_space(required_mb: int, path: Path) -> bool:
    """Check if there's sufficient disk space"""
    try:
        space_info = get_free_space(path)
        free_mb = space_info.get('free_gb', 0) * 1024  # Convert GB to MB
        return free_mb >= required_mb
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return False

def get_free_space(path: Path) -> Dict[str, float]:
    """Get available disk space information"""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            'total_gb': total / (2**30),  # Convert to GB
            'used_gb': used / (2**30),
            'free_gb': free / (2**30),
            'percent_free': (free / total) * 100
        }
    except Exception as e:
        logger.error(f"Error getting disk space: {str(e)}")
        return {}

def format_time(seconds: Union[int, float]) -> str:
    """Convert seconds to human-readable time format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    return str(timedelta(seconds=int(seconds)))

def cleanup_temp_files(temp_dir: Optional[Path] = None):
    """Clean up temporary files and directories"""
    try:
        if temp_dir is None:
            temp_dir = Path('temp')
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")

def load_json(file_path: Path) -> Dict:
    """Load JSON file with error handling"""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}")
    return {}

def save_json(data: Dict, file_path: Path):
    """Save data to JSON file with error handling"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}")

def add_visualization_to_summary(summary_info: Dict, visualization_path: Path) -> Dict:
    """
    Add visualization metadata to summary
    
    Args:
        summary_info: Dictionary containing summary data
        visualization_path: Path to saved visualization file
        
    Returns:
        Updated summary dictionary
    """
    try:
        if visualization_path and visualization_path.exists():
            summary_info['visualization'] = str(visualization_path)
            
            # Add visualization metadata
            summary_info['visualization_metadata'] = {
                'path': str(visualization_path),
                'filename': visualization_path.name,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file_size': visualization_path.stat().st_size  # in bytes
            }
            
        return summary_info
        
    except Exception as e:
        logger.error(f"Error adding visualization to summary: {str(e)}")
        return summary_info  # Return original summary if error occurs