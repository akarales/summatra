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
from typing import Dict, Optional, Union
from wordcloud import WordCloud
import textwrap
import shutil
import os
from tqdm import tqdm
import psutil
from rich.console import Console
from rich.logging import RichHandler
from typing import List
import subprocess

console = Console()

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
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Content Type Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        plot_content_types(summary_info, ax1)
        
        # 2. Sentiment Analysis (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        plot_sentiment_distribution(summary_info, ax2)
        
        # 3. Key Concepts Word Cloud (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        create_word_cloud(summary_info, ax3)
        
        # 4. Content Timeline (Middle Row)
        ax4 = fig.add_subplot(gs[1, :])
        plot_content_timeline(summary_info, ax4)
        
        # 5. Sentiment Timeline (Bottom Left)
        ax5 = fig.add_subplot(gs[2, 0])
        plot_sentiment_timeline(summary_info, ax5)
        
        # 6. Topic Distribution (Bottom Middle)
        ax6 = fig.add_subplot(gs[2, 1])
        plot_topic_distribution(summary_info, ax6)
        
        # 7. Processing Stats (Bottom Right)
        ax7 = fig.add_subplot(gs[2, 2])
        plot_processing_stats(summary_info, ax7)
        
        # Add title and metadata
        plt.suptitle(f"Video Analysis: {summary_info['video_metadata']['title']}", 
                    fontsize=16, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualizations
        vis_path = output_dir / 'visualizations'
        vis_path.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        plot_file = vis_path / f"analysis_{timestamp}.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

def plot_content_types(summary_info: Dict, ax):
    """Plot content type distribution"""
    content_types = summary_info['content_analysis']['content_types']
    if content_types:
        df = pd.DataFrame(list(content_types.items()), 
                         columns=['Content Type', 'Count'])
        
        sns.barplot(data=df, x='Count', y='Content Type', ax=ax)
        ax.set_title('Content Type Distribution')
        ax.set_xlabel('Count')
        ax.set_ylabel('Content Type')

def plot_sentiment_distribution(summary_info: Dict, ax):
    """Plot sentiment distribution"""
    sentiments = summary_info['sentiment_analysis']
    if sentiments:
        df = pd.DataFrame(sentiments)
        
        # Create violin plot for sentiment scores
        sns.violinplot(data=df, y='score', ax=ax)
        ax.set_title('Sentiment Distribution')
        ax.set_ylabel('Sentiment Score')

def create_word_cloud(summary_info: Dict, ax):
    """Create word cloud from key concepts"""
    key_concepts = summary_info['content_analysis']['key_concepts']
    if key_concepts:
        # Create frequency dict
        text = ' '.join(key_concepts)
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white',
                            colormap='viridis').generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Key Concepts')

def plot_content_timeline(summary_info: Dict, ax):
    """Plot content analysis timeline"""
    try:
        # Create timeline data
        duration = summary_info['video_metadata']['duration']
        segments = np.linspace(0, duration, 20)  # 20 time segments
        
        # Create mock data for visualization
        df = pd.DataFrame({
            'Time': segments,
            'Content': np.random.choice(list(summary_info['content_analysis']['content_types'].keys()), 
                                     size=len(segments))
        })
        
        # Create timeline plot
        sns.scatterplot(data=df, x='Time', y='Content', ax=ax, s=100)
        ax.set_title('Content Timeline')
        ax.set_xlabel('Time (seconds)')
        
    except Exception as e:
        ax.text(0.5, 0.5, 'Timeline data not available', 
                ha='center', va='center')

def plot_sentiment_timeline(summary_info: Dict, ax):
    """Plot sentiment over time"""
    sentiments = summary_info['sentiment_analysis']
    if sentiments:
        df = pd.DataFrame(sentiments)
        df['index'] = range(len(df))
        
        sns.lineplot(data=df, x='index', y='score', ax=ax)
        ax.set_title('Sentiment Timeline')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Sentiment Score')

def plot_topic_distribution(summary_info: Dict, ax):
    """Plot topic distribution"""
    key_concepts = summary_info['content_analysis']['key_concepts']
    if key_concepts:
        # Take top 10 concepts
        concepts = key_concepts[:10]
        frequencies = range(len(concepts), 0, -1)
        
        # Create pie chart
        ax.pie(frequencies, labels=concepts, autopct='%1.1f%%')
        ax.set_title('Topic Distribution')

def plot_processing_stats(summary_info: Dict, ax):
    """Plot processing statistics"""
    stats = {
        'Duration': summary_info['video_metadata']['duration'],
        'Processing Time': summary_info['processing_time'],
        'Segments': len(summary_info['sentiment_analysis'])
    }
    
    # Create bar plot
    colors = sns.color_palette("husl", len(stats))
    bars = ax.bar(range(len(stats)), list(stats.values()), color=colors)
    
    # Customize the plot
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([textwrap.fill(label, 10) for label in stats.keys()])
    ax.set_title('Processing Statistics')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

def add_visualization_to_summary(summary_info: Dict, visualization_path: Path) -> Dict:
    """Add visualization metadata to summary"""
    if visualization_path and visualization_path.exists():
        summary_info['visualization'] = str(visualization_path)
    return summary_info

def setup_logging(log_dir: str = "logs", verbose: bool = False) -> logging.Logger:
    """Configure logging with formatting and file output"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create formatted log file name
    from datetime import datetime
    log_file = log_dir / f"video_summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set logging level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup handlers with rich formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(rich_handler)
    
    return root_logger

def check_gpu() -> Dict[str, Union[bool, str, int, float]]:
    """Check GPU availability and capabilities"""
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_name': None,
        'cuda_version': None,
        'device_count': 0,
        'memory_allocated': 0,
        'memory_cached': 0
    }
    
    if gpu_info['available']:
        gpu_info.update({
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_cached': torch.cuda.memory_reserved(0)
        })
        
    return gpu_info

def format_time(seconds: Union[int, float]) -> str:
    """Convert seconds to human-readable time format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    return str(timedelta(seconds=int(seconds)))

def extract_video_id(url: str) -> str:
    """Extract video ID from various video platform URLs"""
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

def create_directory_structure(base_dir: str = ".") -> Dict[str, Path]:
    """Create and return all necessary directories"""
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

def create_visualizations(summary_info: Dict, output_dir: Path) -> Optional[Path]:
    """Create and save visualization of the content analysis"""
    try:
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Content Type Distribution
        content_types = summary_info['content_analysis']['content_types']
        content_df = pd.DataFrame(list(content_types.items()),
                                columns=['Content Type', 'Count'])
        
        sns.barplot(data=content_df,
                   x='Content Type',
                   y='Count',
                   palette='viridis',
                   ax=axes[0, 0])
        axes[0, 0].set_title('Content Type Distribution')
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        
        # 2. Key Concepts Cloud
        if 'key_concepts' in summary_info['content_analysis']:
            key_concepts = summary_info['content_analysis']['key_concepts'][:10]
            concepts_df = pd.DataFrame({
                'Concept': key_concepts,
                'Index': range(len(key_concepts))
            })
            
            sns.barplot(data=concepts_df,
                       x='Index',
                       y='Concept',
                       palette='rocket',
                       ax=axes[0, 1])
            axes[0, 1].set_title('Top Key Concepts')
            axes[0, 1].set(xlabel='Frequency Rank')
            
        # 3. Processing Time Analysis
        if 'processing_times' in summary_info:
            times_df = pd.DataFrame(list(summary_info['processing_times'].items()),
                                  columns=['Step', 'Time'])
            
            sns.barplot(data=times_df,
                       x='Time',
                       y='Step',
                       palette='mako',
                       ax=axes[1, 0])
            axes[1, 0].set_title('Processing Time by Step')
            
        # 4. Sentiment Distribution
        if 'sentiment_analysis' in summary_info:
            sentiment_df = pd.DataFrame(summary_info['sentiment_analysis'])
            
            sns.histplot(data=sentiment_df,
                        x='score',
                        hue='label',
                        multiple="stack",
                        palette='flare',
                        ax=axes[1, 1])
            axes[1, 1].set_title('Sentiment Distribution')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualizations
        vis_path = output_dir / 'visualizations'
        vis_path.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = vis_path / f"analysis_{timestamp}.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
        return None

def cleanup_temp_files(temp_dir: Optional[Path] = None):
    """Clean up temporary files and directories"""
    try:
        if temp_dir is None:
            temp_dir = Path('temp')
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
    
    except Exception as e:
        logging.error(f"Error cleaning up temporary files: {e}")

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
        logging.error(f"Error getting disk space: {e}")
        return {}

def ensure_sufficient_space(required_mb: int, path: Path) -> bool:
    """Check if there's sufficient disk space available"""
    try:
        space_info = get_free_space(path)
        free_mb = space_info.get('free_gb', 0) * 1024  # Convert GB to MB
        return free_mb >= required_mb
    except Exception as e:
        logging.error(f"Error checking disk space: {e}")
        return False

def copy_with_progress(src: Path, dst: Path):
    """Copy file with progress bar"""
    try:
        total_size = os.path.getsize(src)
        with tqdm(total=total_size, 
                 unit='B', 
                 unit_scale=True, 
                 desc=f"Copying {src.name}") as pbar:
            shutil.copy2(src, dst, follow_symlinks=True)
            pbar.update(total_size)
    except Exception as e:
        logging.error(f"Error copying file: {e}")
        raise

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

def print_summary_stats(summary_info: Dict):
    """Print summary statistics"""
    console.print("\n" + "="*50, style="bold blue")
    console.print("📊 Summary Statistics", style="bold green")
    console.print("="*50, style="bold blue")

    console.print(f"\n⏱️ Processing Time: {format_time(summary_info['processing_time'])}")
    console.print(f"🖥️ Device Used: {summary_info['device_used']}")

    if 'content_analysis' in summary_info:
        console.print("\n📋 Content Analysis:", style="bold yellow")
        console.print("-" * 30)

        content_types = summary_info['content_analysis']['content_types']
        for content_type, count in content_types.items():
            console.print(f"- {content_type.title()}: {count} sections")

        console.print("\n🔑 Key Concepts:", style="bold yellow")
        for concept in summary_info['content_analysis']['key_concepts'][:10]:
            console.print(f"- {concept}")

    console.print("\n" + "="*50, style="bold blue")