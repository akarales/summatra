# SUMMATRA - Smart Universal Media Mining And Transcription Recognition Assistant

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CUDA Support](https://img.shields.io/badge/CUDA-11.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FFmpeg Required](https://img.shields.io/badge/FFmpeg-required-red.svg)](https://ffmpeg.org/)

## 🎯 Overview

SUMMATRA is a powerful, GPU-accelerated video content analysis tool that combines state-of-the-art machine learning models to provide comprehensive video understanding. It automatically transcribes, summarizes, and analyzes video content while generating detailed visualizations and insights.

### 🚀 Key Features

- **Advanced Transcription**: Utilizes OpenAI's Whisper model for accurate speech-to-text conversion
- **Intelligent Summarization**: Implements BART-based text summarization for concise content overview
- **Sentiment Analysis**: Analyzes emotional tone and content sentiment throughout the video
- **Content Classification**: Automatically categorizes content types and extracts key concepts
- **Visual Analytics**: Generates comprehensive visualizations of video analysis results
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Memory Efficient**: Smart memory management for processing long videos
- **Interactive Mode**: User-friendly command-line interface for easy interaction

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- CUDA Toolkit 11.x (for GPU support)
- FFmpeg with required codecs

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/summatra.git
cd summatra

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run initial setup
python setup.py install
```

### Optional: GPU Support

```bash
# Install CUDA toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvidia-smi
```

## 💻 Usage

### Command Line Interface

```bash
# Process a single video
python main.py --url "https://youtube.com/watch?v=example"

# Enable GPU acceleration with safe memory usage
python main.py --url "https://youtube.com/watch?v=example" --safe-mode

# Process video and clean up temporary files
python main.py --url "https://youtube.com/watch?v=example" --cleanup

# Force CPU usage
python main.py --url "https://youtube.com/watch?v=example" --cpu-only
```

### Interactive Mode

```bash
# Start interactive mode
python main.py

# Commands in interactive mode:
# - Enter URL to process video
# - 'h' for help
# - 'q' to quit
```

## 📊 Output Examples

### Transcription

```plaintext
[Timestamp] Speaker: "Transcribed content of the video..."
```

### Summary Analysis

```json
{
  "summary": "Concise overview of video content...",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "sentiment": {
    "overall": "positive",
    "confidence": 0.85
  }
}
```

### Visualizations

- Sentiment Timeline
- Content Type Distribution
- Key Concepts Word Cloud
- Processing Performance Metrics

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Create .env file
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
WHISPER_MODEL_SIZE=base
```

### Custom Model Configuration

```python
# config.yaml
models:
  whisper:
    size: "base"
    language: "en"
  summarizer:
    model: "facebook/bart-base"
    max_length: 150
    min_length: 40
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for transcription capabilities
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for text processing
- [FFmpeg](https://ffmpeg.org/) for media processing
- All other open-source contributors

## 📬 Contact

Alex Karales - [@X.com](https://x.com/alex_karales) - karales@gmail.com

Project Link: [https://github.com/akarales/summatra](https://github.com/akarales/summatra)

## 📈 Project Status

SUMMATRA is under active development. Check our [Project Board](https://github.com/akarales/summatra/projects) for planned features and current progress.

---
