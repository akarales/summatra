# SUMMATRA - Smart Universal Media Mining And Transcription Recognition Assistant

![SUMMATRA Logo](/api/placeholder/800/400)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CUDA Support](https://img.shields.io/badge/CUDA-11.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FFmpeg Required](https://img.shields.io/badge/FFmpeg-required-red.svg)](https://ffmpeg.org/)
[![Version](https://img.shields.io/badge/version-0.3--alpha-orange.svg)](https://github.com/akarales/summatra/releases)
[![Website](https://img.shields.io/badge/website-karales.com-blue.svg)](https://karales.com)
[![Twitter](https://img.shields.io/badge/X-alex__karales-black.svg)](https://x.com/alex_karales)

## 🎯 Overview

SUMMATRA is a powerful, GPU-accelerated video content analysis tool that combines state-of-the-art machine learning models to provide comprehensive video understanding. Version 0.3-alpha introduces advanced PDF report generation, offering detailed insights and visualizations of your video content.

## 📜 Version History

### V0.3-alpha (Current)

- **Major Features**:
  - PDF Report Generation with detailed analytics
  - Interactive visualizations and charts
  - Memory optimization improvements
  - Enhanced error handling and recovery
  - Support for conda environments

- **Improvements**:
  - Advanced GPU memory management
  - Expanded documentation
  - More comprehensive configuration options

### V0.2-alpha

- **Major Features**:
  - Introduction of sentiment analysis
  - Content classification system
  - Visual analytics generation
  - Multi-threading support

- **Improvements**:
  - Enhanced memory efficiency
  - Better language support
  - Improved error handling
  - Command-line interface enhancements

### V0.1-alpha

- **Initial Release**:
  - Basic transcription using Whisper
  - Simple summarization
  - GPU acceleration support
  - Basic CLI interface

- **Core Features**:
  - Video download capabilities
  - Audio extraction and processing
  - Basic error handling
  - Elementary memory management

## 🔍 Demo

![SUMMATRA Demo](/api/placeholder/600/300)

### 🚀 Key Features

#### New in V0.3-alpha

- **PDF Report Generation**: Professional PDF reports with:
  - Summary analysis
  - Full transcripts
  - Sentiment analysis graphs
  - Content type distribution charts
  - Key concepts visualization
  - Processing statistics

- **Interactive Visualizations**: Dynamic charts and graphs showing:
  - Content type distribution
  - Sentiment timeline
  - Topic analysis
  - Processing metrics

- **Memory-Optimized Processing**: Improved GPU memory management

- **Enhanced Error Handling**: Robust recovery from processing failures

#### Core Features

- **Advanced Transcription**: Utilizes OpenAI's Whisper model for accurate speech-to-text conversion
- **Intelligent Summarization**: Implements BART-based text summarization for concise content overview
- **Sentiment Analysis**: Analyzes emotional tone and content sentiment throughout the video
- **Content Classification**: Automatically categorizes content types and extracts key concepts
- **Visual Analytics**: Generates comprehensive visualizations of video analysis results
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Memory Efficient**: Smart memory management for processing long videos
- **Interactive Mode**: User-friendly command-line interface for easy interaction

## 🏆 Features in Detail

### Transcription Engine

- Powered by OpenAI's Whisper ASR system
- Supports multiple languages
- Optimized for various accents and speech patterns
- Real-time transcription capabilities

### Analysis Capabilities

- Topic extraction and classification
- Entity recognition
- Sentiment analysis with temporal tracking
- Key point summarization

### Performance

- GPU-accelerated processing with CUDA support
- Multi-threading support
- Memory-efficient chunking
- Automatic resource optimization

## 🚦 System Requirements

### Minimum Requirements

- Python 3.9+
- 8GB RAM
- 2GB GPU Memory (for GPU mode)
- 10GB Disk Space
- FFmpeg with required codecs

### Recommended

- 16GB RAM
- 6GB+ GPU Memory (NVIDIA)
- 20GB SSD Storage
- CUDA-compatible GPU
- Ubuntu 20.04 or newer

## 🛠️ Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n summatra python=3.9
conda activate summatra

# Clone the repository
git clone https://github.com/akarales/summatra.git
cd summatra

# Install PyTorch with CUDA support (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
conda install -c conda-forge ffmpeg
conda install -c conda-forge transformers
conda install pandas numpy matplotlib seaborn

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Using venv (Alternative)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: GPU Support

```bash
# Install CUDA toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit

# Verify CUDA installation
nvidia-smi
```

### Environment Setup Verification

```bash
# Verify the installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import whisper; print('Whisper available')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Test FFmpeg installation
ffmpeg -version
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

# Generate PDF report
python main.py --url "https://youtube.com/watch?v=example" --generate-pdf

# Generate only transcript PDF
python main.py --url "https://youtube.com/watch?v=example" --transcript-pdf-only
```

### Interactive Mode

```bash
# Start interactive mode
python main.py

# Commands in interactive mode:
# - Enter URL to process video
# - 'pdf' to toggle PDF generation
# - 'h' for help
# - 'q' to quit
```

## 📈 Project Structure

```plaintext
summatra/
├── src/
│   ├── content_processor.py
│   ├── cuda_setup.py
│   ├── model_manager.py
│   ├── pdf_generator.py      # New PDF generation module
│   ├── pipeline/
│   │   ├── batch_processor.py
│   │   ├── data_structures.py
│   │   ├── gpu_optimizer.py
│   │   └── __init__.py
│   ├── summarizer.py
│   ├── utils.py
│   ├── video_handler.py
│   └── __init__.py
├── cache/            # Cache directory for models
├── downloads/        # Downloaded video files
├── logs/            # Application logs
├── models/          # Trained models and weights
├── results/         # Output files and reports
├── temp/            # Temporary processing files
├── main.py          # Main application entry
├── requirements.txt # Project dependencies
├── LICENSE
└── README.md
```

## 🔧 Advanced Configuration

### Environment Variables

```bash
# Create .env file
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
WHISPER_MODEL_SIZE=base
PDF_REPORT_DPI=300
PDF_REPORT_QUALITY=high
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

### PDF Generation Configuration

```python
# config.yaml
pdf_settings:
  dpi: 300
  page_size: "A4"
  font_size: 12
  include_visualizations: true
  compress_images: false
```

## 🛣️ Roadmap

### Short Term

- [ ] Add customizable PDF templates
- [ ] Implement real-time analysis mode
- [ ] Add batch processing support

### Long Term

- [ ] Create web interface
- [ ] Add custom model training
- [ ] Implement distributed processing

## 💡 Use Cases

- **Content Creation**: Automated video summarization and report generation
- **Research**: Analysis of interview recordings with detailed PDF reports
- **Education**: Creating searchable lecture content and study materials
- **Business**: Meeting analysis and documentation
- **Media**: Content analysis and metadata generation

## 🔐 Security

- Data processed locally
- No cloud dependencies
- Optional encryption for sensitive content
- Configurable data retention policies

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/akarales/summatra/issues).

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
- [ReportLab](https://www.reportlab.com/) for PDF generation
- [FFmpeg](https://ffmpeg.org/) for media processing
- All other open-source contributors

## 👤 Author

### **Alex Karales**

- Website: [karales.com](https://karales.com)
- X (Twitter): [@alex_karales](https://x.com/alex_karales)
- Email: [karales@gmail.com](mailto:karales@gmail.com)
- Github: [@akarales](https://github.com/akarales)

## 🌍 Community

- Follow development on [GitHub](https://github.com/akarales/summatra)
- Read our [Blog](https://karales.com/blog/summatra)
- Follow [@alex_karales](https://x.com/alex_karales) for updates

## 📈 Project Status

SUMMATRA is under active development. Check our [Project Board](https://github.com/akarales/summatra/projects) for planned features and current progress.

---

![Karales.com](/api/placeholder/200/50)

Made with ❤️ by [Alex Karales](https://karales.com)
