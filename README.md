# SUMMATRA - Smart Universal Media Mining And Transcription Recognition Assistant

<p align="center">
  <img src="/api/placeholder/800/400" alt="SUMMATRA Logo"/>
</p>

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CUDA Support](https://img.shields.io/badge/CUDA-11.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![FFmpeg Required](https://img.shields.io/badge/FFmpeg-required-red.svg)](https://ffmpeg.org/)
[![Website](https://img.shields.io/badge/website-karales.com-blue.svg)](https://karales.com)
[![Twitter](https://img.shields.io/badge/X-alex__karales-black.svg)](https://x.com/alex_karales)

## 🎯 Overview

SUMMATRA is a powerful, GPU-accelerated video content analysis tool that combines state-of-the-art machine learning models to provide comprehensive video understanding. It automatically transcribes, summarizes, and analyzes video content while generating detailed visualizations and insights.

## 🔍 Demo

![SUMMATRA Demo](/api/placeholder/600/300)

### 🚀 Key Features

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

```bash
# Clone the repository
git clone https://github.com/akarales/summatra.git
cd summatra

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

## 📈 Project Structure

```plaintext
summatra/
├── src/
│   ├── content_processor.py
│   ├── cuda_setup.py
│   ├── model_manager.py
│   ├── pipeline/
│   │   ├── batch_processor.py
│   │   ├── data_structures.py
│   │   ├── gpu_optimizer.py
│   │   └── __init__.py
│   ├── summarizer.py
│   ├── utils.py
│   ├── video_handler.py
│   └── __init__.py
├── cache/            # Cache directory for models and temporary files
├── downloads/        # Downloaded video files
├── logs/            # Application logs
├── models/          # Trained models and weights
├── results/         # Output files and visualizations
├── temp/            # Temporary processing files
├── main.py          # Main application entry point
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

## 🛣️ Roadmap

### Short Term

- [ ] Add support for batch processing
- [ ] Implement real-time analysis mode
- [ ] Enhance visualization options

### Long Term

- [ ] Create web interface
- [ ] Add custom model training
- [ ] Implement distributed processing

## 💡 Use Cases

- **Content Creation**: Automated video summarization and transcription
- **Research**: Analysis of interview recordings and presentations
- **Education**: Creating searchable lecture content
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
- [FFmpeg](https://ffmpeg.org/) for media processing
- All other open-source contributors

## 👤 Author

### **Alex Karales**

- Website: [karales.com](https://karales.com)
- X (Twitter): [@alex_karales](https://x.com/alex_karales)
- Email: karales@gmail.com
- Github: [@akarales](https://github.com/akarales)

## 🌍 Community

- Follow development on [GitHub](https://github.com/akarales/summatra)
- Read our [Blog](https://karales.com/blog/summatra)
- Follow [@alex_karales](https://x.com/alex_karales) for updates

## 📈 Project Status

SUMMATRA is under active development. Check our [Project Board](https://github.com/akarales/summatra/projects) for planned features and current progress.

---

<p align="center">
  <a href="https://karales.com">
    <img src="/api/placeholder/200/50" alt="Karales.com"/>
  </a>
  <br>
  Made with ❤️ by <a href="https://karales.com">Alex Karales</a>
</p>
