#!/usr/bin/env python3
import sys
import logging
import argparse
import time
import os
import re
import torch
from rich.console import Console
from rich.prompt import Confirm
from rich.progress import Progress

# Adjust system path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import (
    VideoSummarizer,
    setup_logging,
    create_directory_structure,
    verify_ffmpeg_installation
)

# Adjust system path for module imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from src.summarizer import VideoSummarizer
# from src.utils import setup_logging, create_directory_structure, verify_ffmpeg_installation

console = Console()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Video Summarizer - Extract and summarize content from videos'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Video URL to process'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up downloaded files after processing'
    )
    parser.add_argument(
        '--force-update',
        action='store_true',
        help='Force update of all models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to store results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    parser.add_argument(
        '--safe-mode',
        action='store_true',
        help='Run in safe mode with minimal GPU memory usage'
    )
    return parser.parse_args()

def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL format"""
    pattern = re.compile(
        r'^(https?://)?(www\.)?'
        r'(youtube\.com/watch\?v=|youtu\.be/)[\w-]+(&.*)?$'
    )
    return bool(re.match(pattern, url))

def process_single_video(url: str, summarizer: VideoSummarizer, cleanup: bool = False) -> bool:
    """Process a single video URL"""
    try:
        start_time = time.time()
        result = summarizer.process_video(url, cleanup=cleanup)

        if result:
            processing_time = time.time() - start_time
            console.print(f"\n✅ Processing completed in {processing_time:.2f} seconds", style="green")
            console.print(f"📁 Results saved to: {result['output_file']}", style="blue")
            return True
        
        console.print("\n❌ Processing failed - check logs for details", style="red")
        return False

    except KeyboardInterrupt:
        console.print("\n⚠️ Processing interrupted by user", style="yellow")
        return False
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}", exc_info=True)
        return False

def setup_gpu(args):
    """Setup GPU configuration based on arguments"""
    if args.cpu_only:
        return "cpu"
        
    if not torch.cuda.is_available():
        return "cpu"
        
    try:
        if args.safe_mode:
            # Configure for minimal memory usage
            torch.cuda.set_per_process_memory_fraction(0.5)
        else:
            torch.cuda.set_per_process_memory_fraction(0.8)
            
        # Enable device-side assertions
        torch.backends.cuda.enable_device_side_assertions = True
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return "cuda"
    except Exception as e:
        console.print(f"Error setting up GPU: {e}", style="red")
        return "cpu"

def interactive_mode(summarizer: VideoSummarizer):
    """Run in interactive mode"""
    console.print("\n" + "="*50, style="blue")
    console.print("🎥 Video Summarizer - Interactive Mode", style="green bold")
    console.print("="*50, style="blue")
    console.print("\nType 'q' to quit, 'h' for help")

    while True:
        try:
            command = console.input("\n📌 Enter video URL: ").strip()

            if command.lower() == 'q':
                break
            elif command.lower() == 'h':
                console.print("\nCommands:", style="yellow")
                console.print("  URL  - Process a video")
                console.print("  h    - Show this help")
                console.print("  q    - Quit the program")
                continue
            elif not command:
                continue

            if not is_valid_youtube_url(command):
                console.print("❌ Invalid YouTube URL. Please try again.", style="red")
                continue

            cleanup = Confirm.ask("🗑️ Clean up downloaded files?")

            if process_single_video(command, summarizer, cleanup):
                console.print("\n✨ Processing completed successfully!", style="green")
            else:
                console.print("\n❌ Processing failed. Please try another URL or check logs.", style="red")

            console.print("\n" + "="*50, style="blue")

        except KeyboardInterrupt:
            console.print("\n⚠️ Operation cancelled", style="yellow")
            break
        except Exception as e:
            logging.error(f"Interactive mode error: {str(e)}", exc_info=True)
            console.print("\n❌ An error occurred. Please try again.", style="red")

def main():
    """Main entry point"""
    try:
        # Setup basic structure and logging
        create_directory_structure()
        args = parse_arguments()
        logger = setup_logging(verbose=args.verbose)

        # Check system requirements
        if not verify_ffmpeg_installation():
            console.print("❌ FFmpeg is not properly installed. Please install FFmpeg and required codecs.", style="red")
            sys.exit(1)

        # Setup device
        device = setup_gpu(args)
        
        # Initialize summarizer
        console.print("\n🚀 Initializing Video Summarizer...", style="green")
        summarizer = VideoSummarizer(device=device)

        # Handle model updates
        if args.force_update:
            console.print("\n🔄 Forcing model updates...", style="yellow")
            summarizer.model_manager.setup_all_models(force=True)
        else:
            # Check for available updates
            updates = summarizer.model_manager.check_updates_available()
            if updates:
                console.print("\n📦 Updates available for:", ", ".join(updates), style="yellow")
                if Confirm.ask("Update now?"):
                    summarizer.model_manager.setup_all_models(force=True)

        # Process video or run interactive mode
        if args.url:
            success = process_single_video(args.url, summarizer, args.cleanup)
            sys.exit(0 if success else 1)
        else:
            interactive_mode(summarizer)

    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="blue")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        console.print("\n✨ Thank you for using Video Summarizer!", style="green")

if __name__ == "__main__":
    main()