import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import json
import time
from pydub import AudioSegment
from .utils import extract_video_id, ensure_sufficient_space, cleanup_temp_files

class VideoHandler:
    """Handles video download and audio processing operations"""
    
    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('VideoHandler')
        self.verify_dependencies()

    def verify_dependencies(self):
        """Verify required system dependencies"""
        try:
            # Check ffmpeg and required codecs
            result = subprocess.run(
                ['ffmpeg', '-formats'],
                capture_output=True,
                text=True,
                check=True
            )
            if 'aac' not in result.stdout or 'm4a' not in result.stdout:
                self.logger.warning("FFmpeg may be missing required codecs. Installing...")
                subprocess.run(
                    ['sudo', 'apt', 'install', '-y', 'ffmpeg', 'libavcodec-extra'],
                    check=True
                )

            subprocess.run(['yt-dlp', '--version'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Missing required dependencies: {str(e)}")
            raise RuntimeError("Missing dependencies")

    def create_video_directory(self, video_id: str) -> Path:
        """Creates a unique directory for video files"""
        video_dir = self.download_dir / video_id
        video_dir.mkdir(exist_ok=True)
        return video_dir

    def download_video(self, url: str, keep_video: bool = False) -> Optional[Dict]:
        """Downloads video and audio with progress tracking"""
        try:
            video_id = extract_video_id(url)
            video_dir = self.create_video_directory(video_id)

            # Check disk space (estimate 500MB per video)
            if not ensure_sufficient_space(500, self.download_dir):
                raise Exception("Insufficient disk space")

            self.logger.info("\n📥 Starting download process...")

            # Get video information
            info = self._get_video_info(url)
            if not info:
                raise ValueError("Could not fetch video information")

            video_title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)

            self.logger.info(f"\n📺 Video Title: {video_title}")
            self.logger.info(f"⏱️ Duration: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

            # Download video and audio
            video_file = self._download_video_file(url, video_dir, video_id)
            audio_file = self._download_audio_file(url, video_dir, video_id)

            if not video_file or not audio_file:
                raise Exception("Download failed")

            # Process audio for better quality
            processed_audio = self._process_audio(audio_file)
            
            # Use original audio if processing fails
            final_audio = processed_audio if processed_audio else audio_file

            result = {
                'video_id': video_id,
                'title': video_title,
                'duration': duration,
                'video_file': str(video_file),
                'audio_file': str(final_audio),
                'original_audio': str(audio_file),
                'video_dir': str(video_dir),
                'url': url,
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save metadata
            self._save_metadata(video_dir, result)

            self.logger.info("\n✅ Downloads completed successfully!")
            if not keep_video and video_file.exists():
                self.logger.info(f"Deleting original video file: {video_file}")
                video_file.unlink()

            return result

        except Exception as e:
            self.logger.error(f"\n❌ Download error: {str(e)}")
            return None

    def _get_video_info(self, url: str) -> Dict:
        """Fetch video information"""
        try:
            result = subprocess.run(
                ['yt-dlp', '--dump-json', url],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except Exception as e:
            self.logger.error(f"Error fetching video info: {str(e)}")
            return {}

    def _download_video_file(self, url: str, video_dir: Path, video_id: str) -> Optional[Path]:
        """Download video file"""
        try:
            self.logger.info("\n🎬 Downloading video file...")
            output_template = str(video_dir / f"{video_id}_video.%(ext)s")

            command = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                '--merge-output-format', 'mp4',
                '-o', output_template,
                '--newline',
                '--progress',
                url
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            for line in process.stdout:
                print(f"\r{line.strip()}", end='', flush=True)

            process.wait()
            print()

            if process.returncode != 0:
                stderr = process.stderr.read() if process.stderr else "No error message"
                raise Exception(f"Video download failed with code {process.returncode}: {stderr}")

            video_file = next(video_dir.glob(f"{video_id}_video.*"))
            return video_file

        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            return None

    def _download_audio_file(self, url: str, video_dir: Path, video_id: str) -> Optional[Path]:
        """Download audio file"""
        try:
            self.logger.info("\n🔊 Downloading audio file...")
            output_template = str(video_dir / f"{video_id}_audio.%(ext)s")

            command = [
                'yt-dlp',
                '-f', 'bestaudio[ext=m4a]',
                '-o', output_template,
                '--newline',
                '--progress',
                url
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            for line in process.stdout:
                print(f"\r{line.strip()}", end='', flush=True)

            process.wait()
            print()

            if process.returncode != 0:
                stderr = process.stderr.read() if process.stderr else "No error message"
                raise Exception(f"Audio download failed with code {process.returncode}: {stderr}")

            audio_file = next(video_dir.glob(f"{video_id}_audio.*"))
            return audio_file

        except Exception as e:
            self.logger.error(f"Error downloading audio: {str(e)}")
            return None

    def _process_audio(self, audio_file: Path) -> Optional[Path]:
        """Process audio file to improve quality"""
        try:
            self.logger.info("\n🎵 Processing audio...")
            
            # Create processed file path
            final_audio_file = audio_file.with_name(audio_file.stem + "_processed.m4a")
            
            # Use ffmpeg directly for audio processing
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_file),      # Input file
                '-acodec', 'aac',           # Use AAC codec
                '-b:a', '192k',             # Bitrate
                '-ar', '44100',             # Sample rate
                '-af', 'volume=1.5',        # Normalize volume
                '-movflags', '+faststart',  # Optimize for streaming
                str(final_audio_file)
            ]
            
            # Run ffmpeg command
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                self.logger.error(f"FFmpeg conversion failed: {process.stderr}")
                self.logger.info("Using original audio file as fallback")
                return audio_file
                
            self.logger.info(f"Processed audio saved as: {final_audio_file}")
            return final_audio_file

        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            self.logger.info("Using original audio file as fallback")
            return audio_file

    def _save_metadata(self, video_dir: Path, metadata: Dict):
        """Save metadata to a JSON file"""
        try:
            metadata_file = video_dir / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Metadata saved to: {metadata_file}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")

    def cleanup_files(self, video_info: Dict):
        """Clean up downloaded files"""
        try:
            video_dir = Path(video_info['video_dir'])
            if video_dir.exists():
                for file_path in video_dir.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                video_dir.rmdir()
                self.logger.info(f"Cleaned up directory: {video_dir}")
        except Exception as e:
            self.logger.error(f"Error cleaning up files: {str(e)}")

    def get_download_status(self, video_id: str) -> Dict[str, Union[bool, str]]:
        """Check download status for a video"""
        video_dir = self.download_dir / video_id
        return {
            'exists': video_dir.exists(),
            'has_video': any(video_dir.glob(f"{video_id}_video.*")) if video_dir.exists() else False,
            'has_audio': any(video_dir.glob(f"{video_id}_audio.*")) if video_dir.exists() else False,
            'has_processed_audio': any(video_dir.glob(f"{video_id}_audio_processed.*")) if video_dir.exists() else False,
            'status': 'complete' if all([
                video_dir.exists(),
                any(video_dir.glob(f"{video_id}_audio.*"))
            ]) else 'incomplete'
        }