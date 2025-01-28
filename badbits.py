from pathlib import Path
import logging
import cv2
from PIL import Image, ImageDraw
import moondream as md
import time
from datetime import datetime
import json
import numpy as np
import sys
import termios
import tty
from threading import Timer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostureAnalyzer:
    def __init__(self, model_path: str | Path):
        """Initialize the Moondream vision language model and webcam."""
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Create output directory
            self.output_dir = Path("posture_analysis")
            self.output_dir.mkdir(exist_ok=True)
            
            # Store reference image
            self.reference_image = None
            
            logger.info("Loading Moondream model...")
            self.model = md.vl(model=str(self.model_path))
            logger.info("Model loaded successfully")
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def __del__(self):
        """Cleanup webcam resources"""
        if hasattr(self, 'cap'):
            self.cap.release()

    def capture_frame(self) -> Image.Image:
        """Capture a frame from the webcam and convert to PIL Image."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def create_collage(self, current_image: Image.Image) -> Image.Image:
        """Create a collage with reference image on top and current image below."""
        if self.reference_image is None:
            raise RuntimeError("Reference image not set")
        
        # Ensure both images are the same size
        width = max(self.reference_image.width, current_image.width)
        height = max(self.reference_image.height, current_image.height)
        
        ref_resized = self.reference_image.resize((width, height))
        curr_resized = current_image.resize((width, height))
        
        # Create larger white strip for middle border (50 pixels high)
        border_height = 50
        
        # Add black borders (2 pixels) around images
        border_width = 2
        
        # Create new image with space for both images, borders, and white strip
        collage = Image.new('RGB', (width, height * 2 + border_height), 'white')
        
        # Create black borders for reference image
        black_border_top = Image.new('RGB', (width + 2*border_width, height + 2*border_width), 'black')
        black_border_top.paste(ref_resized, (border_width, border_width))
        
        # Create black borders for current image
        black_border_bottom = Image.new('RGB', (width + 2*border_width, height + 2*border_width), 'black')
        black_border_bottom.paste(curr_resized, (border_width, border_width))
        
        # Paste bordered images
        collage.paste(black_border_top, (0, 0))
        collage.paste(black_border_bottom, (0, height + border_height))
        
        # Add labels with larger font
        draw = ImageDraw.Draw(collage)
        draw.text((10, height//2 - 20), "Reference Posture", fill='black')
        draw.text((10, height * 1.5 + border_height - 20), "Current Posture", fill='black')
        
        return collage

    def analyze_posture(self, collage: Image.Image) -> dict:
        """Analyze posture comparison in the collage."""
        try:
            encoded_image = self.model.encode_image(collage)
            
            prompts = {
                "posture_check": "Looking at the bottom image only: Is the person slouching or sitting with poor posture? Answer with only 'yes' or 'no'.",
                "nail_biting": "Looking at the bottom image only: Is the person biting their nails or have their hands near their mouth? Answer with only 'yes' or 'no'.",
                "posture_details": "Looking at the bottom image only: What specific issues do you see with their posture? List up to 3 main issues, separated by commas. If posture looks good, respond with 'good'."
            }
            
            results = {}
            for key, prompt in prompts.items():
                result = self.model.query(encoded_image, prompt)
                results[key] = result["answer"]
            
            return results
                
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            raise

    def capture_reference(self):
        """Capture reference posture image interactively."""
        print("\n=== Reference Posture Capture ===")
        print("Let's capture your ideal posture!")
        print("\nInstructions:")
        print("1. Sit in your best posture")
        print("2. Type the word 'yellow' when ready")
        print("3. The photo will be taken while you're typing!")
        print("\nGet ready and start typing 'yellow' when you're in position...")
        
        # Store original terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            target_word = "yellow"
            typed = ""
            capture_done = False
            
            while len(typed) < len(target_word):
                char = sys.stdin.read(1)
                # Exit on Ctrl-C
                if ord(char) == 3:
                    raise KeyboardInterrupt
                
                typed += char
                sys.stdout.write(char)
                sys.stdout.flush()
                
                # Take photo around the middle of the word
                if len(typed) == len(target_word) // 2 and not capture_done:
                    self.reference_image = self.capture_frame()
                    capture_done = True
            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        print("\n\nReference image captured! 📸")
        
        # Save reference image
        ref_path = self.output_dir / "reference_posture.jpg"
        self.reference_image.save(ref_path)
        print(f"Reference image saved to: {ref_path}")
        print("\nStarting posture monitoring...")

    def save_analysis(self, collage: Image.Image, results: dict, timestamp: str):
        """Save the collage and analysis results."""
        analysis_dir = self.output_dir / timestamp
        analysis_dir.mkdir(exist_ok=True)
        
        collage_path = analysis_dir / "comparison.jpg"
        collage.save(collage_path)
        
        results_path = analysis_dir / "analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return analysis_dir

    def run_continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous posture monitoring with the specified interval."""
        try:
            # First capture reference image
            self.capture_reference()
            
            logger.info("Starting continuous posture monitoring...")
            logger.info("Press Ctrl+C to stop")
            
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\n=== Posture Check at {timestamp} ===")
                
                current_image = self.capture_frame()
                collage = self.create_collage(current_image)
                
                results = self.analyze_posture(collage)
                
                analysis_dir = self.save_analysis(collage, results, timestamp)
                print(f"Analysis saved to: {analysis_dir}")
                
                print("\nAnalysis Results:")
                print(f"Poor Posture: {results['posture_check']}")
                print(f"Nail Biting: {results['nail_biting']}")
                if results['posture_check'].lower() == 'yes':
                    print(f"Posture Issues: {results['posture_details']}")
                
                print(f"\nNext check in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise
        finally:
            self.cap.release()

def download_model(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download the model file with a progress bar if it doesn't exist."""
    import requests
    from tqdm import tqdm
    import gzip
    import shutil

    # Create models directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if uncompressed model already exists
    if output_path.exists():
        logger.info(f"Model already exists at {output_path}")
        return

    compressed_path = output_path.parent / (output_path.name + '.gz')
    
    # Download if compressed file doesn't exist
    if not compressed_path.exists():
        logger.info(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        # Create progress bar
        progress = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc="Downloading model"
        )

        # Download with progress bar
        with open(compressed_path, 'wb') as f:
            for data in response.iter_content(chunk_size):
                progress.update(len(data))
                f.write(data)
        progress.close()

    # Decompress the file
    if compressed_path.exists():
        logger.info("Decompressing model file...")
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Optionally remove the compressed file
        compressed_path.unlink()
        logger.info(f"Model ready at {output_path}")

def main():
    # Model information
    MODEL_URL = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"
    MODEL_PATH = Path("models/moondream-2b-int8.mf")
    
    try:
        # Download model if needed
        download_model(MODEL_URL, MODEL_PATH)
        
        # Initialize analyzer
        analyzer = PostureAnalyzer(MODEL_PATH)
        analyzer.run_continuous_monitoring()
            
    except Exception as e:
        logger.error(f"Program failed: {e}")
        raise

if __name__ == "__main__":
    main()