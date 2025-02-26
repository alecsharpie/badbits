#!/usr/bin/env python
"""
BadBits - AI-powered posture coach and habit monitor.

This program uses your webcam and AI to help you maintain good posture
and break bad habits like nail-biting, providing gentle reminders when
you need them most.
"""

from pathlib import Path
import logging
import sys
import json
import time
import termios
import tty
import argparse
import os
import platform
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Union, List, Literal, NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moondream as md
from plyer import notification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Alert types - extensible for custom habits
AlertType = str  # Can be any string identifier for a habit

class CheckStats:
    """Statistics for a monitoring session with dynamic habit tracking."""
    
    def __init__(self, 
                 habit_types: List[str] = None,
                 total_checks: int = 0, 
                 start_time: datetime = None,
                 last_check_time: Optional[datetime] = None,
                 habit_alerts: Dict[str, int] = None):
        """
        Initialize session statistics with dynamic habit tracking.
        
        Args:
            habit_types: List of habit identifiers being tracked
            total_checks: Number of checks performed
            start_time: When monitoring started
            last_check_time: When last check was performed
            habit_alerts: Dictionary mapping habit types to alert counts
        """
        self.total_checks = total_checks
        self.start_time = start_time or datetime.now()
        self.last_check_time = last_check_time
        self.habit_alerts = habit_alerts or {}
        
        # Initialize alert counts for any new habit types
        if habit_types:
            for habit in habit_types:
                if habit not in self.habit_alerts:
                    self.habit_alerts[habit] = 0
    
    @property
    def duration_minutes(self) -> int:
        """Get session duration in minutes."""
        return (datetime.now() - self.start_time).seconds // 60
    
    def get_alert_percent(self, habit_type: str) -> int:
        """
        Get percentage of checks with alerts for specific habit.
        
        Args:
            habit_type: The habit identifier to get percentage for
            
        Returns:
            Percentage of checks that triggered this habit alert
        """
        if self.total_checks == 0 or habit_type not in self.habit_alerts:
            return 0
        return int((self.habit_alerts[habit_type] / self.total_checks) * 100)
    
    def update(self, alerts: List["AlertResult"]) -> "CheckStats":
        """
        Create updated stats based on new alerts.
        
        Args:
            alerts: List of alert results from current check
            
        Returns:
            New CheckStats object with updated counts
        """
        # Create a new habit_alerts dictionary with current values
        updated_alerts = dict(self.habit_alerts)
        
        # Update alert counts for each habit type that triggered
        for alert in alerts:
            if alert.is_active:
                if alert.alert_type in updated_alerts:
                    updated_alerts[alert.alert_type] += 1
                else:
                    updated_alerts[alert.alert_type] = 1
        
        # Return a new stats object with updated values
        return CheckStats(
            habit_types=list(updated_alerts.keys()),
            total_checks=self.total_checks + 1,
            start_time=self.start_time,
            last_check_time=datetime.now(),
            habit_alerts=updated_alerts
        )

class AlertResult:
    """
    Represents a binary alert result with supporting details.
    
    This class provides a standardized format for all alerts in the system,
    supporting both serialization to JSON and human-readable display formats.
    """
    def __init__(self, 
                 alert_type: AlertType,
                 is_active: bool, 
                 details: str = "",
                 timestamp: Optional[datetime] = None):
        """
        Initialize an alert result.
        
        Args:
            alert_type: Type of alert (posture or nail_biting)
            is_active: Whether the alert is active (True = bad behavior detected)
            details: Additional details about the alert
            timestamp: When the alert was generated
        """
        self.alert_type = alert_type
        self.is_active = is_active
        self.details = details
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alert_type": self.alert_type,
            "is_active": self.is_active, 
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertResult":
        """Create from dictionary representation."""
        return cls(
            alert_type=data["alert_type"],
            is_active=data["is_active"],
            details=data["details"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def get_status_text(self) -> str:
        """Get a human-readable status text."""
        state = "DETECTED" if self.is_active else "OK"
        return f"{self.alert_type.upper()}: {state}"
    
    def get_emoji(self) -> str:
        """Get an emoji representation of the alert."""
        # Default emojis for built-in habit types
        emoji_map = {
            "posture": "🪑" if self.is_active else "✅",
            "nail_biting": "💅" if self.is_active else "✅",
            "screen_time": "🖥️" if self.is_active else "✅",
            "water": "💧" if self.is_active else "✅",
            "stretching": "🧘" if self.is_active else "✅",
            "eye_strain": "👁️" if self.is_active else "✅",
            "typing_form": "⌨️" if self.is_active else "✅"
        }
        
        # Return the matching emoji or a generic one for custom habits
        return emoji_map.get(self.alert_type, "⚠️" if self.is_active else "✅")

class HabitCheck:
    """
    Defines a custom habit check with its properties.
    
    This class encapsulates the definition of a habit to check,
    including its prompt, details, and display properties.
    """
    
    def __init__(self, 
                 habit_id: str,
                 name: str,
                 emoji: str,
                 prompt: str,
                 details_prompt: Optional[str] = None,
                 description: str = "",
                 active_message: str = "",
                 default_enabled: bool = True):
        """
        Initialize a custom habit check definition.
        
        Args:
            habit_id: Unique identifier for this habit
            name: Display name for the habit
            emoji: Emoji to represent this habit 
            prompt: Vision model prompt to detect this habit
            details_prompt: Optional follow-up prompt for details
            description: Human-readable description of what this checks
            active_message: Message to show when habit is detected
            default_enabled: Whether this check is enabled by default
        """
        self.habit_id = habit_id
        self.name = name
        self.emoji = emoji
        self.prompt = prompt
        self.details_prompt = details_prompt
        self.description = description
        self.active_message = active_message
        self.enabled = default_enabled
    
    def get_display_name(self) -> str:
        """Get formatted display name for UI."""
        return self.name.replace("_", " ").title()
        
    def get_active_message(self) -> str:
        """Get message to show when habit is detected."""
        if self.active_message:
            return self.active_message
        return f"{self.get_display_name()} detected!"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "habit_id": self.habit_id,
            "name": self.name,
            "emoji": self.emoji,
            "prompt": self.prompt,
            "details_prompt": self.details_prompt,
            "description": self.description,
            "active_message": self.active_message,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HabitCheck":
        """Create from dictionary representation."""
        return cls(
            habit_id=data["habit_id"],
            name=data["name"],
            emoji=data["emoji"],
            prompt=data["prompt"],
            details_prompt=data.get("details_prompt"),
            description=data.get("description", ""),
            active_message=data.get("active_message", ""),
            default_enabled=data.get("enabled", True)
        )


class HabitMonitor:
    """
    Analyzes habits using webcam and vision model.
    
    This class handles webcam access, image capture, posture comparison,
    and habit detection using the Moondream vision language model.
    """
    
    def __init__(self, model_path: Union[str, Path], camera_id: int = 0, 
                 backup_camera_ids: List[int] = None, output_dir: str = "habit_monitor",
                 custom_habits_file: Optional[str] = None):
        """
        Initialize the HabitMonitor with a vision model and webcam.
        
        Args:
            model_path: Path to the Moondream model file
            camera_id: Camera device ID (default: 0 for built-in webcam)
            backup_camera_ids: List of backup camera IDs to try if primary fails
            output_dir: Directory to save analysis results
            custom_habits_file: Path to JSON file containing custom habits
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If no webcam can be accessed
        """
        try:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Create output directory
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
            
            # Store reference image
            self.reference_image: Optional[Image.Image] = None
            
            # Store camera IDs
            self.camera_id = camera_id
            self.backup_camera_ids = backup_camera_ids or []
            self.camera_options = [camera_id] + self.backup_camera_ids
            
            # Load habits - start with default habits
            self.habits = self._load_default_habits()
            
            # Try to load custom habits if specified
            if custom_habits_file:
                try:
                    self._load_custom_habits(custom_habits_file)
                except Exception as e:
                    logger.warning(f"Failed to load custom habits: {e}")
            
            logger.info("Loading Moondream model...")
            self.model = md.vl(model=str(self.model_path))
            logger.info("Model loaded successfully")
            
            # Try to initialize webcam with primary and backup options if needed
            self.cap = None
            
            for cam_id in self.camera_options:
                logger.info(f"Trying camera with ID {cam_id}...")
                self.cap = cv2.VideoCapture(cam_id)
                if self.cap.isOpened():
                    self.camera_id = cam_id
                    logger.info(f"Successfully connected to camera with ID {cam_id}")
                    break
                else:
                    self.cap.release()
            
            if self.cap is None or not self.cap.isOpened():
                available_cameras = self._list_available_cameras()
                if available_cameras:
                    raise RuntimeError(f"Could not open any specified webcams. Available camera IDs might be: {available_cameras}")
                else:
                    raise RuntimeError("Could not open any webcams. Please check your camera connections.")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
            
    def _load_default_habits(self) -> Dict[str, HabitCheck]:
        """
        Load the default set of habit checks.
        
        Returns:
            Dictionary mapping habit IDs to HabitCheck objects
        """
        default_habits = {
            "posture": HabitCheck(
                habit_id="posture",
                name="Poor Posture",
                emoji="🪑",
                prompt="Looking at the bottom image only: Is the person slouching or sitting with poor posture? Answer with only 'yes' or 'no'.",
                details_prompt="Looking at the bottom image only: What specific issues do you see with their posture? List up to 3 main issues, separated by commas. If posture looks good, respond with 'good'.",
                description="Detects poor sitting posture compared to your reference image",
                active_message="Poor posture detected! Straighten your back and adjust your position.",
                default_enabled=True
            ),
            "nail_biting": HabitCheck(
                habit_id="nail_biting",
                name="Nail Biting",
                emoji="💅",
                prompt="Looking at the bottom image only: Is the person biting their nails or have their hands near their mouth? Answer with only 'yes' or 'no'.",
                description="Detects nail biting or hands near mouth",
                active_message="Nail biting detected! Be mindful of your hands.",
                default_enabled=True
            ),
            "eye_strain": HabitCheck(
                habit_id="eye_strain",
                name="Eye Strain",
                emoji="👁️",
                prompt="Looking at the bottom image only: Is the person leaning too close to the screen (less than arm's length away)? Answer with only 'yes' or 'no'.",
                description="Detects when you're sitting too close to the screen",
                active_message="You're too close to the screen! Sit back to reduce eye strain.",
                default_enabled=False
            ),
            "screen_break": HabitCheck(
                habit_id="screen_break",
                name="Screen Break",
                emoji="⏱️",
                prompt="This is a timed reminder. Please answer 'yes' to indicate it's time for a screen break.",
                description="Reminds you to take regular breaks from screen time",
                active_message="Time for a screen break! Look away from the screen for 20 seconds.",
                default_enabled=False
            )
        }
        
        logger.info(f"Loaded {len(default_habits)} default habits")
        return default_habits
    
    def _load_custom_habits(self, custom_file: str) -> None:
        """
        Load custom habits from a JSON file.
        
        Args:
            custom_file: Path to JSON file containing habit definitions
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has invalid format
        """
        custom_path = Path(custom_file)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom habits file not found: {custom_file}")
        
        try:
            with open(custom_path, 'r') as f:
                custom_data = json.load(f)
            
            # Parse each habit definition
            for habit_data in custom_data:
                habit = HabitCheck.from_dict(habit_data)
                self.habits[habit.habit_id] = habit
                
            logger.info(f"Loaded {len(custom_data)} custom habits from {custom_file}")
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in custom habits file: {custom_file}")
        except KeyError as e:
            raise ValueError(f"Missing required field in custom habit: {e}")
    
    def save_habits(self, output_file: str) -> None:
        """
        Save current habits to a JSON file.
        
        Args:
            output_file: Path to save habits to
        """
        habits_data = [habit.to_dict() for habit in self.habits.values()]
        
        with open(output_file, 'w') as f:
            json.dump(habits_data, f, indent=2)
        
        logger.info(f"Saved {len(habits_data)} habits to {output_file}")
    
    def enable_habit(self, habit_id: str, enabled: bool = True) -> bool:
        """
        Enable or disable a habit.
        
        Args:
            habit_id: ID of habit to modify
            enabled: Whether to enable or disable it
            
        Returns:
            True if successful, False if habit not found
        """
        if habit_id in self.habits:
            self.habits[habit_id].enabled = enabled
            return True
        return False
            
    def _list_available_cameras(self, max_to_check: int = 5) -> List[int]:
        """
        List available camera devices.
        
        Args:
            max_to_check: Maximum number of camera IDs to check
            
        Returns:
            List of available camera IDs
        """
        available = []
        for i in range(max_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def __del__(self):
        """Cleanup webcam resources"""
        if hasattr(self, 'cap'):
            self.cap.release()

    def capture_frame(self) -> Image.Image:
        """
        Capture a frame from the webcam and convert to PIL Image.
        
        Returns:
            PIL Image object containing the current webcam frame
            
        Raises:
            RuntimeError: If frame capture fails after trying all cameras
        """
        # Try up to 3 times to capture a frame
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # Check if camera is still open
            if not self.cap.isOpened():
                logger.warning(f"Webcam connection lost. Attempting to reconnect (attempt {attempt+1}/{max_attempts})...")
                
                # Try each of our camera options
                for cam_id in self.camera_options:
                    logger.info(f"Trying camera with ID {cam_id}...")
                    
                    # Release previous cap if it exists
                    if hasattr(self, 'cap') and self.cap is not None:
                        self.cap.release()
                    
                    # Try to connect to this camera
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(cam_id)
                    
                    if self.cap.isOpened():
                        logger.info(f"Successfully reconnected to camera with ID {cam_id}")
                        self.camera_id = cam_id
                        break
                
                # If we couldn't connect to any camera
                if not self.cap.isOpened():
                    if attempt == max_attempts - 1:
                        available_cameras = self._list_available_cameras()
                        if available_cameras:
                            raise RuntimeError(f"Failed to connect to any camera. Available camera IDs might be: {available_cameras}")
                        else:
                            raise RuntimeError("Failed to connect to any camera. Please check your camera connections.")
                    continue
            
            # Try to capture frame
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB for PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            
            # If we failed but have more attempts
            if attempt < max_attempts - 1:
                logger.warning(f"Frame capture failed. Retrying ({attempt+1}/{max_attempts})...")
                time.sleep(1)
        
        # If we get here, all attempts failed
        raise RuntimeError("Failed to capture frame from webcam after multiple attempts with all cameras")

    def create_collage(self, current_image: Image.Image) -> Image.Image:
        """
        Create a collage with reference image on top and current image below.
        
        Args:
            current_image: The current webcam frame as PIL Image
            
        Returns:
            A collage image containing reference and current images
            
        Raises:
            RuntimeError: If reference image has not been set
        """
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

    def analyze_habits(self, collage: Image.Image) -> List[AlertResult]:
        """
        Analyze habits in the collage image using the vision model.
        
        Args:
            collage: The composite image containing reference and current posture
            
        Returns:
            List of AlertResult objects for each detected behavior
                
        Raises:
            Exception: If image analysis fails
        """
        try:
            encoded_image = self.model.encode_image(collage)
            
            now = datetime.now()
            results: List[AlertResult] = []
            
            # Process each enabled habit
            for habit_id, habit in self.habits.items():
                # Skip disabled habits
                if not habit.enabled:
                    continue
                
                # Process the habit check
                response = self.model.query(encoded_image, habit.prompt)
                answer = response["answer"].lower().strip()
                
                # Create a binary result (is_active = True means the alert is active)
                is_active = answer == "yes"
                
                # Get details if needed and available
                details = ""
                if is_active and habit.details_prompt:
                    detail_response = self.model.query(encoded_image, habit.details_prompt)
                    details = detail_response["answer"]
                
                # Create the alert result
                result = AlertResult(
                    alert_type=habit_id,
                    is_active=is_active,
                    details=details,
                    timestamp=now
                )
                
                results.append(result)
            
            # If we have no enabled habits, add a dummy result
            if not results:
                results.append(AlertResult(
                    alert_type="no_checks",
                    is_active=False,
                    details="No habit checks enabled",
                    timestamp=now
                ))
            
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

    def save_analysis(self, collage: Image.Image, alerts: List[AlertResult], timestamp: str, archive_mode: bool = False) -> Optional[Path]:
        """
        Save the collage and analysis results to a timestamped directory if archiving is enabled.
        
        Args:
            collage: The comparison image to save
            alerts: List of AlertResult objects
            timestamp: Timestamp string for the directory name
            archive_mode: Whether to save data to disk (privacy protection)
            
        Returns:
            Path to the created analysis directory or None if archiving is disabled
        """
        # If archiving is disabled, don't save anything to disk
        if not archive_mode:
            return None
            
        # Create the analysis directory
        analysis_dir = self.output_dir / timestamp
        analysis_dir.mkdir(exist_ok=True)
        
        # Save the collage image
        collage_path = analysis_dir / "comparison.jpg"
        collage.save(collage_path)
        
        # Convert alerts to dictionary for JSON serialization
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "alerts": [alert.to_dict() for alert in alerts]
        }
        
        # Save the analysis results
        results_path = analysis_dir / "analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        return analysis_dir

    def send_alert_notification(self, alert: AlertResult) -> None:
        """
        Send a desktop notification for an active alert.
        
        Args:
            alert: The AlertResult to send a notification for
        """
        if not alert.is_active:
            return
            
        # Get the habit details if available
        habit = self.habits.get(alert.alert_type)
        
        # Create the notification title
        if habit:
            alert_name = habit.get_display_name()
            title = f"BadBits Alert: {alert_name}"
        else:
            title = f"BadBits Alert: {alert.alert_type.replace('_', ' ').title()}"
        
        # Create the notification message
        if habit and habit.active_message:
            # Use the custom message from the habit definition
            message = habit.active_message
            # Append details if available
            if alert.details:
                message = f"{message} {alert.details}"
        else:
            # Default message if habit is not found
            message = f"Issue detected: {alert.details}" if alert.details else "Issue detected!"
            
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="BadBits",
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    def render_dashboard(self, 
                         stats: CheckStats,
                         current_alerts: List[AlertResult], 
                         next_check_time: Optional[datetime] = None,
                         error_message: str = "") -> str:
        """
        Render a dashboard-style UI.
        
        Args:
            stats: Current session statistics
            current_alerts: List of current alert results
            next_check_time: When the next check will occur
            error_message: Error message to show (if any)
            
        Returns:
            Formatted string for terminal display
        """
        # Get terminal size
        terminal_width = shutil.get_terminal_size().columns
        
        # Prepare header
        border = "═" * terminal_width
        title = "BadBits Posture Monitor".center(terminal_width)
        subtitle = "Real-time AI-powered habit tracking".center(terminal_width)
        
        # Prepare stats section
        stats_width = terminal_width - 4  # Account for padding
        stats_left_width = stats_width // 2
        stats_right_width = stats_width - stats_left_width
        
        duration_str = f"Session: {stats.duration_minutes} minutes"
        checks_str = f"Checks: {stats.total_checks}"
        
        stats_line1_left = duration_str.ljust(stats_left_width)
        stats_line1_right = checks_str.rjust(stats_right_width)
        
        # Prepare alert section
        alert_header = "CURRENT STATUS".center(terminal_width)
        alert_border = "─" * terminal_width
        
        alert_lines = []
        for alert in current_alerts:
            status = "⚠️  ALERT" if alert.is_active else "✅ OK"
            alert_type_display = alert.alert_type.replace("_", " ").title()
            alert_line = f"  {alert.get_emoji()} {alert_type_display}: {status}"
            alert_lines.append(alert_line)
            
            if alert.is_active and alert.details:
                alert_lines.append(f"     ℹ️  {alert.details}")
        
        # Prepare history section
        history_header = "SESSION HISTORY".center(terminal_width)
        
        # Generate progress bars for each habit
        history_lines = []
        bar_width = terminal_width - 30
        
        # Create a progress bar for each habit that has been tracked
        for habit_id, habit in self.habits.items():
            if not habit.enabled:
                continue
                
            # Get percent for this habit
            percent = stats.get_alert_percent(habit_id)
            
            # Calculate bar fill
            fill_width = int((percent / 100) * bar_width)
            habit_bar = f"[{'■' * fill_width}{' ' * (bar_width - fill_width)}]"
            
            # Create the line
            line = f"  {habit.emoji} {habit.get_display_name()}: {percent}% {habit_bar}"
            history_lines.append(line)
        
        # If no habits enabled, show message
        if not history_lines:
            history_lines = ["  No habit checks enabled."]
        
        # Prepare next check and status
        next_check_str = ""
        if next_check_time:
            time_left = max(0, (next_check_time - datetime.now()).seconds)
            next_check_str = f"Next check in {time_left} seconds".center(terminal_width)
        
        # Error message (if any)
        error_line = ""
        if error_message:
            error_line = f"\n⚠️  {error_message}".center(terminal_width)
        
        # Last check time
        last_check_str = ""
        if stats.last_check_time:
            last_check_str = f"Last check: {stats.last_check_time.strftime('%H:%M:%S')}".center(terminal_width)
        
        # Assemble dashboard
        dashboard = [
            border,
            title,
            subtitle,
            border,
            "",
            f"  {stats_line1_left}{stats_line1_right}",
            "",
            alert_header,
            alert_border,
        ] + alert_lines + [
            "",
            history_header,
            alert_border,
        ] + history_lines + [
            "",
            last_check_str,
            next_check_str,
            error_line,
            "",
            "Press Ctrl+C to exit".center(terminal_width),
            border
        ]
        
        return "\n".join(dashboard)
                    
    def run_continuous_monitoring(self, interval_seconds: int = 60, 
                                  notification_enabled: bool = True,
                                  archive_mode: bool = False,
                                  dashboard_mode: bool = True):
        """
        Run continuous posture monitoring with the specified interval.
        
        Args:
            interval_seconds: Time between checks in seconds
            notification_enabled: Whether to enable desktop notifications
            archive_mode: Whether to save check data to disk (privacy)
            dashboard_mode: Whether to use dashboard UI mode
            
        Raises:
            Exception: If monitoring fails
        """
        try:
            # First capture reference image
            self.capture_reference()
            
            logger.info("Starting continuous posture monitoring...")
            logger.info("Press Ctrl+C to stop")
            
            # Initialize session tracking
            stats = CheckStats()
            last_alerts: List[AlertResult] = []
            error_message = ""
            
            # Initial banner (only for non-dashboard mode)
            if not dashboard_mode:
                print("\n" + "="*50)
                print("🚀 BadBits Monitoring Started")
                print("="*50)
                print("💡 Real-time alerts will appear as desktop notifications")
                if archive_mode:
                    print("💾 Saving all checks to disk for review")
                else:
                    print("🔒 Privacy mode: No images saved to disk")
                print("❌ Press Ctrl+C to stop monitoring\n")
            
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                next_check_time = datetime.now() + timedelta(seconds=interval_seconds)
                
                # Try to capture and analyze current frame
                try:
                    current_image = self.capture_frame()
                    collage = self.create_collage(current_image)
                    current_alerts = self.analyze_habits(collage)
                    
                    # Save analysis if archiving is enabled
                    analysis_dir = self.save_analysis(
                        collage, 
                        current_alerts, 
                        timestamp, 
                        archive_mode=archive_mode
                    )
                    
                    # Update stats and save the current alerts
                    stats = stats.update(current_alerts)
                    last_alerts = current_alerts
                    error_message = ""
                    
                    # Send notifications
                    for alert in current_alerts:
                        if notification_enabled and alert.is_active:
                            self.send_alert_notification(alert)
                    
                except RuntimeError as e:
                    error_message = str(e)
                    logger.warning(f"Check failed: {error_message}")
                
                # Display results
                if dashboard_mode:
                    # Clear screen
                    if platform.system() != "Windows":
                        os.system('clear')
                    else:
                        os.system('cls')
                    
                    # Render and display dashboard
                    dashboard = self.render_dashboard(
                        stats=stats,
                        current_alerts=last_alerts,
                        next_check_time=next_check_time,
                        error_message=error_message
                    )
                    print(dashboard)
                else:
                    # Traditional output mode
                    print(f"\n🔍 CHECK #{stats.total_checks} at {timestamp}")
                    print("="*50)
                    
                    # Print storage message
                    if archive_mode and analysis_dir:
                        print(f"📁 Analysis saved to: {analysis_dir}")
                    elif not archive_mode:
                        print("🔒 Privacy mode: No data saved to disk")
                    
                    # Display alerts
                    print("\n📊 CURRENT STATUS:")
                    print("-"*40)
                    
                    for alert in last_alerts:
                        status = "⚠️ DETECTED" if alert.is_active else "✅ OK"
                        alert_type_display = alert.alert_type.replace("_", " ").title()
                        print(f"{alert.get_emoji()} {alert_type_display}: {status}")
                        
                        if alert.is_active and alert.details:
                            print(f"   Details: {alert.details}")
                    
                    # Show session stats
                    print("\n📈 SESSION SUMMARY:")
                    print("-"*40)
                    print(f"• Duration: {stats.duration_minutes} minutes ({stats.total_checks} checks)")
                    print(f"• Poor posture detected: {stats.posture_alerts}/{stats.total_checks} checks ({stats.posture_alert_percent}%)")
                    print(f"• Nail biting detected: {stats.nail_biting_alerts}/{stats.total_checks} checks ({stats.nail_biting_alert_percent}%)")
                    
                    # Show error if any
                    if error_message:
                        print(f"\n⚠️ WARNING: {error_message}")
                    
                    print(f"\n⏱️  Next check in {interval_seconds} seconds...")
                
                # Wait for the next check
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            # Final summary on exit
            if dashboard_mode:
                # Clear screen once more for final message
                if platform.system() != "Windows":
                    os.system('clear')
                else:
                    os.system('cls')
            
            # Show final stats
            print("\n" + "="*50)
            print("👋 MONITORING SESSION ENDED")
            print("="*50)
            print(f"• Session duration: {stats.duration_minutes} minutes ({stats.total_checks} checks)")
            
            # Show stats for each habit
            for habit_id, habit in self.habits.items():
                if habit.enabled:
                    alert_count = stats.habit_alerts.get(habit_id, 0)
                    percent = stats.get_alert_percent(habit_id)
                    print(f"• {habit.emoji} {habit.get_display_name()}: {alert_count}/{stats.total_checks} checks ({percent}%)")
            
            if archive_mode:
                print("\nAnalysis data saved to: " + str(self.output_dir))
            print("="*50)
            
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()

def download_model(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """
    Download and decompress the model file if it doesn't exist.
    
    Args:
        url: URL to download the model from
        output_path: Path where the model should be saved
        chunk_size: Size of chunks to download at a time
    """
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

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="BadBits - AI-powered posture coach and habit monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--interval", "-i", 
        type=int, 
        default=60,
        help="Time between posture checks in seconds"
    )
    
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=0,
        help="Camera device ID to use"
    )
    
    parser.add_argument(
        "--backup-cameras",
        type=str,
        default="",
        help="Comma-separated list of backup camera IDs to try if primary fails (e.g. '1,2')"
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        default="posture_analysis",
        help="Directory to save analysis results"
    )
    
    parser.add_argument(
        "--no-alerts", "-n",
        action="store_true",
        help="Disable desktop notifications"
    )
    
    parser.add_argument(
        "--download-only", "-d",
        action="store_true",
        help="Only download the model without starting monitoring"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default="models/moondream-2b-int8.mf",
        help="Path to the Moondream model file"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--track", "-t",
        action="store_true",
        help="Save images and analysis data to disk for tracking progress"
    )
    
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Use simple console output instead of dashboard UI"
    )
    
    parser.add_argument(
        "--habits", "-H",
        type=str,
        help="Path to custom habits configuration JSON file"
    )
    
    parser.add_argument(
        "--save-habits",
        type=str,
        help="Export current habits configuration to specified JSON file"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function to run the posture monitoring application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Model information
    MODEL_URL = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"
    MODEL_PATH = Path(args.model_path)
    
    try:
        # Print banner
        if not args.quiet:
            print("\n" + "="*60)
            print(" "*20 + "BadBits Monitor")
            print(" "*15 + "Habit Tracking & Coaching")
            print("="*60)
            if not args.track:
                print("\n🔒 Privacy mode enabled: No images will be saved to disk")
                print("   Use --track (-t) to save data for progress tracking")
            else:
                print("\n📊 Tracking mode enabled: Images and data will be saved")
                
            if not args.simple:
                print("\n🖥️  Dashboard interface enabled")
                
            if args.habits:
                print(f"\n📋 Loading custom habits from: {args.habits}")
                
            print("")
        
        # Download model if needed
        download_model(MODEL_URL, MODEL_PATH)
        
        if args.download_only:
            logger.info("Model downloaded successfully. Exiting as requested.")
            return
        
        # Parse backup cameras
        backup_cameras = []
        if args.backup_cameras:
            try:
                backup_cameras = [int(cam_id) for cam_id in args.backup_cameras.split(',')]
                logger.info(f"Using backup cameras: {backup_cameras}")
            except ValueError:
                logger.warning("Invalid backup camera IDs provided. Using only primary camera.")
        
        # Initialize habit monitor
        monitor = HabitMonitor(
            model_path=MODEL_PATH,
            camera_id=args.camera,
            backup_camera_ids=backup_cameras,
            output_dir=args.output_dir,
            custom_habits_file=args.habits
        )
        
        # Save habits configuration if requested
        if args.save_habits:
            monitor.save_habits(args.save_habits)
            if not args.quiet:
                print(f"Habits configuration saved to: {args.save_habits}")
                print("Exiting as requested.")
            return
        
        # Start monitoring with selected options
        monitor.run_continuous_monitoring(
            interval_seconds=args.interval,
            notification_enabled=not args.no_alerts,
            archive_mode=args.track,
            dashboard_mode=not args.simple
        )
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"ERROR: File not found: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Handled inside run_continuous_monitoring
        pass
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()