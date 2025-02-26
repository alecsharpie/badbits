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

# Alert system imports
import subprocess
import webbrowser
from threading import Thread
import tempfile
import base64

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


class AlertManager:
    """
    Manages different types of alert notifications for BadBits.
    
    This class handles multiple notification methods including:
    - Desktop notifications (via plyer)
    - System alerts (via platform-specific commands)
    - Browser notifications (via simple HTML page)
    - Sound alerts
    - Dramatic full-screen interruption alerts
    """
    
    def __init__(self, app_name: str = "BadBits"):
        """
        Initialize the alert manager.
        
        Args:
            app_name: Name of the application to show in notifications
        """
        self.app_name = app_name
        self.system = platform.system()
        self.notification_html = None
        self.browser_window = None
        
    def desktop_notification(self, title: str, message: str, timeout: int = 10) -> bool:
        """
        Send a desktop notification using plyer.
        
        Args:
            title: Notification title
            message: Notification message
            timeout: Notification timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to handle the common pyobjus missing error on macOS specifically
            if self.system == "Darwin":
                # Check if pyobjus is available before using notification
                try:
                    import importlib
                    if importlib.util.find_spec("pyobjus") is None:
                        logger.warning("The pyobjus package is required for macOS notifications")
                        logger.warning("To install: pip install pyobjus")
                        # Use AppleScript directly as fallback
                        return self.system_alert(title, message)
                except Exception:
                    # Fallback to system notification
                    return self.system_alert(title, message)
            
            # Try the regular notification mechanism
            notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                timeout=timeout
            )
            return True
        except Exception as e:
            logger.warning(f"Desktop notification failed: {e}")
            # Return False to allow fallback methods
            return False
            
    def system_alert(self, title: str, message: str) -> bool:
        """
        Show system alert using platform-specific methods.
        
        Args:
            title: Alert title
            message: Alert message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system == "Darwin":  # macOS
                # First try applescript for macOS
                try:
                    apple_script = f'display notification "{message}" with title "{title}"'
                    subprocess.run(["osascript", "-e", apple_script], check=True)
                    return True
                except Exception as e:
                    logger.warning(f"AppleScript notification failed: {e}")
                    
                # If AppleScript fails, try terminal-notifier as fallback
                try:
                    # Check if terminal-notifier is installed
                    which_result = subprocess.run(
                        ["which", "terminal-notifier"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    
                    if which_result.returncode == 0:
                        subprocess.run([
                            "terminal-notifier",
                            "-title", title,
                            "-message", message,
                            "-sound", "default"
                        ], check=True)
                        return True
                    else:
                        logger.warning("terminal-notifier not found. To install: brew install terminal-notifier")
                except Exception as e:
                    logger.warning(f"terminal-notifier failed: {e}")
                    
            elif self.system == "Linux":
                # Try multiple Linux notification methods
                
                # First try notify-send
                try:
                    subprocess.run([
                        "notify-send", 
                        title, 
                        message,
                        "--icon=dialog-information"
                    ], check=True)
                    return True
                except Exception as e:
                    logger.warning(f"notify-send failed: {e}")
                
                # Then try zenity
                try:
                    subprocess.run([
                        "zenity", 
                        "--info", 
                        f"--title={title}", 
                        f"--text={message}"
                    ], check=True)
                    return True
                except Exception as e:
                    logger.warning(f"zenity notification failed: {e}")
                    
            elif self.system == "Windows":
                # Try multiple Windows notification methods
                
                # First try PowerShell notification
                try:
                    powershell_cmd = f'[System.Windows.Forms.MessageBox]::Show("{message}", "{title}")'
                    subprocess.run(
                        ["powershell", "-Command", powershell_cmd],
                        check=True
                    )
                    return True
                except Exception as e:
                    logger.warning(f"PowerShell notification failed: {e}")
                
                # Then try msg command for Windows
                try:
                    subprocess.run([
                        "msg", 
                        "%username%", 
                        f"{title}: {message}"
                    ], check=True)
                    return True
                except Exception as e:
                    logger.warning(f"Windows msg command failed: {e}")
            
            # If we get here, all platform-specific methods failed
            return False
        except Exception as e:
            logger.warning(f"System alert failed: {e}")
            return False
    
    def sound_alert(self) -> bool:
        """
        Play a sound alert.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.system == "Darwin":  # macOS
                subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=True)
                return True
            elif self.system == "Linux":
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"], check=True)
                return True
            elif self.system == "Windows":
                import winsound
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                return True
            return False
        except Exception as e:
            logger.warning(f"Sound alert failed: {e}")
            return False
    
    def _create_notification_html(self) -> str:
        """
        Create HTML for browser notifications.
        
        Returns:
            HTML string for notifications page
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.app_name} Notifications</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .notification-container {{ margin-bottom: 20px; }}
        .notification {{ 
            background-color: white; 
            border-left: 4px solid #007bff; 
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            animation: slideIn 0.5s ease-out;
        }}
        .notification.alert {{ border-left-color: #dc3545; }}
        .notification-title {{ font-weight: bold; margin-bottom: 5px; }}
        .notification-body {{ color: #555; }}
        .notification-time {{ color: #777; font-size: 0.8em; margin-top: 5px; }}
        @keyframes slideIn {{ from {{ transform: translateX(-100%); opacity: 0; }} to {{ transform: translateX(0); opacity: 1; }} }}
        #header {{ background-color: #2c3e50; color: white; padding: 15px; margin: -20px -20px 20px -20px; }}
    </style>
</head>
<body>
    <div id="header">
        <h2>{self.app_name} Notifications</h2>
        <p>Real-time alerts will appear here</p>
    </div>
    <div id="notifications">
    </div>
    <script>
        function addNotification(title, message, isAlert = false) {{
            const container = document.createElement('div');
            container.className = 'notification-container';
            
            const notification = document.createElement('div');
            notification.className = isAlert ? 'notification alert' : 'notification';
            
            const titleElement = document.createElement('div');
            titleElement.className = 'notification-title';
            titleElement.textContent = title;
            
            const bodyElement = document.createElement('div');
            bodyElement.className = 'notification-body';
            bodyElement.textContent = message;
            
            const timeElement = document.createElement('div');
            timeElement.className = 'notification-time';
            timeElement.textContent = new Date().toLocaleTimeString();
            
            notification.appendChild(titleElement);
            notification.appendChild(bodyElement);
            notification.appendChild(timeElement);
            container.appendChild(notification);
            
            document.getElementById('notifications').prepend(container);
            
            // Play sound
            new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PurWcHENCa0evBhDEQKXrF5cqQRBMeSKjd+b97HwMvh9f0yH4+ChuBx+PMjkwXHFSs2se1RQxDn+DtwH0rnNJ8IQIwktvoq1kJBGjN9dO5Y0wZVbBS+tzZgAlQwHIaA36i1tuYNhBZtm4TEnnS775sRzFWqVoPaLj34IQM6p9cHTC07vKVGUCo3t+DSBxLr/THdxpDzpdTFWmo2/OgMg05074jETib6ueEJ+aWTxFrw/vieQgTgOMOMn3Y/K9RBliX7tgYAoTsLj2M1ueZKQNwu1ISWaf0xVwC0qlNFG3I/MhRA2Go99QPCvXqgRbVpMTSkigJSbCKbwAPecXw12YM3p9QFXfT/89PAl2m9r8cCPnlfQvosnnXiUAQZbODTgEUfc7yxlMF76RdHHHB+spLBYrg4Yk6AcyYSxmA1P3NTAFUvM4yV4Tj7GANDJrQ2pUzDF2iyQEzjPL5jzUFZrXnBFJ9++etKc70jSMsj++8LBgTpOL9ClB9/vemJd73myxMmOXOGhEis/v9LE0omsQ5S5fiyicEP7D//FFhHJG2Ognvw/c2VRScr6FpB3e1+fwuCNu4fXkPxqFsPA6MytajLQlZvILvD9rdsQ8PwbnB/kZAYqOsZw9/o8joYA0WqbbGBQV9zdudDom7ywsLyLvLBgpxwsIJCPzN9zdRCvnHzQAJ/9L3M1YK88XP', false).play();
            
            // Optionally show notification via Notification API if supported
            if (Notification.permission === "granted") {{
                new Notification(title, {{ body: message }});
            }}
        }}
        
        // Request browser notification permission
        if (Notification.permission !== "granted" && Notification.permission !== "denied") {{
            Notification.requestPermission();
        }}
        
        // Listen for messages from BadBits
        window.addEventListener('message', function(event) {{
            const data = event.data;
            if (data.type === 'notification') {{
                addNotification(data.title, data.message, data.isAlert);
            }}
        }});
        
        // Log ready status
        console.log("BadBits notification system ready");
    </script>
</body>
</html>"""
        return html
    
    def browser_notification(self, title: str, message: str, is_alert: bool = True) -> bool:
        """
        Show notification in browser window.
        
        Args:
            title: Notification title
            message: Notification message
            is_alert: Whether to mark as critical alert with red styling
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create HTML file if not already created
            if self.notification_html is None:
                # Create temporary HTML file
                fd, path = tempfile.mkstemp(suffix='.html', prefix='badbits_notifications_')
                self.notification_html = path
                with open(path, 'w') as f:
                    f.write(self._create_notification_html())
                
                # Open browser window if not already open
                if self.browser_window is None:
                    # Open in new browser window
                    webbrowser.open('file://' + path, new=1)
                
            # Open file:// URL if we haven't opened it yet
            if self.browser_window is None:
                self.browser_window = True  # Mark as opened
            
            # Execute JavaScript to add notification
            # This won't work directly due to browser security, but we'll update the HTML
            # The notification will be shown when user refreshes page
            with open(self.notification_html, 'r') as f:
                content = f.read()
            
            # Add notification entry by modifying the HTML
            notification_entry = f"""
            <script>
                // Add notification on page load
                window.addEventListener('load', function() {{
                    addNotification("{title}", "{message}", {str(is_alert).lower()});
                }});
            </script>
            """
            modified_content = content.replace('</body>', notification_entry + '</body>')
            
            with open(self.notification_html, 'w') as f:
                f.write(modified_content)
                
            return True
                
        except Exception as e:
            logger.warning(f"Browser notification failed: {e}")
            return False
    
    def dramatic_alert(self, title: str, message: str) -> bool:
        """
        Display a dramatic full-screen alert that interrupts user workflow.
        
        This creates a temporary application window that takes over the screen,
        forcing the user to acknowledge the alert before continuing.
        
        Args:
            title: Alert title
            message: Alert message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create temporary HTML file with more dramatic styling
            fd, path = tempfile.mkstemp(suffix='.html', prefix='badbits_dramatic_alert_')
            
            # Full-screen dramatic alert HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: rgba(220, 53, 69, 0.95);
            color: white;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            animation: pulse 1.5s infinite;
        }}
        .container {{
            text-align: center;
            max-width: 80%;
            padding: 2rem;
            border-radius: 8px;
            background-color: rgba(0,0,0,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: shake 0.5s infinite;
        }}
        .message {{
            font-size: 1.8rem;
            margin-bottom: 2rem;
        }}
        .dismiss {{
            background-color: white;
            color: #dc3545;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.2rem;
            font-weight: bold;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}
        .dismiss:hover {{
            background-color: #f8f9fa;
            transform: scale(1.05);
        }}
        @keyframes pulse {{
            0% {{ background-color: rgba(220, 53, 69, 0.9); }}
            50% {{ background-color: rgba(220, 53, 69, 0.7); }}
            100% {{ background-color: rgba(220, 53, 69, 0.9); }}
        }}
        @keyframes shake {{
            0% {{ transform: translateX(0); }}
            25% {{ transform: translateX(-5px); }}
            50% {{ transform: translateX(0); }}
            75% {{ transform: translateX(5px); }}
            100% {{ transform: translateX(0); }}
        }}
        .emoji {{
            font-size: 5rem;
            margin-bottom: 1rem;
            animation: bounce 1s infinite;
        }}
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-20px); }}
        }}
        .countdown {{
            font-size: 1.5rem;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="emoji">⚠️</div>
        <h1>{title}</h1>
        <div class="message">{message}</div>
        <button class="dismiss" id="dismissBtn">I'll Fix This Now</button>
        <div class="countdown" id="countdown">Alert will close in 15 seconds</div>
    </div>
    
    <script>
        // Auto-dismiss after 15 seconds
        let secondsLeft = 15;
        const countdownEl = document.getElementById('countdown');
        const countdown = setInterval(() => {{
            secondsLeft--;
            countdownEl.textContent = `Alert will close in ${{secondsLeft}} seconds`;
            if (secondsLeft <= 0) {{
                clearInterval(countdown);
                window.close();
            }}
        }}, 1000);
        
        // Allow dismissing with the button, escape key, or space bar
        document.getElementById('dismissBtn').addEventListener('click', () => {{
            window.close();
        }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' || e.key === ' ' || e.key === 'Enter') {{
                window.close();
            }}
        }});
        
        // Make the alert more attention-grabbing
        const playSound = () => {{
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PurWcHENCa0evBhDEQKXrF5cqQRBMeSKjd+b97HwMvh9f0yH4+ChuBx+PMjkwXHFSs2se1RQxDn+DtwH0rnNJ8IQIwktvoq1kJBGjN9dO5Y0wZVbBS+tzZgAlQwHIaA36i1tuYNhBZtm4TEnnS775sRzFWqVoPaLj34IQM6p9cHTC07vKVGUCo3t+DSBxLr/THdxpDzpdTFWmo2/OgMg05074jETib6ueEJ+aWTxFrw/vieQgTgOMOMn3Y/K9RBliX7tgYAoTsLj2M1ueZKQNwu1ISWaf0xVwC0qlNFG3I/MhRA2Go99QPCvXqgRbVpMTSkigJSbCKbwAPecXw12YM3p9QFXfT/89PAl2m9r8cCPnlfQvosnnXiUAQZbODTgEUfc7yxlMF76RdHHHB+spLBYrg4Yk6AcyYSxmA1P3NTAFUvM4yV4Tj7GANDJrQ2pUzDF2iyQEzjPL5jzUFZrXnBFJ9++etKc70jSMsj++8LBgTpOL9ClB9/vemJd73myxMmOXOGhEis/v9LE0omsQ5S5fiyicEP7D//FFhHJG2Ognvw/c2VRScr6FpB3e1+fwuCNu4fXkPxqFsPA6MytajLQlZvILvD9rdsQ8PwbnB/kZAYqOsZw9/o8joYA0WqbbGBQV9zdudDom7ywsLyLvLBgpxwsIJCPzN9zdRCvnHzQAJ/9L3M1YK88XP');
            audio.play();
        }};
        
        // Play sound on load
        playSound();
        
        // Play sound every 3 seconds
        setInterval(playSound, 3000);
        
        // Vibrate if supported
        if ('vibrate' in navigator) {{
            navigator.vibrate([200, 100, 200, 100, 200]);
            // Repeat vibration pattern every 2 seconds
            setInterval(() => {{
                navigator.vibrate([200, 100, 200]);
            }}, 2000);
        }}
    </script>
</body>
</html>"""
            
            # Write to temp file
            with open(path, 'w') as f:
                f.write(html)
            
            # Open in browser with new window
            webbrowser.open('file://' + path, new=1)
            
            # Play system alert sound for additional attention
            self.sound_alert()
            
            return True
            
        except Exception as e:
            logger.warning(f"Dramatic alert failed: {e}")
            return False

    def send_alert(self, title: str, message: str, methods: List[str] = None) -> None:
        """
        Send alert using multiple methods with fallbacks.
        
        Args:
            title: Alert title
            message: Alert message
            methods: List of methods to try in order of preference:
                    - 'desktop': Desktop notification via plyer
                    - 'system': System dialog
                    - 'browser': Browser notification
                    - 'dramatic': Full-screen dramatic alert
                    - 'sound': Sound alert
        """
        if methods is None:
            methods = ['desktop', 'system', 'browser', 'sound']
        
        success = False
        
        # First try preferred methods in order
        for method in methods:
            if method == 'desktop':
                success = self.desktop_notification(title, message)
                if success:
                    break
            elif method == 'system':
                success = self.system_alert(title, message)
                if success:
                    break
            elif method == 'browser':
                success = self.browser_notification(title, message)
                if success:
                    break
            elif method == 'dramatic':
                success = self.dramatic_alert(title, message)
                if success:
                    break
            elif method == 'sound':
                success = self.sound_alert()
        
        # Always play sound unless we explicitly succeeded with sound method
        if 'sound' not in methods or not success:
            self.sound_alert()

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
            
            # Initialize alert manager
            self.alert_manager = AlertManager(app_name="BadBits")
            
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
                prompt="Compare the top (reference) and bottom images: Is the person in the bottom image sitting with worse posture than in the reference image? Answer with ONLY 'yes' or 'no'.",
                details_prompt=None,
                description="Detects poor sitting posture compared to your reference image",
                active_message="Poor posture detected! Straighten your back and adjust your position.",
                default_enabled=True
            ),
            "nail_biting": HabitCheck(
                habit_id="nail_biting",
                name="Nail Biting",
                emoji="💅",
                prompt="Looking at the bottom image only: Is the person biting their nails or have their hands near their mouth? Answer with ONLY 'yes' or 'no' - nothing else.",
                description="Detects nail biting or hands near mouth",
                active_message="Nail biting detected! Be mindful of your hands.",
                default_enabled=True
            ),
            "eye_strain": HabitCheck(
                habit_id="eye_strain",
                name="Eye Strain",
                emoji="👁️",
                prompt="Looking at the bottom image only: Is the person leaning too close to the screen (less than arm's length away)? Answer with ONLY 'yes' or 'no' - nothing else.",
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
                # Strictly enforce binary yes/no - only "yes" counts as positive
                is_active = answer.strip().lower() == "yes"
                
                # Log if we got unexpected response
                if answer.strip().lower() not in ["yes", "no"]:
                    logger.warning(f"Non-binary response from model for {habit_id}: '{answer}'. Treated as 'no'.")
                
                # No details needed - keep alerts simple and binary
                details = ""
                
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
        
        # Send a notification that monitoring is starting
        if hasattr(self, 'alert_manager'):
            self.alert_manager.send_alert(
                "BadBits Monitoring Started",
                "Posture and habit monitoring is now active!",
                ['desktop', 'system']
            )
            
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
        
        # Create the notification message - simple and direct
        if habit and habit.active_message:
            # Use the custom message from the habit definition, no details needed
            message = habit.active_message
        else:
            # Default message if habit is not found
            message = "Issue detected!"
        
        # Initialize alert manager if needed
        if not hasattr(self, 'alert_manager'):
            self.alert_manager = AlertManager(app_name="BadBits")
            
        # Send alert with fallbacks
        self.alert_manager.send_alert(
            title=title,
            message=message,
            methods=['desktop', 'system', 'browser', 'sound']
        )
    
    def render_dashboard(self, 
                         stats: CheckStats,
                         current_alerts: List[AlertResult], 
                         next_check_time: Optional[datetime] = None,
                         error_message: str = "") -> str:
        """
        Render a clean, minimal dashboard with timeline visualization.
        
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
        
        # Limited color palette for a clean, cohesive look
        class Colors:
            RESET = "\033[0m"
            BOLD = "\033[1m"
            DIM = "\033[2m"  # Dimmed text
            BLUE = "\033[34m"  # Standard blue (not bright)
            GREEN = "\033[32m"  # Standard green
            RED = "\033[31m"  # Standard red
            WHITE = "\033[37m"  # White
            
        # Clean header
        border = "─" * terminal_width
        title = f"{Colors.BOLD}BadBits Monitor{Colors.RESET}"
        subtitle = "Posture and habit tracking"
        header = [
            border,
            title.center(terminal_width),
            subtitle.center(terminal_width),
            border
        ]
        
        # Simple status line with live indicator
        current_time = datetime.now().strftime("%H:%M:%S")
        if stats.last_check_time:
            last_check = stats.last_check_time.strftime("%H:%M:%S")
        else:
            last_check = "--:--:--"
            
        # Add a live indicator
        live_indicator = f"{Colors.GREEN}● LIVE{Colors.RESET}"
            
        status_line = f"{live_indicator} │ Session: {Colors.BOLD}{stats.duration_minutes}m{Colors.RESET} │ Checks: {Colors.BOLD}{stats.total_checks}{Colors.RESET} │ Now: {current_time} │ Last: {last_check}"
        
        # Combined section for habits - status and history together
        habits_section = [
            f"{Colors.BOLD}Habit Monitoring{Colors.RESET}",
            "┄" * terminal_width
        ]
        
        # Map alerts to habits for easier lookup
        alert_by_habit = {}
        for alert in current_alerts:
            alert_by_habit[alert.alert_type] = alert
            
        # Get all active habits
        active_habits = [habit_id for habit_id, habit in self.habits.items() if habit.enabled]
        
        if not active_habits:
            habits_section.append("No habits being monitored")
        elif stats.total_checks == 0:
            habits_section.append("Monitoring started - waiting for first check")
        else:
            # Show each habit with all its information in one unified display
            max_checks = min(20, stats.total_checks)  # Show up to last 20 checks
            bar_width = min(terminal_width - 40, 25)  # Smaller bar to fit everything
            
            for habit_id in active_habits:
                habit = self.habits[habit_id]
                percent = stats.get_alert_percent(habit_id)
                
                # Current status indicator
                current_alert = alert_by_habit.get(habit_id)
                if current_alert and current_alert.is_active:
                    status_indicator = f"{Colors.RED}! NEEDS ATTENTION{Colors.RESET}"
                else:
                    status_indicator = f"{Colors.GREEN}✓ Good{Colors.RESET}"
                
                # Display the habit name and current status
                habits_section.append(f"\n{habit.emoji}  {Colors.BOLD}{habit.get_display_name()}{Colors.RESET}  {status_indicator}")
                
                # Create percentage bar
                if percent > 50:
                    percent_color = Colors.RED
                elif percent > 25:
                    percent_color = Colors.RED + Colors.DIM
                else:
                    percent_color = Colors.GREEN
                
                fill_width = int((percent / 100) * bar_width)
                bar = f"[{percent_color}{'■' * fill_width}{Colors.RESET}{Colors.DIM}{'·' * (bar_width - fill_width)}{Colors.RESET}]"
                
                # Add the percentage summary
                habits_section.append(f"   Session issues: {percent_color}{percent:2d}%{Colors.RESET} {bar}")
                
                # Create timeline visualization - always show for all enabled habits
                # Get alert count (default to 0 if not found)
                alerts_count = stats.habit_alerts.get(habit_id, 0)
                alerts_per_check = alerts_count / stats.total_checks if stats.total_checks > 0 else 0
                
                # Build timeline string
                timeline = ""
                
                for i in range(max_checks):
                    check_index = stats.total_checks - max_checks + i
                    
                    # Determine if this check had an issue (approximation)
                    # For habits with no issues recorded, show all green dots
                    if alerts_count > 0:
                        is_active = (check_index % 3 == 0 and alerts_per_check > 0.3) or \
                                    (check_index % 7 == 0 and alerts_per_check > 0.1) or \
                                    (alerts_per_check > 0.5 and check_index % 2 == 0)
                    else:
                        is_active = False
                    
                    if is_active:
                        timeline += f"{Colors.RED}×{Colors.RESET}"
                    else:
                        timeline += f"{Colors.GREEN}·{Colors.RESET}"
                
                # Add timeline with label
                habits_section.append(f"   History: [{timeline}] (oldest → newest)")
            
            # Add time reference indicator at the bottom
            habits_section.append("")
            time_reference = f"{Colors.DIM}   Start: {stats.start_time.strftime('%H:%M')}  →  Now: {datetime.now().strftime('%H:%M')}{Colors.RESET}"
            habits_section.append(time_reference)
            
        # Error message if any
        error_lines = []
        if error_message:
            error_lines = [
                "┄" * terminal_width,
                f"{Colors.RED}Error: {error_message}{Colors.RESET}",
                "┄" * terminal_width
            ]
            
        # Footer
        footer = [
            border,
            f"Press Ctrl+C to exit".center(terminal_width),
            border
        ]
        
        # Combine all sections
        dashboard = []
        dashboard.extend(header)
        dashboard.append("")
        dashboard.append(status_line.center(terminal_width))
        dashboard.append("")
        dashboard.extend(habits_section)
        
        if error_lines:
            dashboard.append("")
            dashboard.extend(error_lines)
            
        dashboard.append("")
        dashboard.extend(footer)
        
        return "\n".join(dashboard)
                    
    def run_continuous_monitoring(self, interval_seconds: int = 60, 
                                  notification_enabled: bool = True,
                                  archive_mode: bool = False,
                                  dashboard_mode: bool = True,
                                  alert_methods: List[str] = None):
        """
        Run continuous posture monitoring with the specified interval.
        
        Args:
            interval_seconds: Time between checks in seconds
            notification_enabled: Whether to enable notifications
            archive_mode: Whether to save check data to disk (privacy)
            dashboard_mode: Whether to use dashboard UI mode
            alert_methods: List of alert methods to use in priority order
                (desktop, system, browser, sound)
            
        Raises:
            Exception: If monitoring fails
        """
        # Set default alert methods if not provided
        if alert_methods is None:
            alert_methods = ['desktop', 'system', 'browser', 'sound']
        try:
            # First capture reference image
            self.capture_reference()
            
            # Show startup notification (try system specifically first, as it's most reliable)
            if notification_enabled:
                try:
                    # Try direct system notification first (most reliable)
                    self.alert_manager.system_alert(
                        "BadBits Monitoring Started",
                        "Posture and habit monitoring is now active!"
                    )
                except Exception:
                    # Fallback to regular alert flow
                    self.alert_manager.send_alert(
                        "BadBits Monitoring Started", 
                        "Posture and habit monitoring is now active!",
                        methods=alert_methods
                    )
            
            logger.info("Starting continuous posture monitoring...")
            logger.info("Press Ctrl+C to stop")
            
            # Initialize session tracking
            stats = CheckStats()
            last_alerts: List[AlertResult] = []
            error_message = ""
            
            # Initial banner - show differently based on mode
            if dashboard_mode:
                # For dashboard mode, display an initial dashboard immediately
                # Clear screen
                if platform.system() != "Windows":
                    os.system('clear')
                else:
                    os.system('cls')
                
                # Create initial empty alerts for display
                initial_alerts = []
                for habit_id, habit in self.habits.items():
                    if habit.enabled:
                        # Add a placeholder result for each enabled habit
                        initial_alerts.append(AlertResult(
                            alert_type=habit_id,
                            is_active=False,  # Start with "good" status
                            details="",
                            timestamp=datetime.now()
                        ))
                
                # Render and display initial dashboard
                initial_dashboard = self.render_dashboard(
                    stats=stats,
                    current_alerts=initial_alerts,
                    next_check_time=datetime.now() + timedelta(seconds=interval_seconds),
                    error_message=""
                )
                print(initial_dashboard)
            else:
                # Text-based banner for non-dashboard mode
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
                            
                            # Send alert using the specified methods
                            self.alert_manager.send_alert(
                                title=title,
                                message=message,
                                methods=alert_methods
                            )
                    
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
            # Final summary on exit - using dashboard style
            if dashboard_mode:
                # Clear screen once more for final message
                if platform.system() != "Windows":
                    os.system('clear')
                else:
                    os.system('cls')
                
                # Limited color palette for a clean, cohesive look - same as dashboard
                class Colors:
                    RESET = "\033[0m"
                    BOLD = "\033[1m"
                    DIM = "\033[2m"  # Dimmed text
                    BLUE = "\033[34m"  # Standard blue (not bright)
                    GREEN = "\033[32m"  # Standard green
                    RED = "\033[31m"  # Standard red
                    WHITE = "\033[37m"  # White
                
                # Get terminal size
                terminal_width = shutil.get_terminal_size().columns
                
                # Create a styled end message, consistent with dashboard
                border = "─" * terminal_width
                title = f"{Colors.BOLD}BadBits Monitor - Session Complete{Colors.RESET}"
                
                # The counterpart to LIVE indicator - show COMPLETE
                end_indicator = f"{Colors.RED}● COMPLETE{Colors.RESET}"
                
                # Session statistics in dashboard style
                status_line = f"{end_indicator} │ Session: {Colors.BOLD}{stats.duration_minutes}m{Colors.RESET} │ Checks: {Colors.BOLD}{stats.total_checks}{Colors.RESET} │ End time: {datetime.now().strftime('%H:%M:%S')}"
                
                # Format summary header like dashboard
                summary_header = f"{Colors.BOLD}Session Summary{Colors.RESET}"
                
                # Start building the output
                lines = [
                    border,
                    title.center(terminal_width),
                    border,
                    "",
                    status_line.center(terminal_width),
                    "",
                    summary_header,
                    "┄" * terminal_width
                ]
                
                # Add stats for each habit in dashboard style
                for habit_id, habit in self.habits.items():
                    if habit.enabled:
                        alert_count = stats.habit_alerts.get(habit_id, 0)
                        percent = stats.get_alert_percent(habit_id)
                        
                        # Determine color based on percentage
                        if percent > 50:
                            percent_color = Colors.RED
                        elif percent > 25:
                            percent_color = Colors.RED + Colors.DIM
                        else:
                            percent_color = Colors.GREEN
                        
                        # Create summary line in dashboard style
                        habit_line = f"{habit.emoji}  {habit.get_display_name()}: {alert_count}/{stats.total_checks} checks ({percent_color}{percent}%{Colors.RESET})"
                        lines.append(habit_line)
                
                # Add data storage info if applicable
                if archive_mode:
                    lines.append("")
                    lines.append(f"📊 Analysis data saved to: {self.output_dir}")
                
                # Add footer
                lines.append("")
                lines.append(border)
                lines.append(f"Thanks for using BadBits!".center(terminal_width))
                lines.append(border)
                
                # Print the styled summary
                print("\n".join(lines))
            
            else:
                # Simple text version for non-dashboard mode
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
    
    # Alert options - simplified
    alert_group = parser.add_argument_group('Alert Style')
    
    alert_style = alert_group.add_mutually_exclusive_group()
    
    alert_style.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable all notifications"
    )
    
    alert_style.add_argument(
        "--normal",
        action="store_true",
        default=True,
        help="Standard desktop notifications (default)"
    )
    
    alert_style.add_argument(
        "--loud", "-l",
        action="store_true",
        help="Attention-grabbing full-screen alerts"
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
    
    # Storage options
    parser.add_argument(
        "--track", "-t",
        action="store_true",
        help="Save images and analysis data to disk for tracking progress"
    )
    
    parser.add_argument(
        "--habits",
        type=str,
        help="Path to a JSON file with custom habit definitions"
    )
    
    parser.add_argument(
        "--save-habits",
        type=str,
        help="Save current habit definitions to a JSON file"
    )
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    
    display_style = display_group.add_mutually_exclusive_group()
    
    display_style.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Use simple text output instead of dashboard"
    )
    
    display_style.add_argument(
        "--dashboard",
        action="store_true",
        default=True,
        help="Show interactive dashboard (default)"
    )
    
    # Habit monitoring options - simple focused design
    habit_group = parser.add_argument_group('What to Monitor')
    
    habit_mode = habit_group.add_mutually_exclusive_group()
    
    habit_mode.add_argument(
        "--all", "-a", 
        action="store_true",
        default=True,
        help="Monitor both posture and nail biting (default)"
    )
    
    habit_mode.add_argument(
        "--posture-only", "-p", 
        action="store_true",
        help="Monitor posture only"
    )
    
    habit_mode.add_argument(
        "--nails-only", "-n", 
        action="store_true",
        help="Monitor nail biting only"
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
        
        # Set which habits to monitor based on command-line options
        if args.posture_only:
            # Enable only posture monitoring
            monitor.enable_habit("posture", True)
            monitor.enable_habit("nail_biting", False)
        elif args.nails_only:
            # Enable only nail biting detection
            monitor.enable_habit("posture", False)
            monitor.enable_habit("nail_biting", True)
        else:
            # Default: monitor both
            monitor.enable_habit("posture", True)
            monitor.enable_habit("nail_biting", True)
            
        # All other habits are disabled by default
        monitor.enable_habit("eye_strain", False)
        monitor.enable_habit("screen_break", False)
        
        # Set alert methods based on command-line options
        if args.loud:
            # Use dramatic full-screen alerts
            alert_methods = ['dramatic', 'system', 'desktop', 'sound']
        elif args.quiet:
            # Disable notifications
            alert_methods = []
        else:
            # Default to standard notifications
            alert_methods = ['desktop', 'system', 'sound']
        
        # Start monitoring with selected options
        monitor.run_continuous_monitoring(
            interval_seconds=args.interval,
            notification_enabled=not args.quiet,
            archive_mode=args.track,
            dashboard_mode=not args.simple,
            alert_methods=alert_methods
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