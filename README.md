# BadBits

> AI-powered posture coach and habit monitor that protects your health while you work

BadBits uses computer vision and AI to help you maintain good posture and break bad habits like nail-biting, providing gentle reminders when you need them most.

## Features

- 🔍 **Smart Detection**: Analyzes your posture and detects nail-biting in real-time
- 🔒 **Privacy-First**: No data saved to disk by default
- 🖥️ **Live Dashboard**: Visual feedback on your current habits and improvement over time
- 🚨 **Smart Alerts**: Comprehensive notification system with desktop, system, browser, and sound alerts
- 🔄 **Resilient**: Works with multiple camera setups and recovers from disconnections

## Quick Start

```bash
# Install dependencies
pip install -e .

# For macOS users, to enable desktop notifications
pip install pyobjus

# Run with recommended settings
python badbits.py

# Run with data tracking (saves images)
python badbits.py --track

# Run with specific alert methods
python badbits.py --alert-methods=system,browser,sound
```

## Usage Modes

BadBits has two main modes:

### 1. Privacy Mode (Default)
- No images saved to disk
- Real-time monitoring and alerts
- Dashboard interface

```bash
python badbits.py
```

### 2. Tracking Mode
- Saves reference and comparison images
- Tracks progress over time
- Useful for analyzing patterns

```bash
python badbits.py --track
```

## Command Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--interval` | `-i` | Seconds between checks | 60 |
| `--camera` | `-c` | Primary camera ID | 0 |
| `--backup-cameras` | | Fallback cameras (comma-separated) | None |
| `--track` | `-t` | Save images and analysis data | False |
| `--quiet` | `-q` | Reduce console output | False |
| `--no-alerts` | `-n` | Disable all notifications | False |
| `--alert-methods` | | Notification types in priority order | desktop,system,browser,sound |
| `--dramatic-alerts` | | Use attention-demanding full-screen alerts | False |
| `--simple` | `-s` | Use simple console output (no dashboard) | False |
| `--download-only` | `-d` | Just download the model | False |

## Notification System

BadBits features a comprehensive notification system with multiple fallback methods:

1. **Desktop Notifications**: Native OS notifications (requires pyobjus on macOS)
2. **System Alerts**: OS-specific alerts using:
   - macOS: AppleScript or terminal-notifier
   - Linux: notify-send or zenity
   - Windows: PowerShell or msg command
3. **Browser Notifications**: Popup alerts in a browser window
4. **Dramatic Alerts**: Full-screen attention-demanding notifications that must be acknowledged
5. **Sound Alerts**: Audio cues for immediate attention

Customize using the `--alert-methods` option:

```bash
# Prioritize system alerts over desktop notifications
python badbits.py --alert-methods=system,desktop,sound,browser

# Only use browser notifications and sounds
python badbits.py --alert-methods=browser,sound

# Use dramatic full-screen alerts that demand attention
python badbits.py --dramatic-alerts

# Specify dramatic alerts in priority chain
python badbits.py --alert-methods=dramatic,desktop,sound
```

### Platform-Specific Setup

**macOS**:
- For desktop notifications: `pip install pyobjus`
- For terminal alerts: `brew install terminal-notifier`

**Linux**:
- For desktop notifications: Ensure you have notify-send or zenity installed
  ```bash
  sudo apt-get install libnotify-bin zenity
  ```

**Windows**:
- No additional setup required

## How It Works

1. You capture a reference image of your ideal posture
2. BadBits monitors your posture and hand position at regular intervals
3. When issues are detected, you receive notifications through multiple channels
4. The dashboard shows your improvement over time

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Type checking
mypy .

# Linting
ruff check .

# Formatting
black .
```

## License

MIT