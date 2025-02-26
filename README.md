# BadBits

> AI-powered posture coach and habit monitor that protects your health while you work

BadBits uses computer vision and AI to help you maintain good posture and break bad habits like nail-biting, providing gentle reminders when you need them most.

## Features

- 🔍 **Smart Detection**: Analyzes your posture and detects nail-biting in real-time
- 🔒 **Privacy-First**: No data saved to disk by default
- 🖥️ **Live Dashboard**: Visual feedback on your current habits and improvement over time
- 🚨 **Gentle Reminders**: Desktop notifications when issues are detected
- 🔄 **Resilient**: Works with multiple camera setups and recovers from disconnections

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run with recommended settings
python badbits.py

# Run with data tracking (saves images)
python badbits.py --track
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
| `--no-alerts` | `-n` | Disable desktop notifications | False |
| `--simple` | `-s` | Use simple console output (no dashboard) | False |
| `--download-only` | `-d` | Just download the model | False |

## How It Works

1. You capture a reference image of your ideal posture
2. BadBits monitors your posture and hand position at regular intervals
3. When issues are detected, you receive a notification
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