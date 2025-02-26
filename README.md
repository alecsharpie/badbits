# BadBits

> AI-powered posture coach and nail-biting detector

BadBits uses your webcam and AI to help you maintain good posture and break the nail-biting habit. It provides real-time monitoring with a clean dashboard interface and helpful alerts.

## Quick Setup

```bash
# Install dependencies
pip install -e .

# Run with default settings (monitors both posture and nail-biting)
python badbits.py
```

## Common Commands

BadBits is designed to be simple with sensible defaults. Here are the most common ways to use it:

```bash
# Monitor both posture and nail-biting (default)
python badbits.py

# Monitor posture only
python badbits.py --posture-only

# Monitor nail-biting only
python badbits.py --nails-only

# Use attention-grabbing alerts (for breaking stubborn habits)
python badbits.py --loud

# Run silently (dashboard only, no alerts)
python badbits.py --quiet

# Save progress data for review
python badbits.py --track
```

## Features

- **Focused Monitoring**: Tracks posture and nail-biting habits
- **Clean Dashboard**: Real-time visualization of your habits
- **Smart Alerts**: Notifications when issues are detected
- **Privacy-First**: No images saved to disk by default
- **Progress Tracking**: Optional data saving for reviewing improvement

## Command Options

BadBits has a simplified command interface focused on the essentials:

### What to Monitor

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--all` | `-a` | Monitor both posture and nail-biting | ✓ |
| `--posture-only` | `-p` | Monitor posture only | |
| `--nails-only` | `-n` | Monitor nail-biting only | |

### Alert Style

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--normal` | | Standard desktop notifications | ✓ |
| `--quiet` | `-q` | Disable all notifications | |
| `--loud` | `-l` | Attention-grabbing full-screen alerts | |

### Display Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--dashboard` | | Show interactive dashboard | ✓ |
| `--simple` | `-s` | Use simple text output | |

### Other Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--interval` | `-i` | Seconds between checks | 60 |
| `--camera` | `-c` | Camera device ID | 0 |
| `--track` | `-t` | Save data for progress tracking | |
| `--backup-cameras` | | Fallback cameras (comma-separated) | |

## Platform Notes

### macOS
For better desktop notifications on macOS:
```bash
pip install pyobjus
```

### All Platforms
BadBits works on Windows, macOS, and Linux with no additional configuration required.

## How It Works

1. **Reference Image**: First, you capture a reference image of your ideal posture
2. **Continuous Monitoring**: BadBits regularly checks your posture and hand position
3. **Smart Detection**: AI compares current posture with reference image
4. **Helpful Feedback**: Notifications alert you when issues are detected
5. **Progress Visualization**: Dashboard shows improvement over time

## Development

```bash
# Install development dependencies
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