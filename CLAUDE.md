# BadBits Project Guide

## Commands
- Run application (default dashboard): `python badbits.py`
- Run with tracking mode: `python badbits.py --track`
- Run with simple interface: `python badbits.py --simple`
- Run with backup cameras: `python badbits.py --camera 0 --backup-cameras "1,2"`
- Run with custom interval: `python badbits.py --interval 30`
- Run without notifications: `python badbits.py --no-alerts`
- Download model only: `python badbits.py --download-only`
- Run with quiet output: `python badbits.py --quiet`
- Type check: `mypy .`
- Lint code: `ruff check .`
- Format code: `black .`
- Install dev dependencies: `pip install -e ".[dev]"`

## Code Style Guidelines
- Python 3.12+ with strict type annotations
- Follow PEP 8 conventions with 100 character line limit
- Use snake_case for variables/functions, CamelCase for classes
- Import order: standard → third-party → local (grouped by type)
- Error handling: use specific exceptions with descriptive messages
- Documentation: Google-style docstrings with Args/Returns/Raises
- Use f-strings for string formatting
- Store analysis results in timestamped directories
- Strong input validation and defensive programming

## Architecture
- CV2 for webcam capture
- PIL for image processing
- Moondream vision model for image analysis
- Command-line interface with argparse
- Process flow: capture → analyze → display → store
- Isolation of concerns with class-based design