#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}"
echo -e "+-----------------------------------------+"
echo -e "|      BadBits Model Downloader           |"
echo -e "+-----------------------------------------+${NC}"
echo

# URL of the file to download
URL="https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"

# Output directories and files
MODEL_DIR="models"
OUTPUT_FILE="$MODEL_DIR/moondream-2b-int8.mf.gz"
FINAL_FILE="$MODEL_DIR/moondream-2b-int8.mf"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if the final file already exists
if [ -f "$FINAL_FILE" ]; then
    echo -e "${YELLOW}Model already exists at $FINAL_FILE${NC}"
    echo -e "To re-download, delete the existing file first."
    exit 0
fi

echo -e "Downloading Moondream model..."
echo -e "This may take a few minutes depending on your internet speed."
echo

# Download the file with progress bar
if command -v wget > /dev/null; then
    wget --progress=bar:force -O "$OUTPUT_FILE" "$URL" || { echo -e "${RED}Download failed!${NC}"; exit 1; }
else
    curl -L --progress-bar -o "$OUTPUT_FILE" "$URL" || { echo -e "${RED}Download failed!${NC}"; exit 1; }
fi

echo -e "${GREEN}Download complete!${NC}"
echo -e "Decompressing model file..."

# Unzip the file
gzip -d "$OUTPUT_FILE" || { echo -e "${RED}Decompression failed!${NC}"; exit 1; }

echo -e "${GREEN}Model ready at $FINAL_FILE${NC}"
echo -e "You can now run the application with: python badbits.py"