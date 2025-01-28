#!/bin/bash

# URL of the file to download
URL="https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"

# Output file name
OUTPUT_FILE="moondream-2b-int8.mf.gz"

# Download the file
curl -L -o $OUTPUT_FILE $URL

# Unzip the file
gzip -d $OUTPUT_FILE