#!/bin/bash

# Navigate to the parent directory (project root) where the WAV files are located
# This assumes the script is inside 'src/'
cd "$(dirname "$0")/../target_music/" || exit 1

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    exit 1
fi

echo "Converting WAV files to MP3..."

# Loop through all .wav files in the current directory
for wav_file in *.wav; do
    # Check if file exists to handle case where no .wav files are found
    if [ -e "$wav_file" ]; then
        # Get filename without extension
        filename="${wav_file%.*}"

        echo "Converting: $wav_file -> $filename.mp3"

        # Convert using ffmpeg
        # -y: Overwrite output files
        # -codec:a libmp3lame: Use LAME encoder
        # -qscale:a 2: Variable bit rate (High quality, roughly 190kbps)
        ffmpeg -y -i "$wav_file" -codec:a libmp3lame -qscale:a 2 "$filename.mp3" < /dev/null
    fi
done

echo "All conversions complete!"
