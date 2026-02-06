# Python scripts for unit testing using pre-generated music files.
# Jackson Holbrook, Feb 2026

# We want to be testing four things:
# - Pitch accuracy score
# - Pitch accuracy per-frame, to make sure the correct note is caught
# - Tempo accuracy score
# - Tempo accuracy frames
#
# The source of truth for the scores and the frames needs to be generated
# along with the music in the Rust component in order for it to make sense.

import json
import os
import subprocess

AVAILABLE_TESTS_JSON = "piano-synth/available_tests.json"
TOLERANCE = 0.05

# Use the hardcoded JSON file to see what tests are available.
# Returns a list of test case dictionaries
def get_available_tests():
    # Resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, AVAILABLE_TESTS_JSON)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find test configuration at {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)

def run_all_tests(algo):
    """
    Runs the provided algo function against all available tests.

    algo: A function that takes parameters (ideal_filepath, test_filepath) and
          returns a result containing pitch and tempo accuracy.
          Expected return format: dict with keys 'pitch_accuracy', 'tempo_accuracy'
          OR a tuple (pitch_accuracy, tempo_accuracy).
    """
    tests = get_available_tests()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Audio files are in piano-synth/target_music/
    audio_dir = os.path.join(script_dir, "piano-synth", "target_music")

    passed_count = 0
    total_count = len(tests)
    # TOLERANCE = 0.05  # Allow for small floating point differences

    print(f"Starting execution of {total_count} tests...")

    for test in tests:
        filename = test['filename']
        ideal_filename = test['ideal_filename']
        expected_pitch = test['expected_pitch_accuracy']
        expected_tempo = test['expected_tempo_accuracy']

        filename = filename.replace(".wav", ".mp3")
        ideal_filename = ideal_filename.replace(".wav", ".mp3")

        filepath = os.path.join(audio_dir, filename)
        ideal_filepath = os.path.join(audio_dir, ideal_filename)

        if not os.path.exists(filepath) or not os.path.exists(ideal_filepath):
            print(f"Missing audio files. Attempting to run conversion script...")
            convert_script = os.path.join(script_dir, "piano-synth", "src", "convert_to_mp3.bash")

            try:
                subprocess.check_call(["bash", convert_script])
            except subprocess.CalledProcessError:
                print("ERROR: Audio conversion failed. Ensure ffmpeg is installed.")
                return

            if not os.path.exists(filepath) or not os.path.exists(ideal_filepath):
                print(f"ERROR: Files {filename} or {ideal_filename} still not found after conversion attempts.")
                return

        try:
            print(f"Testing {filename}...")
            result = algo(ideal_filepath, filepath)

            # Normalize result
            actual_pitch = 0.0
            actual_tempo = 0.0

            if isinstance(result, dict):
                actual_pitch = result.get('pitch_accuracy', 0.0)
                actual_tempo = result.get('tempo_accuracy', 0.0)
            elif isinstance(result, (tuple, list)) and len(result) >= 2:
                actual_pitch = result[0]
                actual_tempo = result[1]

            # Verify accuracy
            pitch_pass = abs(actual_pitch - expected_pitch) <= TOLERANCE
            tempo_pass = abs(actual_tempo - expected_tempo) <= TOLERANCE

            if pitch_pass and tempo_pass:
                print("  PASS")
                passed_count += 1
            else:
                print("  FAIL")
                if not pitch_pass:
                    print(f"    Pitch Accuracy: Expected {expected_pitch}, Got {actual_pitch}")
                if not tempo_pass:
                    print(f"    Tempo Accuracy: Expected {expected_tempo}, Got {actual_tempo}")

        except Exception as e:
            print(f"  ERROR: Exception during test execution: {e}")

    print(f"Test Suite Completed: {passed_count}/{total_count} passed.")
