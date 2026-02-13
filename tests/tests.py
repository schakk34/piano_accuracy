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
import math
import os
import subprocess

AVAILABLE_TESTS_JSON = "piano-synth/available_tests.json"
TOLERANCE = 0.05
STATE_BOUNDARY_TOLERANCE = 3

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

def extract_note_arrays(test_case):
    """
    Extracts frequency, duration, and note name arrays from a test case dictionary.
    Returns: (frequencies, durations, note_names)
    Each is a list of lists (one per track).
    """
    tracks = test_case.get('tracks', [])
    frequencies = []
    durations = []
    note_names = []

    for track in tracks:
        track_freqs = []
        track_durs = []
        track_names = []
        for note in track:
            freq = note.get('frequency', 0.0)
            if freq > 0:
                midi_note = int(round(69 + 12 * math.log2(freq / 440.0)))
            else:
                midi_note = 0
            track_freqs.append(midi_note)
            track_durs.append(note.get('duration', 0.0))
            track_names.append(note.get('name', ''))

        frequencies.append(track_freqs)
        durations.append(track_durs)
        note_names.append(track_names)

    return frequencies, durations, note_names

def get_note_separation_ground_truth(test_case, sample_rate=22050, hop_length=512):
    """
    Generates ground truth for note separation.
    Treats each note in the first track as a state.

    Returns:
        state_frames: List of state indices for each frame.
        boundary_frames: List of frame indices where notes change.
    """
    _, durations, _ = extract_note_arrays(test_case)

    if not durations:
        return [], []

    # Use first track
    track_durs = durations[0]

    state_frames = []
    boundary_frames = []

    current_time = 0.0

    for i, dur in enumerate(track_durs):
        start_frame = int((current_time * sample_rate) / hop_length)
        boundary_frames.append(start_frame)

        current_time += dur
        end_frame = int((current_time * sample_rate) / hop_length)

        # Fill state frames
        current_frame_count = len(state_frames)
        if end_frame > current_frame_count:
            state_frames.extend([i] * (end_frame - current_frame_count))

    # Add final boundary
    boundary_frames.append(len(state_frames))

    return state_frames, boundary_frames

def run_score_tests(algo):
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

def run_note_seperation_tests(algo, sr=22050, hop_length=512):
    """
    Runs note separation tests.
    algo: A function that takes (filepath, frequencies, durations) and returns (state_frames, boundary_frames).
    """
    tests = get_available_tests()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "piano-synth", "target_music")

    print(f"Starting execution of {len(tests)} note separation tests...")

    for test in tests:
        filename = test['filename']
        filename = filename.replace(".wav", ".mp3")
        filepath = os.path.join(audio_dir, filename)

        if not os.path.exists(filepath):
            print(f"Missing audio files. Attempting to run conversion script...")
            convert_script = os.path.join(script_dir, "piano-synth", "src", "convert_to_mp3.bash")

            try:
                subprocess.check_call(["bash", convert_script])
            except subprocess.CalledProcessError:
                print("ERROR: Audio conversion failed. Ensure ffmpeg is installed.")
                return

            if not os.path.exists(filepath):
                print(f"ERROR: File {filename} still not found after conversion attempts.")
                return

        try:
            print(f"Testing {filename}...")

            # Get Ground Truth
            frequencies, durations, _ = extract_note_arrays(test)
            gt_states, gt_boundaries = get_note_separation_ground_truth(test, sr, hop_length)

            # Run Algo
            actual_states, actual_boundaries = algo(filepath, frequencies[0], durations[0], sr, hop_length)

            # Compare States (Accuracy)
            # We compare against the length of the ground truth
            matches = 0
            if gt_states:
                min_len = min(len(gt_states), len(actual_states))
                matches = sum(1 for i in range(min_len) if gt_states[i] == actual_states[i])
                state_acc = matches / len(gt_states)
            else:
                state_acc = 1.0 if not actual_states else 0.0

            # Compare Boundaries (Recall with tolerance of 3 frames)
            matched_boundaries = 0
            if gt_boundaries:
                for gt_b in gt_boundaries:
                    if any(abs(gt_b - act_b) <= STATE_BOUNDARY_TOLERANCE for act_b in actual_boundaries):
                        matched_boundaries += 1
                boundary_recall = matched_boundaries / len(gt_boundaries)
            else:
                boundary_recall = 1.0 if not actual_boundaries else 0.0

            print(f"  State Accuracy: {state_acc:.2%}")
            print(f"  Boundary Recall: {boundary_recall:.2%} (Found {len(actual_boundaries)}/{len(gt_boundaries)})")

        except Exception as e:
            print(f"  ERROR: Exception during test execution: {e}")

    print("Note Separation Tests Completed.")
