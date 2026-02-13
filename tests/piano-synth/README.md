# Piano synthesis for algorithm testing

## Overview
This directory contains the Rust source to synthesize piano songs using Twang.

## Test usage
### Test Accuracy Scores

In the relavent notebook, create a function that follows this signature (the actual name of the method is not relavent and only needs to be consistent with the next step):
```python
# Create a unified function to do everything and output a score
# Parameters:
#   ideal_fp: the filepath of the ideal song, in mp3
#   prac_fp: the filepath of the "test" or "practice" song, in mp3
# Returns: (pitch_accuracy, tempo_accuracy)
# where the accuracy values are within the range [0.0, 1.0].
def analyze_accuracy(ideal_fp, prac_fp):
```

Then, create a new cell and run the following:
```python
# Run unit tests
from tests import tests
tests.run_all_tests(analyze_accuracy)
```
The testing code should run and print your test results. The defualt tolerance is to be within 0.05 of the target values.

The target values are a guess based on the errors introduced into the files, and may not be fully correct.

### Test Note Seperation 
Create a unified method that follows this signature:
```python 
states, boundaries = algo(filepath, frequencies, durations, sr, hop_length)
```
Where:
- `states`: Array with `len(states) == len(frames)`, with each value being equal to the state. Ie, the `path` from the Viterbi decoded state vs time graphs.
- `boundries`: Array of frame numbers that correspond to note changes, ie, the `boundry_frames` from the spectograms.

## Objectives
- [ ] Parse MusicXML to capture notes, durations, etc, and turn that into a list of notes and their durations for synthesis with Twang and as the HMM model. Theoretically this will enable us to parse any sheet music and automatically generate test cases, an ideal version of the song, and the weights of an HMM model to analyze the end user's performance.
- [x] Use Twang to synthesize piano music.
- [ ] Convert the generated .wav's to .mp3.
- [ ] Generate a JSON file detailing the available music tracks.
- [x] Write the Python harness to read the JSON, grab the available tracks, and run them through the model (the actual unit testing harness part)
