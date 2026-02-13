use fon::chan::{Ch16, Ch32};
use fon::{Audio, Frame};
use twang::Synth;
use twang::osc::Sine;

mod wav;

/// First ten harmonic volumes of a piano sample.
const HARMONICS: [f32; 10] = [
    0.700, 0.243, 0.229, 0.095, 0.139, 0.087, 0.288, 0.199, 0.124, 0.090,
];

// Frequencies
const E2: f32 = 82.41;
const A2: f32 = 110.00;
const C3: f32 = 130.81;
const E3: f32 = 164.81;
const G3: f32 = 196.00;
const G_SHARP_3: f32 = 207.65;
const A3: f32 = 220.00;
const C4: f32 = 261.63;
const D4: f32 = 293.66;
const E4: f32 = 329.63;
const F4: f32 = 349.23;
const G4: f32 = 392.00;
const G_SHARP_4: f32 = 415.30;
const A4: f32 = 440.00;
const B4: f32 = 493.88;
const C5: f32 = 523.25;
const D5: f32 = 587.33;
const D_SHARP_5: f32 = 622.25;
const E5: f32 = 659.25;

// Note duration in seconds
const S: f32 = 0.22; // Sixteenth note
const E: f32 = 0.44; // Eighth note

// Fur Elise Main Theme
const FUR_ELISE: &[(f32, f32)] = &[
    // Phrase 1
    (E5, S), (D_SHARP_5, S), (E5, S), (D_SHARP_5, S), (E5, S), (B4, S), (D5, S), (C5, S), (A4, E),
    (C4, S), (E4, S), (A4, S), (B4, E),
    (E4, S), (G_SHARP_4, S), (B4, S), (C5, E),
    (E4, S),
    // Phrase 1 Repeat (Variation at end)
    (E5, S), (D_SHARP_5, S), (E5, S), (D_SHARP_5, S), (E5, S), (B4, S), (D5, S), (C5, S), (A4, E),
    (C4, S), (E4, S), (A4, S), (B4, E),
    (E4, S), (C5, S), (B4, S), (A4, E),
];

const ODE_TO_JOY: &[(f32, f32)] = &[
    (E4, E), (E4, E), (F4, E), (G4, E), (G4, E), (F4, E), (E4, E), (D4, E),
    (C4, E), (C4, E), (D4, E), (E4, E), (E4, 0.66), (D4, 0.22), (D4, 0.88),
];

const ODE_TO_JOY_HARMONY: &[(f32, f32)] = &[
    (C3, E * 4.0), (G3, E * 4.0), (C3, E * 4.0), (G3, E * 4.0),
];

const FUR_ELISE_HARMONY: &[(f32, f32)] = &[
    // Intro
    (0.0, 1.76),
    // Am Arpeggio
    (A2, S), (E3, S), (A3, 3.0*S),// (0.0, 0.44),
    // E Major Arpeggio
    (E2, S), (E3, S), (G_SHARP_3, 3.0*S),// (0.0, 0.44),
    // Am Arpeggio (Turnaround)
    (A2, S), (E3, S), (A3, 3.0*S),

    // Repeat Intro
    (0.0, 1.76-3.0*S),
    // Am Arpeggio
    (A2, S), (E3, S), (A3, 3.0*S),// (0.0, 0.44),
    // Ending phrase
    (E2, S), (E3, S), (G_SHARP_3, 3.0*S), (A2, E),
];

// Single voice state
struct Voice {
    // 10 harmonics oscillators
    sines: [Sine; 10],
    // State to track song position
    sample_counter: usize,
    current_note_idx: usize,
    song: Vec<(f32, f32)>,
    speed_mult: f32,
}

impl Voice {
    fn new(song: Vec<(f32, f32)>, speed_mult: f32) -> Self {
        Self {
            sines: Default::default(),
            sample_counter: 0,
            current_note_idx: usize::MAX,
            song,
            speed_mult,
        }
    }

    fn step(&mut self) -> f32 {
        let sample_rate = 48_000.0f32;
        let mut time_cursor = 0.0f32;
        let mut active_freq = 0.0;
        let mut note_elapsed = 0.0;
        let mut found_note = false;
        let mut note_idx = 0;

        // Determine which note to play
        for (i, (freq, dur_raw)) in self.song.iter().enumerate() {
            let dur = dur_raw * self.speed_mult;

            // Use round to ensure contiguous sample ranges without floating point gaps
            let start_sample = (time_cursor * sample_rate).round() as usize;
            let end_sample = ((time_cursor + dur) * sample_rate).round() as usize;

            if self.sample_counter >= start_sample && self.sample_counter < end_sample {
                active_freq = *freq;
                // Calculate elapsed time based on sample difference to avoid jitter
                note_elapsed = (self.sample_counter - start_sample) as f32 / sample_rate;
                found_note = true;
                note_idx = i;
                break;
            }
            time_cursor += dur;
        }

        // Increment sample counter for next call
        self.sample_counter += 1;

        if !found_note {
            return 0.0;
        }

        // Reset oscillators if new note (to reset phase for attack)
        if note_idx != self.current_note_idx {
            self.current_note_idx = note_idx;
            for s in &mut self.sines {
                *s = Sine::default();
            }
        }

        // If freq is 0 (missed note), return silence
        if active_freq <= 0.0 {
            return 0.0;
        }

        // Calculate sample by mixing harmonics
        let mut mixed = 0.0;

        for (i, sine) in self.sines.iter_mut().enumerate() {
            let h_freq = active_freq * (i as f32 + 1.0);
            let sample = sine.step(h_freq);
            // Convert Ch32 to f32
            let s_f32: f32 = sample.into();
            mixed += s_f32 * HARMONICS[i];
        }

        // Piano Envelope (percussive)
        let attack_time = 0.01;
        let envelope = if note_elapsed < attack_time {
            note_elapsed / attack_time
        } else {
             let decay_rate = 3.0;
             (-decay_rate * (note_elapsed - attack_time)).exp()
        };

        mixed * envelope * 0.25 // Scale down volume
    }
}

// State of the synthesizer.
struct Processors {
    voices: Vec<Voice>,
}

impl Processors {
    fn new(tracks: Vec<Vec<(f32, f32)>>, speed_mult: f32) -> Self {
        Self {
            voices: tracks.into_iter().map(|s| Voice::new(s, speed_mult)).collect(),
        }
    }

    // Synthesis logic
    fn step(&mut self, frame: Frame<Ch32, 2>) -> Frame<Ch32, 2> {
        let mut mixed = 0.0;
        for voice in &mut self.voices {
            mixed += voice.step();
        }

        // Pan center
        frame.pan(Ch32::new(mixed), 0.0)
    }
}

fn freq_to_name(freq: f32) -> String {
    if freq < 1.0 {
        return "Rest".to_string();
    }
    let epsilon = 0.1;
    if (freq - E2).abs() < epsilon { return "E2".to_string(); }
    if (freq - A2).abs() < epsilon { return "A2".to_string(); }
    if (freq - C3).abs() < epsilon { return "C3".to_string(); }
    if (freq - E3).abs() < epsilon { return "E3".to_string(); }
    if (freq - G3).abs() < epsilon { return "G3".to_string(); }
    if (freq - G_SHARP_3).abs() < epsilon { return "G#3".to_string(); }
    if (freq - A3).abs() < epsilon { return "A3".to_string(); }
    if (freq - C4).abs() < epsilon { return "C4".to_string(); }
    if (freq - D4).abs() < epsilon { return "D4".to_string(); }
    if (freq - E4).abs() < epsilon { return "E4".to_string(); }
    if (freq - F4).abs() < epsilon { return "F4".to_string(); }
    if (freq - G4).abs() < epsilon { return "G4".to_string(); }
    if (freq - G_SHARP_4).abs() < epsilon { return "G#4".to_string(); }
    if (freq - A4).abs() < epsilon { return "A4".to_string(); }
    if (freq - B4).abs() < epsilon { return "B4".to_string(); }
    if (freq - C5).abs() < epsilon { return "C5".to_string(); }
    if (freq - D5).abs() < epsilon { return "D5".to_string(); }
    if (freq - D_SHARP_5).abs() < epsilon { return "D#5".to_string(); }
    if (freq - E5).abs() < epsilon { return "E5".to_string(); }

    format!("{:.2} Hz", freq)
}

struct NoteInfo {
    name: String,
    freq: f32,
    duration: f32,
}

struct VariationInfo {
    filename: String,
    ideal_filename: String,
    tempo_accuracy: f32,
    pitch_accuracy: f32,
    notes: Vec<Vec<NoteInfo>>,
}

fn generate(filename: &str, tracks: Vec<&[(f32, f32)]>, speed_mult: f32) {
    // Calculate total duration (max of all tracks)
    let total_duration: f32 = tracks.iter()
        .map(|track| track.iter().map(|(_, d)| d * speed_mult).sum::<f32>())
        .fold(0.0, f32::max);

    let sample_rate = 48_000;

    // Initialize audio (buffer size based on song length + 1 second tail)
    let buffer_len = (sample_rate as f32 * (total_duration + 1.0)) as usize;
    let mut audio = Audio::<Ch16, 2>::with_silence(sample_rate, buffer_len);

    // Create audio processors
    let proc = Processors::new(tracks.iter().map(|&s| s.to_vec()).collect(), speed_mult);

    // Build synthesis algorithm
    let mut synth = Synth::new(proc, |proc, frame: Frame<_, 2>| proc.step(frame));

    // Synthesize
    synth.stream(audio.sink());

    // Write to file
    println!("Writing {}", filename);
    wav::write(audio, format!("target_music/{}", filename).as_str()).expect("Failed to write WAV file");
}

fn generate_variations(base_name: &str, tracks: Vec<&[(f32, f32)]>) -> Vec<VariationInfo> {
    let mut variations = Vec::new();

    let get_notes = |tracks: &[&[(f32, f32)]], speed_mult: f32| -> Vec<Vec<NoteInfo>> {
        tracks
            .iter()
            .map(|track| {
                track
                    .iter()
                    .map(|(freq, dur)| NoteInfo {
                        name: freq_to_name(*freq),
                        freq: *freq,
                        duration: *dur * speed_mult,
                    })
                    .collect()
            })
            .collect()
    };

    // 1. Original
    let original_filename = format!("{}.wav", base_name);
    generate(&original_filename, tracks.clone(), 1.0);
    variations.push(VariationInfo {
        filename: original_filename.clone(),
        ideal_filename: original_filename.clone(),
        tempo_accuracy: 1.0,
        pitch_accuracy: 1.0,
        notes: get_notes(&tracks, 1.0),
    });

    // 2. Fast (1.15x speed)
    let speed_fast = 1.0 / 1.15;
    let filename = format!("{}_fast.wav", base_name);
    generate(&filename, tracks.clone(), speed_fast);
    variations.push(VariationInfo {
        filename,
        ideal_filename: original_filename.clone(),
        tempo_accuracy: 0.85,
        pitch_accuracy: 1.0,
        notes: get_notes(&tracks, speed_fast),
    });

    // 3. Slow (0.9x speed)
    let speed_slow = 1.0 / 0.90;
    let filename = format!("{}_slow.wav", base_name);
    generate(&filename, tracks.clone(), speed_slow);
    variations.push(VariationInfo {
        filename,
        ideal_filename: original_filename.clone(),
        tempo_accuracy: 0.90,
        pitch_accuracy: 1.0,
        notes: get_notes(&tracks, speed_slow),
    });

    // 4. Missed Notes (Melody only)
    if !tracks.is_empty() {
        let mut melody = tracks[0].to_vec();
        // Count total playable notes across all tracks (ignoring rests/0.0 freq)
        let total_playable_notes: usize = tracks
            .iter()
            .map(|t| t.iter().filter(|(f, _)| *f > 0.0).count())
            .sum();
        let mut missed_count = 0;

        if melody.len() > 12 {
            // Set frequency to 0.0 to simulate missed note
            if let Some(note) = melody.get_mut(4) {
                if note.0 > 0.0 {
                    note.0 = 0.0;
                    missed_count += 1;
                }
            }
            if let Some(note) = melody.get_mut(11) {
                if note.0 > 0.0 {
                    note.0 = 0.0;
                    missed_count += 1;
                }
            }
        }

        let mut missed_tracks = vec![melody.as_slice()];
        missed_tracks.extend_from_slice(&tracks[1..]);

        let filename = format!("{}_missed_notes.wav", base_name);
        generate(&filename, missed_tracks.clone(), 1.0);

        variations.push(VariationInfo {
            filename,
            ideal_filename: original_filename.clone(),
            tempo_accuracy: 1.0,
            pitch_accuracy: (total_playable_notes - missed_count) as f32
                / total_playable_notes as f32,
            notes: get_notes(&missed_tracks, 1.0),
        });
    }

    variations
}

fn main() {
    let mut all_variations = Vec::new();

    // Fur Elise
    all_variations.extend(generate_variations("fur_elise", vec![FUR_ELISE]));

    // Ode to Joy
    all_variations.extend(generate_variations("ode_to_joy", vec![ODE_TO_JOY]));

    // Fur Elise (Polyphonic)
    all_variations.extend(generate_variations("fur_elise_harmony", vec![FUR_ELISE, FUR_ELISE_HARMONY]));

    // Ode to Joy (Polyphonic)
    all_variations.extend(generate_variations("ode_to_joy_harmony", vec![ODE_TO_JOY, ODE_TO_JOY_HARMONY]));

    // Generate JSON
    let json_items: Vec<String> = all_variations
        .iter()
        .map(|v| {
            let tracks_json = v
                .notes
                .iter()
                .map(|track| {
                    let notes_json = track
                        .iter()
                        .map(|note| {
                            format!(
                                "{{\"name\": \"{}\", \"frequency\": {:.2}, \"duration\": {:.3}}}",
                                note.name, note.freq, note.duration
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{}]", notes_json)
                })
                .collect::<Vec<_>>()
                .join(", ");

            format!(
                "  {{\n    \"filename\": \"{}\",\n    \"ideal_filename\": \"{}\",\n    \"expected_tempo_accuracy\": {:.2},\n    \"expected_pitch_accuracy\": {:.2},\n    \"notes\": [{}]\n  }}",
                v.filename, v.ideal_filename, v.tempo_accuracy, v.pitch_accuracy, tracks_json
            )
        })
        .collect();

    let json_output = format!("[\n{}\n]", json_items.join(",\n"));
    let output_file = "available_tests.json";
    std::fs::write(output_file, json_output).expect("Failed to write JSON output");
    println!("Wrote {}", output_file);
}
