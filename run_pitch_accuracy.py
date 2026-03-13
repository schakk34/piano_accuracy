#!/usr/bin/env python3
"""
Run harmony HMM pitch-accuracy on an MP3.

Fixes added in this version (to reduce false "wrong" due to harmonics/brightness):
1) EXTRA peaks now require *persistence* across the segment (not just a single spike).
2) Harmonic explainability expanded + tolerance relaxed.
3) Extras far above the top expected note are ignored (usually harmonics/brightness).
4) (Important note) Your "detune" check in MIDI space can't detect <1 semitone detune
   because MIDI indices are integers. This file keeps your current approach but
   improves wrong-note suppression a lot.

Usage:
  python run_pitch_accuracy.py --mp3 <path_to.mp3> --model ode_to_joy [--plot]
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_MP3 = "tests/piano-synth/target_music/ode_to_joy_harmony_missed_notes.mp3"
MODEL_NAME = "ode_to_joy"

SR = 22050
HOP_LENGTH = 512

# C2 in Hz (avoid librosa.note_to_hz at import for compatibility)
FMIN = 440.0 * (2.0 ** ((36 - 69) / 12.0))  # ~65.4 Hz

# Pitch resolution
BINS_PER_OCTAVE = 24

# If you previously used N_BINS=84 at 12 bins/octave (~7 octaves),
# then to keep the same frequency span at 24 bins/octave, use 168.
N_BINS = 168


# ---------------------------------------------------------------------------
# Song models: melody + harmony (pitches, beats), BPM, merge_adjacent_same
# ---------------------------------------------------------------------------
def _ode_to_joy_model():
    E_b, Q_b = 1.0, 2.0
    E4, F4, G4, D4, C4, C3, G3 = 64, 65, 67, 62, 60, 48, 55
    mel = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
    mel_beats = [E_b] * 8 + [E_b] * 4 + [1.5 * E_b, 0.5 * E_b, Q_b]
    harm = [C3, 0, G3, 0, C3, 0, G3, 0]
    harm_beats = [2.0 * E_b, 2.0 * E_b, 2.0 * E_b, 2.0 * E_b, 2.0 * E_b, 2.0 * E_b, 2.0 * E_b, 2.0 * E_b]
    bpm_ref = 60.0 / 0.44  # ~136.36
    return mel, mel_beats, harm, harm_beats, bpm_ref, False

def _ode_to_joy_no_harmony_model():

    E_b, Q_b = 1.0, 2.0

    E4, F4, G4, D4, C4 = 64, 65, 67, 62, 60

    mel = [
        E4, E4, F4, G4, G4, F4, E4, D4,
        C4, C4, D4, E4, E4, D4, D4
    ]


    mel_beats = [E_b] * 12 + [1.5 * E_b, 0.5 * E_b, Q_b]

    # No harmony track: make harmony a single "rest" spanning full melody duration
    total_beats = sum(mel_beats)
    harm = [0]                 # rest / silence
    harm_beats = [total_beats] # same total length so combine_tracks_to_segments works cleanly

    bpm_ref = 60.0 / 0.44  # keep consistent with your other models (~136.36)
    return mel, mel_beats, harm, harm_beats, bpm_ref, False

def _fur_elise_model():
    S_b, E_b = 0.5, 1.0  
    # MIDI
    E5, Ds5, D5, C5, B4, A4 = 76, 75, 74, 72, 71, 69
    Gs4, E4, C4 = 68, 64, 60

    mel = [
        E5, Ds5, E5, Ds5, E5, B4, D5, C5, A4,
        C4, E4, A4, B4,
        E4, Gs4, B4, C5,
        E4,
        E5, Ds5, E5, Ds5, E5, B4, D5, C5, A4,
        C4, E4, A4, B4,
        E4, C5, B4, A4,
    ]

    mel_beats = [
        S_b, S_b, S_b, S_b, S_b, S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b,
        S_b, S_b, S_b, S_b, S_b, S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
        S_b, S_b, S_b, E_b,
    ]

    A2, E3, A3 = 45, 52, 57
    E2, Gs3 = 40, 56

    harm = [
        0,
        A2, E3, A3,
        E2, E3, Gs3,
        A2, E3, A3,
        0,
        A2, E3, A3,
        E2, E3, Gs3, A2
    ]

    harm_beats = [
        1.76 / 0.44,        # 4.0 beats
        S_b, S_b, 3*S_b,
        S_b, S_b, 3*S_b,
        S_b, S_b, 3*S_b,
        (1.76 - 3*0.22) / 0.44,  # 2.5 beats
        S_b, S_b, 3*S_b,
        S_b, S_b, 3*S_b, E_b
    ]

    bpm_ref = 60.0 / 0.44
    return mel, mel_beats, harm, harm_beats, bpm_ref, False


SONG_MODELS = {
    "ode_to_joy": _ode_to_joy_model,
    "ode_to_joy_no_harmony": _ode_to_joy_no_harmony_model,
    "fur_elise": _fur_elise_model,
}


def get_model(model_name):
    if model_name not in SONG_MODELS:
        print(f"Unknown model: {model_name}. Available: {list(SONG_MODELS.keys())}", file=sys.stderr)
        sys.exit(1)
    return SONG_MODELS[model_name]()


def combine_tracks_to_segments(mel_pitches, mel_beats, harm_pitches, harm_beats, merge_adjacent_same=True):
    i = j = 0
    rem_m, rem_h = mel_beats[0], harm_beats[0]
    cur_m, cur_h = mel_pitches[0], harm_pitches[0]
    segments = []
    while i < len(mel_pitches) and j < len(harm_pitches):
        dt = min(rem_m, rem_h)
        pitches = tuple(sorted([p for p in (cur_m, cur_h) if p and p > 0]))
        segments.append((pitches, dt))
        rem_m -= dt
        rem_h -= dt
        if rem_m <= 1e-9:
            i += 1
            if i >= len(mel_pitches):
                break
            cur_m, rem_m = mel_pitches[i], mel_beats[i]
        if rem_h <= 1e-9:
            j += 1
            if j >= len(harm_pitches):
                break
            cur_h, rem_h = harm_pitches[j], harm_beats[j]

    if not merge_adjacent_same:
        return segments

    merged = []
    for ps, b in segments:
        if merged and merged[-1][0] == ps:
            merged[-1] = (ps, merged[-1][1] + b)
        else:
            merged.append((ps, b))
    return merged


# ---------------------------------------------------------------------------
# Audio & CQT
# ---------------------------------------------------------------------------
def mp3_to_cqt_db(audio_path, sr, hop_length, fmin, n_bins, bins_per_octave):
    y, sr_used = librosa.load(audio_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    C = librosa.cqt(
        y=y,
        sr=sr_used,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    C_mag = np.abs(C)
    X_db = librosa.amplitude_to_db(C_mag, ref=np.max)
    return X_db, y, sr_used


def dt_ms_from_cqt(sr, hop_length):
    return 1000.0 * hop_length / sr


def onset_strength_envelope(y, sr, hop_length):
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    o = (o.astype(float) - o.min()) / (o.max() + 1e-9)
    return o


# ---------------------------------------------------------------------------
# HMM: templates, states, transitions, emissions, Viterbi
# ---------------------------------------------------------------------------
def silence_template(n_bins):
    tpl = np.ones(n_bins, float)
    return tpl / (np.linalg.norm(tpl) + 1e-9)


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def midi_to_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (m // 12) - 1
    return f"{names[m % 12]}{octave}"


def hz_to_cqt_bin(hz, fmin, bins_per_octave):
    return int(np.round(bins_per_octave * np.log2(hz / fmin)))


def note_template_cqt(midi, fmin, n_bins, bins_per_octave, window_bins=2):
    tpl = np.zeros(n_bins, float)
    f0 = midi_to_hz(midi)
    harm_w = {1: 1.0, 2: 0.5, 3: 0.25}
    for h, w in harm_w.items():
        k = hz_to_cqt_bin(h * f0, fmin, bins_per_octave)
        for kk in range(k - window_bins, k + window_bins + 1):
            if 0 <= kk < n_bins:
                tpl[kk] += w
    return tpl / (np.linalg.norm(tpl) + 1e-9)


def chord_template_cqt(pitches, fmin, n_bins, bins_per_octave, window_bins=2):
    tpl = np.zeros(n_bins, dtype=float)
    for midi in pitches:
        tpl += note_template_cqt(midi, fmin, n_bins, bins_per_octave, window_bins=window_bins)
    return tpl / (np.linalg.norm(tpl) + 1e-9)


PHASES = ["A", "S", "R"]


def build_ASR_states_segments(
    segments,
    bpm_ref,
    dt_ms,
    frac_attack=0.15,
    frac_release=0.15,
    min_A=1,
    min_S=1,
    min_R=1,
    p_start_to_first=0.02,
    tempo_slack=1.02,
):
    ms_per_beat = 60000.0 / bpm_ref
    frames_per_beat = ms_per_beat / dt_ms
    seg_steps = [max(1, int(round(dur_beats * frames_per_beat))) for _, dur_beats in segments]

    states = [{"seg_idx": -1, "pitches": (), "phase": "START", "dur_steps": None}]
    for i, ((pitchset, _), Tn) in enumerate(zip(segments, seg_steps)):
        Ta = max(min_A, int(round(frac_attack * Tn)))
        Tr = max(min_R, int(round(frac_release * Tn)))
        Ts = max(min_S, Tn - Ta - Tr)
        for ph, Td in zip(PHASES, [Ta, Ts, Tr]):
            states.append({"seg_idx": i, "pitches": tuple(pitchset), "phase": ph, "dur_steps": Td})

    S = len(states)
    A = np.zeros((S, S), float)

    def idx(seg_i, phase):
        return 1 + 3 * seg_i + PHASES.index(phase)

    start, first_attack = 0, idx(0, "A")
    A[start, start] = 1.0 - p_start_to_first
    A[start, first_attack] = p_start_to_first

    N = len(segments)
    for seg_i in range(N):
        a, s, r = idx(seg_i, "A"), idx(seg_i, "S"), idx(seg_i, "R")

        def set_geometric(from_state, to_state, expected_frames, slack=1.0):
            expected = max(1, int(round(expected_frames * slack)))
            p_adv = 1.0 / expected
            A[from_state, from_state] = 1.0 - p_adv
            A[from_state, to_state] = p_adv

        set_geometric(a, s, states[a]["dur_steps"], slack=tempo_slack)
        set_geometric(s, r, states[s]["dur_steps"], slack=tempo_slack)

        if seg_i < N - 1:
            set_geometric(r, idx(seg_i + 1, "A"), states[r]["dur_steps"], slack=tempo_slack)
        else:
            A[r, r] = 1.0

    pi = np.zeros(S, float)
    pi[start] = 1.0
    return states, A, pi


# --- FIX: expanded harmonic explainability + tolerance usage ----------------
def _harmonic_explainable(m_peak: int, expected_set, max_offset_semitones=1.0):
    """
    True if m_peak is plausibly a harmonic/partial of any expected note.

    NOTE: because your peaks are integer-MIDI, we use a slightly larger tolerance.
    """
    harmonic_offsets = [0, 12, 19, 24, 28, 31, 36]  # expanded
    for p in expected_set:
        for off in harmonic_offsets:
            if abs(m_peak - (p + off)) <= max_offset_semitones:
                return True
    return False


def _filter_unexplained_extras(extra_peaks, expected_set, max_offset_semitones=1.0):
    return {m for m in extra_peaks if not _harmonic_explainable(m, expected_set, max_offset_semitones)}


def _ignore_above_top_note(extra_peaks, expected_set, margin_semitones=6):
    """
    Ignore peaks that are far above the top expected note.
    These are usually harmonics/brightness, not "wrong notes".
    """
    if not expected_set:
        return extra_peaks
    top = max(expected_set)
    return {m for m in extra_peaks if m <= top + margin_semitones}


def build_ASR_templates_segments(states, fmin, n_bins, bins_per_octave, window_bins=2):
    sil = silence_template(n_bins)
    TPL = []
    for st in states:
        if st["phase"] in ("A", "S"):
            TPL.append(chord_template_cqt(st["pitches"], fmin, n_bins, bins_per_octave, window_bins=window_bins))
        else:
            TPL.append(sil)
    return np.stack(TPL, axis=0)


def boundary_mask_segments(states):
    S = len(states)
    mask = np.zeros((S, S), dtype=bool)
    seg_idx = np.array([st.get("seg_idx", -999) for st in states], dtype=int)
    phase = np.array([st["phase"] for st in states])

    for i in range(S):
        if phase[i] == "START":
            continue
        for j in range(S):
            if phase[j] != "A":
                continue
            if seg_idx[j] == seg_idx[i] + 1:
                mask[i, j] = True
    return mask


def emissions_cosine(C, TPL, scale=1.0):
    C = C - C.mean(axis=0, keepdims=True)
    C /= (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    scores = TPL @ C
    return scale * scores, scores


def emissions_ASR(
    C_db,
    TPL,
    onset01,
    states,
    scale_pitch=2.0,
    w_attack_onset=2.0,
    w_sustain_onset=-0.5,
    w_release_onset=-1.0,
    w_release_energy=-0.5,
):
    B_base, scores = emissions_cosine(C_db, TPL, scale=scale_pitch)
    S, T = B_base.shape

    o = onset01[:T] if len(onset01) >= T else np.pad(onset01, (0, T - len(onset01)), mode="edge")
    energy = C_db.mean(axis=0)
    energy = (energy - energy.mean()) / (energy.std() + 1e-9)

    B = B_base.copy()
    for s in range(S):
        ph = states[s]["phase"]
        if ph == "A":
            B[s, :] += w_attack_onset * o
        elif ph == "S":
            B[s, :] += w_sustain_onset * o
        else:
            B[s, :] += w_release_onset * o + w_release_energy * energy
    return B, scores


def viterbi_ASR(A, pi, B, on01=None, boundary_boost_mask=None, bonus=3.0):
    S, T = B.shape
    logA = np.where(A > 0, np.log(A), -np.inf)
    logpi = np.where(pi > 0, np.log(pi), -np.inf)

    dp = np.full((S, T), -np.inf)
    bp = np.zeros((S, T), dtype=int)
    dp[:, 0] = logpi + B[:, 0]

    use_bonus = (on01 is not None) and (boundary_boost_mask is not None)

    for t in range(1, T):
        logA_t = logA
        if use_bonus:
            logA_t = logA.copy()
            logA_t[boundary_boost_mask] += bonus * on01[t]

        for s in range(S):
            prev = dp[:, t - 1] + logA_t[:, s]
            pbest = int(np.argmax(prev))
            dp[s, t] = prev[pbest] + B[s, t]
            bp[s, t] = pbest

    last = int(np.argmax(dp[:, -1]))
    path = np.zeros(T, dtype=int)
    path[-1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[path[t], t]
    return path, dp


# ---------------------------------------------------------------------------
# Path -> segment frame ranges
# ---------------------------------------------------------------------------
def path_to_state_segments(path, states, sr, hop_length, T=None):
    path = np.asarray(path, dtype=int)
    if T is None:
        T = len(path)
    path = path[:T]

    segs = []
    start = 0
    for t in range(1, T):
        if path[t] != path[t - 1]:
            segs.append((path[t - 1], start, t - 1))
            start = t
    segs.append((path[T - 1], start, T - 1))

    out = []
    for s, f0, f1 in segs:
        st = states[s]
        t0 = librosa.frames_to_time(f0, sr=sr, hop_length=hop_length)
        t1 = librosa.frames_to_time(f1 + 1, sr=sr, hop_length=hop_length)
        out.append({
            "state": int(s),
            "phase": st.get("phase"),
            "seg_idx": st.get("seg_idx"),
            "pitches": st.get("pitches"),
            "start_frame": int(f0),
            "end_frame": int(f1),
            "start_time": float(t0),
            "end_time": float(t1),
        })
    return out


def get_segment_frame_ranges(path, states, sr, hop_length, phases=("A", "S")):
    segs = path_to_state_segments(path, states, sr=sr, hop_length=hop_length, T=len(path))
    by_seg = {}
    for s in segs:
        if s.get("phase") not in phases:
            continue
        idx = s.get("seg_idx")
        if idx is None or idx < 0:
            continue
        f0, f1 = s["start_frame"], s["end_frame"]
        if idx not in by_seg:
            by_seg[idx] = [f0, f1]
        else:
            by_seg[idx][0] = min(by_seg[idx][0], f0)
            by_seg[idx][1] = max(by_seg[idx][1], f1)
    return {idx: (r[0], r[1]) for idx, r in by_seg.items()}


# ---------------------------------------------------------------------------
# Segment note presence (fixed)
# ---------------------------------------------------------------------------
@dataclass
class NoteDetectParams:
    # global gating
    quiet_percentile: float = 10.0
    quiet_margin_db: float = 10.0

    # peak selection
    presence_margin_db: float = 14.0   # how close to max energy a peak must be (for candidate peaks)
    peak_neighbor: int = 1            # local max check: m-1 and m+1

    # matching (NOTE: since MIDI is integer, detune<1 semitone is not representable here)
    max_detune_semitones: float = 0.45
    match_window_semitones: int = 2

    # octave guard
    octave_penalty_db: float = 6.0

    # --- FIX: extras must be persistent, not just a spike
    extra_presence_margin_db: float = 10.0   # extra peak must be within this of segment max (stricter)
    extra_min_frames_ratio: float = 0.30     # and present in >= this fraction of frames


def _bin_to_approx_midi(bin_idx, fmin, bins_per_octave):
    hz = fmin * (2.0 ** (bin_idx / bins_per_octave))
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _midi_energy_over_time(slice_db, fmin, bins_per_octave):
    """
    Build MIDI -> per-frame energy array by max pooling bins that map to the same MIDI.
    slice_db: (n_bins, n_frames)
    """
    n_bins, _n_frames = slice_db.shape
    midi_to_binrows = {}
    for b in range(n_bins):
        m = _bin_to_approx_midi(b, fmin, bins_per_octave)
        midi_to_binrows.setdefault(m, []).append(b)

    midi_energy_t = {}
    for m, bins in midi_to_binrows.items():
        midi_energy_t[m] = np.max(slice_db[bins, :], axis=0)  # (n_frames,)
    return midi_energy_t


def _compute_midi_energy_from_time(midi_energy_t):
    """
    Convert MIDI->(per-frame) into MIDI->scalar energy using a robust percentile.
    """
    return {m: float(np.percentile(v, 90)) for m, v in midi_energy_t.items()}


def _strong_peaks(midi_energy, max_energy, presence_margin_db, neighbor=1):
    mids = sorted(midi_energy.keys())
    strong = []
    for m in mids:
        e = midi_energy[m]
        if e < max_energy - presence_margin_db:
            continue
        left = midi_energy.get(m - neighbor, -np.inf)
        right = midi_energy.get(m + neighbor, -np.inf)
        if e >= left and e >= right:
            strong.append(m)
    return strong


def _match_expected_to_peaks(exp_set, peaks, midi_energy, params: NoteDetectParams):
    if not peaks:
        return set(), set(), []

    used_peaks = set()
    matched_expected = set()
    peak_list = list(peaks)

    for p in sorted(exp_set):
        candidates = [m for m in peak_list if abs(m - p) <= params.match_window_semitones]
        if not candidates:
            continue

        def score(m):
            e = midi_energy.get(m, -np.inf)
            dist = abs(m - p)
            octaveish = (abs(m - p) >= 10)
            oct_pen = params.octave_penalty_db if octaveish else 0.0
            return e - oct_pen - 2.0 * dist

        best_m = max(candidates, key=score)

        # NOTE: integer MIDI -> this is effectively "exact match" when max_detune<1
        if abs(best_m - p) <= params.max_detune_semitones:
            matched_expected.add(p)
            used_peaks.add(best_m)

    extra_peaks = set(peaks) - used_peaks
    return matched_expected, extra_peaks, sorted(list(used_peaks))


def _peak_persistence_ratio(m, midi_energy_t, max_energy, margin_db):
    v = midi_energy_t.get(m)
    if v is None or len(v) == 0:
        return 0.0
    return float(np.mean(v >= (max_energy - margin_db)))


def simple_segment_note_presence(
    C_db,
    path,
    states,
    expected_segments,
    sr,
    hop_length,
    fmin,
    bins_per_octave,
    params: NoteDetectParams,
):
    _n_bins, T = C_db.shape
    frame_ranges = get_segment_frame_ranges(path, states, sr, hop_length)

    # compute segment max energies for quiet threshold
    segment_max_energies = []
    for seg_idx in range(len(expected_segments)):
        if seg_idx not in frame_ranges:
            segment_max_energies.append(-np.inf)
            continue
        f0, f1 = frame_ranges[seg_idx]
        f1 = min(f1, T - 1)
        if f1 <= f0:
            segment_max_energies.append(-np.inf)
            continue
        slice_db = C_db[:, f0 : f1 + 1]
        midi_energy_t = _midi_energy_over_time(slice_db, fmin, bins_per_octave)
        midi_energy = _compute_midi_energy_from_time(midi_energy_t)
        segment_max_energies.append(max(midi_energy.values()) if midi_energy else -np.inf)

    valid = np.array([e for e in segment_max_energies if np.isfinite(e)])
    quiet_thresh = (np.percentile(valid, params.quiet_percentile) - params.quiet_margin_db) if len(valid) else -np.inf

    results = []
    for seg_idx, (exp_pitches, _exp_dur_beats) in enumerate(expected_segments):
        rec = {
            "expected_idx": seg_idx,
            "expected_pitches": tuple(exp_pitches),
            "present_midi": [],
            "missing_midi": [],
            "extra_midi": [],
            "status": "rest",
            "debug_peaks": [],
        }

        if seg_idx not in frame_ranges:
            rec["status"] = "skipped"
            rec["start_time"] = rec["end_time"] = None
            results.append(rec)
            continue

        f0, f1 = frame_ranges[seg_idx]
        f1 = min(f1, T - 1)
        rec["start_time"] = float(librosa.frames_to_time(f0, sr=sr, hop_length=hop_length))
        rec["end_time"] = float(librosa.frames_to_time(f1 + 1, sr=sr, hop_length=hop_length))

        if f1 <= f0:
            rec["status"] = "rest"
            results.append(rec)
            continue

        slice_db = C_db[:, f0 : f1 + 1]
        midi_energy_t = _midi_energy_over_time(slice_db, fmin, bins_per_octave)
        midi_energy = _compute_midi_energy_from_time(midi_energy_t)
        if not midi_energy:
            rec["status"] = "rest"
            results.append(rec)
            continue

        exp_set = {p for p in exp_pitches if p > 0}
        max_energy = max(midi_energy.values())

        # quiet gating
        if max_energy < quiet_thresh:
            rec["present_midi"] = []
            rec["missing_midi"] = sorted(exp_set)
            rec["extra_midi"] = []
            rec["status"] = "missing" if exp_set else "rest"
            results.append(rec)
            continue

        # candidate peaks
        peaks = _strong_peaks(
            midi_energy,
            max_energy=max_energy,
            presence_margin_db=params.presence_margin_db,
            neighbor=params.peak_neighbor,
        )
        rec["debug_peaks"] = peaks

        if not exp_set:
            rec["present_midi"] = []
            rec["missing_midi"] = []
            rec["extra_midi"] = sorted(peaks[:2])
            rec["status"] = "rest"
            results.append(rec)
            continue

        matched_expected, extra_peaks, _used = _match_expected_to_peaks(exp_set, peaks, midi_energy, params)

        # --- FIX A: filter harmonic explainables with relaxed tolerance
        unexplained = _filter_unexplained_extras(extra_peaks, exp_set, max_offset_semitones=1.0)

        # --- FIX B: ignore very-high peaks (brightness/harmonics)
        unexplained = _ignore_above_top_note(unexplained, exp_set, margin_semitones=6)

        # --- FIX C: require extras to persist across frames (not transient spikes)
        unexplained = {
            m for m in unexplained
            if _peak_persistence_ratio(m, midi_energy_t, max_energy, params.extra_presence_margin_db)
            >= params.extra_min_frames_ratio
        }

        extra = sorted(unexplained)
        missing = sorted(exp_set - matched_expected)

        rec["present_midi"] = sorted(matched_expected)
        rec["missing_midi"] = missing
        rec["extra_midi"] = extra

        # Melody/harmony split (for 2-note chords: low = harmony, high = melody)
        exp_sorted = sorted(exp_set)
        rec["mel_note"] = exp_sorted[-1] if exp_sorted else None
        rec["harm_note"] = exp_sorted[0] if len(exp_sorted) >= 2 else None
        rec["mel_ok"] = rec["mel_note"] in matched_expected if rec["mel_note"] is not None else None
        rec["harm_ok"] = rec["harm_note"] in matched_expected if rec["harm_note"] is not None else None

        # Classification:
        # - fully correct
        # - partially_missing: some but not all expected notes missing
        # - missing: all expected notes missing
        # - extras by themselves are treated as harmless embellishments
        if missing and extra:
            if 0 < len(missing) < len(exp_set):
                rec["status"] = "missing_and_wrong"  # partial + wrong
            else:
                rec["status"] = "missing_and_wrong"
        elif missing:
            if 0 < len(missing) < len(exp_set):
                rec["status"] = "partially_missing"
            else:
                rec["status"] = "missing"
        else:
            rec["status"] = "ok"

        results.append(rec)

    return results


def print_simple_segment_notes(results, max_rows=40):
    for r in results[:max_rows]:
        idx = r["expected_idx"]
        status = r["status"]
        exp = r["expected_pitches"]
        exp_names = [midi_to_name(p) for p in exp if p > 0]
        missing = r["missing_midi"]
        missing_names = [midi_to_name(p) for p in missing]
        extra = r["extra_midi"]
        extra_names = [midi_to_name(p) for p in extra]
        present = r["present_midi"]
        present_names = [midi_to_name(p) for p in present]
        print(
            f"Seg {idx:2d}  expected={exp} ({','.join(exp_names)})  "
            f"status={status:16s}  present={present} ({','.join(present_names)})",
            end="",
        )
        if missing:
            print(f"  missing={missing} ({','.join(missing_names)})", end="")
        if extra:
            print(f"  extra={extra} ({','.join(extra_names)})", end="")
        print()


def evaluate_pitch_accuracy_from_simple(simple_results):
    total = len(simple_results)
    correct_count = sum(1 for r in simple_results if r["status"] == "ok")
    partial_count = sum(1 for r in simple_results if r["status"] == "partially_missing")
    # scoring: ok = 1, partially_missing = 0.5, everything else = 0
    score = correct_count * 1.0 + partial_count * 0.5
    accuracy = score / total if total > 0 else 0.0
    skipped_count = sum(1 for r in simple_results if r["status"] == "skipped")
    return {
        "results": simple_results,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "partial_count": partial_count,
        "wrong_count": total - correct_count - partial_count,
        "skipped_count": skipped_count,
        "total_count": total,
        "correct_segments": [r["expected_idx"] for r in simple_results if r["status"] == "ok"],
        "wrong_segments": [r["expected_idx"] for r in simple_results if r["status"] != "ok"],
        "skipped_segments": [r["expected_idx"] for r in simple_results if r["status"] == "skipped"],
    }


def print_accuracy_report(eval_result, show_details=True, max_rows=50):
    print("=" * 70)
    print("PITCH ACCURACY REPORT")
    print("=" * 70)
    print(f"Overall Accuracy (with partial credit): {eval_result['accuracy']:.1%}")
    print(f"Correct (full): {eval_result['correct_count']}/{eval_result['total_count']}")
    print(f"Partially correct: {eval_result.get('partial_count', 0)}/{eval_result['total_count']}")
    print(f"Skipped: {eval_result['skipped_count']}/{eval_result['total_count']}")
    print(f"Wrong:   {eval_result['wrong_count']}/{eval_result['total_count']}")
    print()
    if show_details:
        print("Detailed Results:")
        print("-" * 70)
        print(f"{'Idx':<5} {'Status':<16} {'Expected':<25} {'Present(exp)':<20} {'Extra(peaks)':<20}")
        print("-" * 70)
        for r in eval_result["results"][:max_rows]:
            exp_str = str(r.get("expected_pitches", ()))
            present_str = str(r.get("present_midi", []))
            extra_str = str(r.get("extra_midi", []))
            icon = {"ok": "✓", "missing": "✗", "wrong": "⚠", "missing_and_wrong": "⚠", "skipped": "—", "rest": "—"}.get(r["status"], "?")
            print(f"{r['expected_idx']:<5} {icon} {r['status']:<15} {exp_str:<25} {present_str:<20} {extra_str:<20}")

        if len(eval_result["results"]) > max_rows:
            print(f"... ({len(eval_result['results']) - max_rows} more segments)")
    print("=" * 70)


def plot_accuracy_timeline_from_simple(simple_results, eval_result=None, title="Pitch accuracy (peak-based)"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if eval_result is None:
        eval_result = evaluate_pitch_accuracy_from_simple(simple_results)

    fig, (ax_mel, ax_harm) = plt.subplots(2, 1, sharex=True, figsize=(14, 4))

    colors = {
        "ok": "green",
        "missing": "darkred",
        "partially_missing": "coral",
        "missing_and_wrong": "purple",
        "skipped": "gray",
        "rest": "lightgray",
    }

    for r in simple_results:
        start, end = r.get("start_time"), r.get("end_time")
        if start is None or end is None:
            continue
        idx = r.get("expected_idx", 0)
        exp = sorted(p for p in r.get("expected_pitches", ()) if p > 0)
        if not exp:
            continue

        mel_note = r.get("mel_note") or (exp[-1] if exp else None)
        harm_note = r.get("harm_note") or (exp[0] if len(exp) >= 2 else None)
        present = set(r.get("present_midi", []))

        # Melody row
        if mel_note is not None:
            mel_status = "ok" if mel_note in present else "missing"
            c_mel = colors.get(mel_status, "gray")
            ax_mel.barh(0.5, end - start, left=start, height=0.4, color=c_mel, alpha=0.8, edgecolor="black", linewidth=0.4)
            if mel_status != "ok":
                ax_mel.text((start + end) / 2, 0.5, f"{idx}", ha="center", va="center", fontsize=7)

        # Harmony row
        if harm_note is not None and harm_note != mel_note:
            harm_status = "ok" if harm_note in present else "missing"
            c_h = colors.get(harm_status, "gray")
            ax_harm.barh(0.5, end - start, left=start, height=0.4, color=c_h, alpha=0.8, edgecolor="black", linewidth=0.4)
            if harm_status != "ok":
                ax_harm.text((start + end) / 2, 0.5, f"{idx}", ha="center", va="center", fontsize=7)

    ax_harm.set_xlabel("Time (seconds)")
    ax_mel.set_yticks([0.5])
    ax_harm.set_yticks([0.5])
    ax_mel.set_yticklabels(["Melody"])
    ax_harm.set_yticklabels(["Harmony"])
    ax_mel.set_ylim(0, 1)
    ax_harm.set_ylim(0, 1)

    # Overall scoring in the title
    acc = eval_result.get("accuracy", 0.0)
    corr = eval_result.get("correct_count", 0)
    part = eval_result.get("partial_count", 0)
    total = eval_result.get("total_count", 0)
    fig.suptitle(f"{title}\nAccuracy={acc:.1%} (full={corr}, partial={part}, total={total})")
    ax_mel.grid(True, alpha=0.3, axis="x")
    ax_harm.grid(True, alpha=0.3, axis="x")

    legend_elements = [
        Patch(facecolor="green", alpha=0.8, label="Correct"),
        Patch(facecolor="darkred", alpha=0.8, label="Missing"),
        Patch(facecolor="coral", alpha=0.8, label="Partially missing (segment)"),
    ]
    ax_mel.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(mp3_path: str, model_name: str, plot: bool = False):
    mp3_path = Path(mp3_path)
    if not mp3_path.exists():
        print(f"File not found: {mp3_path}", file=sys.stderr)
        sys.exit(1)

    mel_pitches, mel_beats, harm_pitches, harm_beats, bpm_ref, merge = get_model(model_name)
    expected_segments = combine_tracks_to_segments(
        mel_pitches, mel_beats, harm_pitches, harm_beats, merge_adjacent_same=merge
    )
    segments = expected_segments

    # Debug print: expected segment pitches and durations (in beats and seconds)
    print("Expected segments (pitches, beats, approx_seconds):")
    ms_per_beat = 60000.0 / bpm_ref
    for idx, (pitches, dur_beats) in enumerate(expected_segments):
        dur_sec = (dur_beats * ms_per_beat) / 1000.0
        print(f"  Seg {idx:2d}: pitches={pitches}  beats={dur_beats:.3f}  ~{dur_sec:.3f}s")

    print(f"Loading: {mp3_path}")
    C_db, y, sr = mp3_to_cqt_db(
        str(mp3_path),
        sr=SR,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    dt_ms = dt_ms_from_cqt(sr, HOP_LENGTH)
    onset_env = onset_strength_envelope(y, sr, HOP_LENGTH)

    print("Building HMM and running Viterbi...")
    states, A, pi = build_ASR_states_segments(
        segments, bpm_ref, dt_ms,
        frac_attack=0.15, frac_release=0.15,
        min_A=1, min_S=1, min_R=1,
        p_start_to_first=0.02, tempo_slack=1.02,
    )

    TPL = build_ASR_templates_segments(
        states, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, window_bins=2
    )
    B, _ = emissions_ASR(
        C_db, TPL, onset_env, states,
        scale_pitch=2.0,
        w_attack_onset=2.0,
        w_sustain_onset=-0.5,
        w_release_onset=-1.0,
        w_release_energy=-0.5,
    )
    mask = boundary_mask_segments(states)
    path, _ = viterbi_ASR(A, pi, B, on01=onset_env, boundary_boost_mask=mask, bonus=6.0)

    print("Segment note presence (peak-based + persistence)...")
    params = NoteDetectParams(
        quiet_percentile=10.0,
        quiet_margin_db=10.0,
        presence_margin_db=14.0,
        peak_neighbor=1,
        max_detune_semitones=0.45,
        match_window_semitones=2,
        octave_penalty_db=6.0,
        extra_presence_margin_db=10.0,   # extras must be *very* near max
        extra_min_frames_ratio=0.30,     # and persistent in the segment
    )

    simple_results = simple_segment_note_presence(
        C_db, path, states, expected_segments,
        sr=sr, hop_length=HOP_LENGTH, fmin=FMIN,
        bins_per_octave=BINS_PER_OCTAVE,
        params=params,
    )

    # Actual segment durations inferred from Viterbi alignment
    print("Actual segment durations from audio (seconds):")
    for r in simple_results:
        t0 = r.get("start_time")
        t1 = r.get("end_time")
        if t0 is None or t1 is None:
            continue
        dur = t1 - t0
        print(f"  Seg {r['expected_idx']:2d}: {dur:.3f}s  status={r['status']}")

    print_simple_segment_notes(simple_results, max_rows=60)
    eval_simple = evaluate_pitch_accuracy_from_simple(simple_results)
    print_accuracy_report(eval_simple, show_details=True, max_rows=60)

    if plot:
        plot_accuracy_timeline_from_simple(
            simple_results,
            eval_result=eval_simple,
            title=f"Pitch accuracy: {mp3_path.name} (model={model_name})",
        )


def main():
    parser = argparse.ArgumentParser(description="Run harmony HMM pitch accuracy on an MP3.")
    parser.add_argument("--mp3", default=INPUT_MP3, help="Path to input MP3")
    parser.add_argument("--model", default=MODEL_NAME, choices=list(SONG_MODELS.keys()), help="Expected song model")
    parser.add_argument("--plot", action="store_true", help="Show timeline plot")
    args = parser.parse_args()
    run(args.mp3, args.model, plot=args.plot)


if __name__ == "__main__":
    main()