#!/usr/bin/env python3
"""
Standalone script: run the harmony HMM pitch-accuracy algorithm on an MP3.

Usage:
  python run_pitch_accuracy.py --mp3 <path_to.mp3> --model ode_to_joy [--plot]

  Or edit INPUT_MP3 and MODEL_NAME at the top of this file and run:
  python run_pitch_accuracy.py

Models: expected melody + harmony (pitches and beat durations) and BPM. To add a new
song, add a function that returns (mel_pitches, mel_beats, harm_pitches, harm_beats,
bpm_ref, merge_adjacent_same) and register it in SONG_MODELS.
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Config (override with CLI)
# ---------------------------------------------------------------------------
INPUT_MP3 = "tests/piano-synth/target_music/ode_to_joy_harmony_missed_notes.mp3"
MODEL_NAME = "ode_to_joy"
SR = 22050
HOP_LENGTH = 512
# C2 in Hz (avoid librosa.note_to_hz at import for compatibility)
FMIN = 440.0 * (2.0 ** ((36 - 69) / 12.0))  # ~65.4
N_BINS = 84
BINS_PER_OCTAVE = 12


# ---------------------------------------------------------------------------
# Song models: melody + harmony (pitches, beats), BPM, merge_adjacent_same
# ---------------------------------------------------------------------------
def _ode_to_joy_model():
    E_b, Q_b = 1.0, 2.0
    E4, F4, G4, D4, C4, C3, G3 = 64, 65, 67, 62, 60, 48, 55
    mel = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4, E4, D4, D4]
    mel_beats = [E_b] * 8 + [E_b] * 4 + [1.5 * E_b, 0.5 * E_b, Q_b]
    harm = [C3, G3, C3, G3]
    harm_beats = [4.0 * E_b] * 4
    bpm_ref = 60.0 / 0.44  # ~136.36
    return mel, mel_beats, harm, harm_beats, bpm_ref, False


SONG_MODELS = {
    "ode_to_joy": _ode_to_joy_model,
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
def mp3_to_cqt_db(audio_path, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12):
    if fmin is None:
        fmin = librosa.note_to_hz("C2")
    y, sr_used = librosa.load(audio_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    C = librosa.cqt(y=y, sr=sr_used, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
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
    tpl = np.ones(n_bins, float) / (np.linalg.norm(np.ones(n_bins)) + 1e-9)
    return tpl


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def hz_to_cqt_bin(hz, fmin, bins_per_octave):
    return int(np.round(bins_per_octave * np.log2(hz / fmin)))


def note_template_cqt(midi, fmin, n_bins, bins_per_octave, window=1):
    tpl = np.zeros(n_bins, float)
    f0 = midi_to_hz(midi)
    harm_w = {1: 1.0, 2: 0.6, 3: 0.4, 4: 0.25}
    for h, w in harm_w.items():
        k = hz_to_cqt_bin(h * f0, fmin, bins_per_octave)
        for kk in range(k - window, k + window + 1):
            if 0 <= kk < n_bins:
                tpl[kk] += w
    tpl /= (np.linalg.norm(tpl) + 1e-9)
    return tpl


def chord_template_cqt(pitches, fmin, n_bins, bins_per_octave, window=1):
    tpl = np.zeros(n_bins, dtype=float)
    for midi in pitches:
        tpl += note_template_cqt(midi, fmin, n_bins, bins_per_octave, window=window)
    tpl /= (np.linalg.norm(tpl) + 1e-9)
    return tpl


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
    end_selfloop=True,
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


def build_ASR_templates_segments(states, fmin, n_bins, bins_per_octave, window=1):
    sil = silence_template(n_bins)
    TPL = []
    for st in states:
        if st["phase"] in ("A", "S"):
            TPL.append(chord_template_cqt(st["pitches"], fmin, n_bins, bins_per_octave, window=window))
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
    B = scale * scores
    return B, scores


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


def viterbi_ASR(A, pi, B, on01=None, boundary_boost_mask=None, bonus=3.0, same_pitch_boundary_mask=None, same_pitch_bonus=0.0):
    S, T = B.shape
    logA = np.where(A > 0, np.log(A), -np.inf)
    logpi = np.where(pi > 0, np.log(pi), -np.inf)
    dp = np.full((S, T), -np.inf)
    bp = np.zeros((S, T), dtype=int)
    dp[:, 0] = logpi + B[:, 0]
    use_bonus = (on01 is not None) and (boundary_boost_mask is not None)
    use_same_pitch = (on01 is not None) and (same_pitch_boundary_mask is not None) and (same_pitch_bonus != 0)
    for t in range(1, T):
        logA_t = logA
        if use_bonus:
            logA_t = logA.copy()
            logA_t[boundary_boost_mask] += bonus * on01[t]
        if use_same_pitch:
            if not use_bonus:
                logA_t = logA.copy()
            logA_t[same_pitch_boundary_mask] += same_pitch_bonus * on01[t]
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
# Segment note presence (simple algorithm)
# ---------------------------------------------------------------------------
def simple_segment_note_presence(
    C_db,
    path,
    states,
    expected_segments,
    sr,
    hop_length,
    fmin,
    bins_per_octave=12,
    pitch_window=3,
    min_energy_db_below_max=25,
    quiet_percentile=25,
    quiet_margin_db=15,
    max_semitones_off=2,
):
    n_bins, T = C_db.shape

    def bin_to_approx_midi(bin_idx):
        hz = fmin * (2.0 ** (bin_idx / bins_per_octave))
        return int(round(69 + 12 * np.log2(hz / 440.0)))

    frame_ranges = get_segment_frame_ranges(path, states, sr, hop_length)
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
        bin_energy = np.max(slice_db, axis=1)
        midi_e = {}
        for b in range(n_bins):
            e = float(bin_energy[b])
            m = bin_to_approx_midi(b)
            if m not in midi_e or e > midi_e[m]:
                midi_e[m] = e
        segment_max_energies.append(max(midi_e.values()) if midi_e else -np.inf)
    valid = np.array([e for e in segment_max_energies if np.isfinite(e)])
    quiet_thresh = (np.percentile(valid, quiet_percentile) - quiet_margin_db) if len(valid) else -np.inf

    results = []
    for seg_idx, (exp_pitches, exp_dur_beats) in enumerate(expected_segments):
        rec = {
            "expected_idx": seg_idx,
            "expected_pitches": tuple(exp_pitches),
            "present_midi": [],
            "missing_midi": [],
            "extra_midi": [],
            "status": "rest",
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
        bin_energy = np.max(slice_db, axis=1)
        midi_energy = {}
        for b in range(n_bins):
            e = float(bin_energy[b])
            m = bin_to_approx_midi(b)
            if m not in midi_energy or e > midi_energy[m]:
                midi_energy[m] = e
        if not midi_energy:
            rec["status"] = "rest"
            results.append(rec)
            continue
        exp_set = {p for p in exp_pitches if p > 0}
        if not exp_set:
            strongest = sorted(midi_energy.items(), key=lambda x: x[1], reverse=True)[:2]
            rec["present_midi"] = sorted(m for m, _ in strongest)
            rec["status"] = "rest"
            results.append(rec)
            continue
        max_energy = max(midi_energy.values())
        if max_energy < quiet_thresh:
            rec["present_midi"] = []
            rec["missing_midi"] = sorted(exp_set)
            rec["extra_midi"] = []
            rec["status"] = "missing"
            results.append(rec)
            continue
        energy_thresh = max_energy - min_energy_db_below_max
        detected = []
        for p in sorted(exp_set):
            window = [m for m in midi_energy if abs(m - p) <= pitch_window]
            if not window:
                continue
            best_m = max(window, key=lambda m: midi_energy[m])
            best_e = midi_energy[best_m]
            if best_e >= energy_thresh and abs(best_m - p) <= max_semitones_off:
                detected.append(best_m)
        present_midi = set(detected)
        missing = sorted(exp_set - present_midi)
        extra = sorted(present_midi - exp_set)
        rec["present_midi"] = sorted(present_midi)
        rec["missing_midi"] = missing
        rec["extra_midi"] = extra
        if missing and extra:
            rec["status"] = "missing_and_wrong"
        elif missing:
            rec["status"] = "missing"
        elif extra:
            rec["status"] = "wrong"
        else:
            rec["status"] = "ok"
        results.append(rec)
    return results


def print_simple_segment_notes(results, max_rows=40):
    for r in results[:max_rows]:
        idx = r["expected_idx"]
        status = r["status"]
        exp = r["expected_pitches"]
        missing = r["missing_midi"]
        extra = r["extra_midi"]
        present = r["present_midi"]
        print(f"Seg {idx:2d}  expected={exp}  status={status:12s}  present={present}", end="")
        if missing:
            print(f"  missing={missing}", end="")
        if extra:
            print(f"  extra={extra}", end="")
        print()


def evaluate_pitch_accuracy_from_simple(simple_results):
    total = len(simple_results)
    correct_count = sum(1 for r in simple_results if r["status"] == "ok")
    wrong_count = total - correct_count
    accuracy = correct_count / total if total > 0 else 0.0
    skipped_count = sum(1 for r in simple_results if r["status"] == "skipped")
    return {
        "results": simple_results,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
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
    print(f"Overall Accuracy: {eval_result['accuracy']:.1%}")
    print(f"Correct: {eval_result['correct_count']}/{eval_result['total_count']}")
    print(f"Skipped: {eval_result['skipped_count']}/{eval_result['total_count']}")
    print(f"Wrong:   {eval_result['wrong_count']}/{eval_result['total_count']}")
    print()
    if show_details:
        print("Detailed Results:")
        print("-" * 70)
        print(f"{'Idx':<5} {'Status':<16} {'Expected':<25} {'Detected':<25} {'Overlap':<10}")
        print("-" * 70)
        for r in eval_result["results"][:max_rows]:
            exp_pitches_str = str(r.get("expected_pitches", ())) if r.get("expected_pitches") else "()"
            status_icon = {"correct": "✓", "ok": "✓", "skipped": "✗", "missing": "✗", "wrong": "⚠", "missing_and_wrong": "⚠"}.get(r["status"], "?")
            det_pitches_str = str(r.get("detected_pitches") or r.get("present_midi", [])) if (r.get("detected_pitches") is not None or r.get("present_midi") is not None) else "N/A"
            overlap_str = f"{r['overlap_ratio']:.1%}" if r.get("overlap_ratio") is not None else "—"
            print(f"{r['expected_idx']:<5} {status_icon} {r['status']:<15} {exp_pitches_str:<25} {det_pitches_str:<25} {overlap_str}")
        if len(eval_result["results"]) > max_rows:
            print(f"... ({len(eval_result['results']) - max_rows} more segments)")
    print()
    if eval_result.get("skipped_segments"):
        print(f"Skipped segment indices: {eval_result['skipped_segments']}")
    if eval_result.get("wrong_segments"):
        print(f"Wrong segment indices: {eval_result['wrong_segments']}")
    print("=" * 70)


def plot_accuracy_timeline_from_simple(simple_results, title="Pitch accuracy (from note presence)"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(14, 3))
    colors = {
        "ok": "green",
        "fully_missing": "darkred",
        "partially_missing": "coral",
        "wrong": "orange",
        "missing_and_wrong": "purple",
        "skipped": "gray",
        "rest": "lightgray",
    }
    for r in simple_results:
        status = r.get("status", "rest")
        start, end = r.get("start_time"), r.get("end_time")
        if start is None or end is None:
            continue
        if status == "missing":
            exp_pitches = r.get("expected_pitches") or ()
            exp_set = {p for p in exp_pitches if p > 0}
            present = r.get("present_midi") or []
            display = "fully_missing" if (not exp_set or len(present) == 0) else "partially_missing"
        elif status == "missing_and_wrong":
            display = "missing_and_wrong"
        else:
            display = status
        c = colors.get(display, "gray")
        ax.barh(0.5, end - start, left=start, height=0.4, color=c, alpha=0.8, edgecolor="black", linewidth=0.4)
        if display != "ok":
            ax.text((start + end) / 2, 0.5, f"{r['expected_idx']}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([0.5])
    ax.set_yticklabels(["Segment"])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    legend_elements = [
        Patch(facecolor="green", alpha=0.8, label="Correct (ok)"),
        Patch(facecolor="darkred", alpha=0.8, label="Fully missing"),
        Patch(facecolor="coral", alpha=0.8, label="Partially missing"),
        Patch(facecolor="orange", alpha=0.8, label="Wrong"),
        Patch(facecolor="purple", alpha=0.8, label="Missing + wrong"),
        Patch(facecolor="gray", alpha=0.8, label="Skipped / rest"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
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
    expected_segments = combine_tracks_to_segments(mel_pitches, mel_beats, harm_pitches, harm_beats, merge_adjacent_same=merge)
    segments = expected_segments  # same for HMM

    print(f"Loading: {mp3_path}")
    C, y, sr = mp3_to_cqt_db(str(mp3_path), sr=SR, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    dt_ms = dt_ms_from_cqt(sr, HOP_LENGTH)
    onset_env = onset_strength_envelope(y, sr, HOP_LENGTH)

    print("Building HMM and running Viterbi...")
    states, A, pi = build_ASR_states_segments(
        segments, bpm_ref, dt_ms,
        frac_attack=0.15, frac_release=0.15,
        min_A=1, min_S=1, min_R=1,
        p_start_to_first=0.02, end_selfloop=True, tempo_slack=1.02,
    )
    TPL = build_ASR_templates_segments(states, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, window=1)
    B, _ = emissions_ASR(C, TPL, onset_env, states, scale_pitch=2.0, w_attack_onset=2.0, w_sustain_onset=-0.5, w_release_onset=-1.0, w_release_energy=-0.5)
    mask = boundary_mask_segments(states)
    same_pitch_mask = np.zeros_like(mask)
    for i in range(len(states)):
        for j in range(len(states)):
            if mask[i, j] and states[i].get("pitches") == states[j].get("pitches"):
                same_pitch_mask[i, j] = True
    path, _ = viterbi_ASR(A, pi, B, on01=onset_env, boundary_boost_mask=mask, bonus=6.0, same_pitch_boundary_mask=same_pitch_mask, same_pitch_bonus=4.0)

    print("Segment note presence...")
    simple_results = simple_segment_note_presence(
        C, path, states, expected_segments,
        sr=sr, hop_length=HOP_LENGTH, fmin=FMIN,
        bins_per_octave=BINS_PER_OCTAVE,
        pitch_window=3, min_energy_db_below_max=25,
        quiet_percentile=25, quiet_margin_db=15, max_semitones_off=2,
    )

    print_simple_segment_notes(simple_results, max_rows=50)
    eval_simple = evaluate_pitch_accuracy_from_simple(simple_results)
    print_accuracy_report(eval_simple, show_details=True, max_rows=30)

    if plot:
        plot_accuracy_timeline_from_simple(simple_results, title=f"Pitch accuracy: {mp3_path.name} (model={model_name})")


def main():
    parser = argparse.ArgumentParser(description="Run harmony HMM pitch accuracy on an MP3.")
    parser.add_argument("--mp3", default=INPUT_MP3, help="Path to input MP3")
    parser.add_argument("--model", default=MODEL_NAME, choices=list(SONG_MODELS.keys()), help="Expected song model")
    parser.add_argument("--plot", action="store_true", help="Show timeline plot")
    args = parser.parse_args()
    run(args.mp3, args.model, plot=args.plot)


if __name__ == "__main__":
    main()
