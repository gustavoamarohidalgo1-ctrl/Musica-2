"""Audio -> instrumentos separados -> notas MIDI.

Flujo:
  1. Demucs (htdemucs_6s) separa la canción en 6 pistas.
  2. Cada pista con notas se transcribe:
       - piano  -> ByteDance piano_transcription (alta precisión) o Basic Pitch
       - resto  -> Basic Pitch
  3. Se guardan los .mid, los audios en .mp3 y un JSON con todas las notas.
"""
from __future__ import annotations

import json
import subprocess
import threading
import warnings
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PIANO_CHECKPOINT = ROOT / "models" / "piano_transcription.pth"

# Etiqueta, color y programa General MIDI de cada pista.
STEMS = {
    "piano":  {"label": "Piano",          "color": "#4f8cff", "program": 0},
    "guitar": {"label": "Guitarra",       "color": "#f5a524", "program": 25},
    "bass":   {"label": "Bajo",           "color": "#e5484d", "program": 33},
    "vocals": {"label": "Voz (melodía)",  "color": "#30a46c", "program": 52},
    "other":  {"label": "Otros",          "color": "#a78bfa", "program": 48},
    "drums":  {"label": "Batería",        "color": "#8b8d98", "program": 0},
    "mix":    {"label": "Audio completo", "color": "#4f8cff", "program": 0},
    # Modo «Teclado con Style»
    "voice":  {"label": "Voice (mano derecha)", "color": "#ffb224", "program": 0},
    "acomp":  {"label": "Acordes y pads",       "color": "#7c8cff", "program": 48},
    "canto":  {"label": "Voz humana (canto o habla)", "color": "#30a46c", "program": 52},
}
# La batería no tiene notas afinadas: se separa pero no se transcribe.
NO_TRANSCRIBE = {"drums"}
# Una pista con menos energía que este % de la pista más fuerte se considera vacía.
EMPTY_RATIO = 0.06
# Duración mínima (s) de un recorte.
MIN_CLIP = 1.0

_models: dict = {}
_models_lock = threading.Lock()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _demucs():
    with _models_lock:
        if "demucs" not in _models:
            from demucs.pretrained import get_model
            m = get_model("htdemucs_6s")
            m.eval()
            _models["demucs"] = m
        return _models["demucs"]


def _basic_pitch():
    with _models_lock:
        if "basic_pitch" not in _models:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model
            _models["basic_pitch"] = Model(ICASSP_2022_MODEL_PATH)
        return _models["basic_pitch"]


def _bytedance():
    with _models_lock:
        if "bytedance" not in _models:
            from piano_transcription_inference import PianoTranscription
            _models["bytedance"] = PianoTranscription(
                checkpoint_path=str(PIANO_CHECKPOINT), device=torch.device("cpu")
            )
        return _models["bytedance"]


def bytedance_available() -> bool:
    return PIANO_CHECKPOINT.exists() and PIANO_CHECKPOINT.stat().st_size > 1.6e8


# --------------------------------------------------------------------------- #
# Pasos
# --------------------------------------------------------------------------- #

def _to_mp3(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "192k", str(dst)],
        check=True,
    )


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _cut(src: Path, dst: Path, start: float, end: float | None) -> None:
    """Recorta src entre start y end (segundos; end=None = hasta el final) a WAV."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    subprocess.run(cmd + ["-vn", "-ac", "2", "-ar", "44100", str(dst)], check=True)


def _clock(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def separate(input_path: Path, out_dir: Path, on_progress) -> dict[str, Path]:
    """Separa la canción en pistas WAV. Devuelve {nombre: ruta}."""
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio

    model = _demucs()
    wav = AudioFile(input_path).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels
    )
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    length = wav.shape[-1]

    def cb(d):
        if d.get("state") != "end":
            return
        models = d.get("models", 1)
        done = d.get("segment_offset", 0) + d.get("model_idx_in_bag", 0) * length
        on_progress(min(done / max(length * models, 1), 1.0))

    with torch.no_grad():
        sources = apply_model(
            model, wav[None], device=_device(), split=True, overlap=0.25,
            progress=False, callback=cb,
        )[0]
    sources = sources * std + mean

    paths = {}
    for name, source in zip(model.sources, sources):
        p = out_dir / f"{name}.wav"
        save_audio(source.cpu(), str(p), model.samplerate)
        paths[name] = p
    return paths


def _loudness(path: Path) -> float:
    audio, _ = sf.read(str(path), dtype="float32")
    return float(np.sqrt(np.mean(audio ** 2)))


def transcribe_basic_pitch(path: Path) -> list[dict]:
    from basic_pitch.inference import predict

    _, _, events = predict(str(path), _basic_pitch())
    notes = [
        {"start": float(s), "end": float(e), "pitch": int(p),
         "velocity": int(np.clip(round(a * 127), 1, 127))}
        for s, e, p, a, _ in events
    ]
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def transcribe_piano(path: Path) -> list[dict]:
    import librosa
    from piano_transcription_inference import sample_rate

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    result = _bytedance().transcribe(audio, None)
    notes = [
        {"start": float(e["onset_time"]), "end": float(e["offset_time"]),
         "pitch": int(e["midi_note"]), "velocity": int(e["velocity"])}
        for e in result["est_note_events"]
    ]
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def write_midi(tracks: list[dict], path: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    for t in tracks:
        inst = pretty_midi.Instrument(program=t["program"], name=t["label"])
        for n in t["notes"]:
            inst.notes.append(pretty_midi.Note(
                velocity=n["velocity"], pitch=n["pitch"], start=n["start"], end=n["end"]))
        pm.instruments.append(inst)
    pm.write(str(path))


# --------------------------------------------------------------------------- #
# Trabajo completo
# --------------------------------------------------------------------------- #

def run_job(job_dir: Path, input_path: Path, mode: str, piano_hq: bool, update,
            clip: dict | None = None) -> dict:
    """mode: 'separar' | 'piano' | 'instrumento'.  update(step, progress).

    clip: {"start": s, "end": e | None} para procesar solo ese tramo. Los tiempos del
    resultado (notas, MIDI, audios) empiezan en 0 al inicio del tramo.
    """
    clip_wav = job_dir / "clip.wav"
    try:
        return _run_job(job_dir, input_path, mode, piano_hq, update, clip, clip_wav)
    finally:
        clip_wav.unlink(missing_ok=True)


def _run_job(job_dir, input_path, mode, piano_hq, update, clip, clip_wav) -> dict:
    use_bytedance = piano_hq and bytedance_available()

    source = input_path
    if clip:
        update("Recortando audio…", 0.01)
        _cut(input_path, clip_wav, clip["start"], clip.get("end"))
        try:
            clip_len = _duration(clip_wav)
        except ValueError:  # ffprobe da "N/A" para un WAV vacío
            clip_len = 0.0
        if clip_len < 0.05:
            raise ValueError(
                f"El recorte quedó vacío: el inicio ({_clock(clip['start'])}) está después "
                "del final de la canción.")
        if clip_len < MIN_CLIP:
            raise ValueError(
                f"El tramo elegido dura solo {clip_len:.1f} s porque la canción termina ahí; "
                f"el mínimo es {MIN_CLIP:g} s.")
        source = clip_wav

    update("Preparando audio…", 0.02)
    _to_mp3(source, job_dir / "original.mp3")
    duration = _duration(job_dir / "original.mp3")

    if mode == "teclado":
        return _run_teclado(job_dir, source, duration, update)

    if mode == "separar":
        update("Separando instrumentos (Demucs)…", 0.05)
        stems = separate(source, job_dir, lambda p: update(None, 0.05 + 0.45 * p))
    else:
        wav = job_dir / "mix.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                        "-ac", "2", "-ar", "44100", str(wav)], check=True)
        stems = {"mix": wav}

    loud = {name: _loudness(p) for name, p in stems.items()}
    loudest = max(loud.values()) or 1.0

    order = ["piano", "guitar", "bass", "vocals", "other", "mix", "drums"]
    names = [n for n in order if n in stems]
    tracks, skipped = [], []
    base = 0.5 if mode == "separar" else 0.1
    for i, name in enumerate(names):
        meta = STEMS[name]
        frac = base + (0.95 - base) * i / len(names)
        if loud[name] < EMPTY_RATIO * loudest:
            skipped.append(meta["label"])
            stems[name].unlink(missing_ok=True)
            continue

        notes, engine = [], None
        if name not in NO_TRANSCRIBE:
            is_piano = name == "piano" or (name == "mix" and mode == "piano")
            if is_piano and use_bytedance:
                engine = "ByteDance (piano)"
                update(f"Transcribiendo {meta['label'].lower()} con ByteDance…", frac)
                notes = transcribe_piano(stems[name])
            else:
                engine = "Basic Pitch"
                update(f"Transcribiendo {meta['label'].lower()} con Basic Pitch…", frac)
                notes = transcribe_basic_pitch(stems[name])
            for n in notes:
                n["end"] = min(n["end"], duration)
            notes = [n for n in notes if n["end"] > n["start"]]

        update(None, frac)
        audio_file = f"{name}.mp3"
        _to_mp3(stems[name], job_dir / audio_file)
        stems[name].unlink(missing_ok=True)

        label = "Piano" if (name == "mix" and mode == "piano") else meta["label"]
        track = {
            "id": name, "label": label, "color": meta["color"],
            "program": meta["program"], "engine": engine,
            "audio": audio_file, "notes": notes,
            "transcribed": name not in NO_TRANSCRIBE,
        }
        if notes:
            track["midi"] = f"{name}.mid"
            write_midi([track], job_dir / track["midi"])
        tracks.append(track)

    with_notes = [t for t in tracks if t["notes"]]
    if with_notes:
        write_midi(with_notes, job_dir / "todo.mid")

    update("Listo", 1.0)
    return {
        "duration": duration,
        "tracks": tracks,
        "skipped": skipped,
        "combined_midi": "todo.mid" if with_notes else None,
    }


def _run_teclado(job_dir: Path, source: Path, duration: float, update) -> dict:
    """Canto / Voice (mano derecha) / Style (acompañamiento) de una grabación de teclado."""
    # Demucs lee cualquier formato con ffmpeg, pero soundfile no (m4a, mp4, aac, webm): se usa un WAV.
    wav = job_dir / "teclado_src.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                    "-vn", "-ac", "2", "-ar", "44100", str(wav)], check=True)
    stems: dict[str, Path] = {}
    try:
        return _teclado_tracks(job_dir, wav, duration, update, stems)
    finally:  # también si falla: las pistas WAV de Demucs ocupan cientos de MB
        wav.unlink(missing_ok=True)
        for p in stems.values():
            p.unlink(missing_ok=True)


def _teclado_tracks(job_dir: Path, source: Path, duration: float, update, stems: dict) -> dict:
    from . import teclado

    update("Separando instrumentos (Demucs)…", 0.05)
    stems.update(separate(source, job_dir, lambda p: update(None, 0.05 + 0.40 * p)))
    update("Buscando la Voice y el Style…", 0.45)
    res = teclado.analyze(source, stems, job_dir / "_teclado", lambda p: update(None, 0.45 + 0.45 * p))

    def clean(notes):
        out = []
        for n in notes:
            c = {"start": float(n["start"]), "end": min(float(n["end"]), duration),
                 "pitch": int(n["pitch"]), "velocity": int(np.clip(n["velocity"], 1, 127))}
            if "p" in n:  # probabilidad de ser Voice: la usa el control «Separación Voice ↔ Style»
                c["p"] = round(float(n["p"]), 3)
            if c["end"] > c["start"]:
                out.append(c)
        return sorted(out, key=lambda n: (n["start"], n["pitch"]))

    # El Style que no es el bajo: acordes, pads y frases.
    bass = clean(res["bass"])
    by_pitch: dict[int, list[float]] = {}
    for n in bass:
        by_pitch.setdefault(n["pitch"], []).append(n["start"])
    acomp = [n for n in clean(res["style"])
             if not any(abs(s - n["start"]) <= 0.05 for s in by_pitch.get(n["pitch"], []))]

    update("Guardando pistas…", 0.92)
    audio = {}
    for name in ("bass", "drums", "vocals"):
        audio[name] = f"{name}.mp3"
        _to_mp3(stems[name], job_dir / audio[name])

    # Mezclas para escuchar: sin batería ni bajo, y sin canto.
    mixes = []
    wav = {k: sf.read(str(p), dtype="float32", always_2d=True)[0] for k, p in stems.items()}
    n = min(len(v) for v in wav.values())
    harm_parts = ["piano", "guitar", "other"] + ([] if res["singer"] else ["vocals"])
    for name, label, parts in (
        ("sin_bateria_bajo", "Sin batería ni bajo", harm_parts),
        ("sin_canto", "Sin voz humana", ["piano", "guitar", "bass", "other", "drums"]),
    ):
        if name == "sin_canto" and not res["singer"]:
            continue
        tmp = job_dir / f"{name}.wav"
        mixed = sum(wav[k][:n] for k in parts)
        sf.write(tmp, np.clip(mixed, -1, 1), 44100)
        _to_mp3(tmp, job_dir / f"{name}.mp3")
        tmp.unlink(missing_ok=True)
        mixes.append({"file": f"{name}.mp3", "label": label})
    for p in stems.values():
        p.unlink(missing_ok=True)

    def track(tid, group, notes, audio_file=None, transcribed=True, engine="Basic Pitch"):
        meta = STEMS[tid]
        t = {"id": tid, "group": group, "label": meta["label"], "color": meta["color"],
             "program": meta["program"], "engine": engine if transcribed else None,
             "audio": audio_file, "notes": notes, "transcribed": transcribed}
        if notes:
            t["midi"] = f"{tid}.mid"
            write_midi([t], job_dir / t["midi"])
        return t

    tracks = [
        track("voice", "voice", clean(res["voice"]), engine="Basic Pitch + separación Voice/Style"),
        track("bass", "style", bass, audio["bass"]),
        track("acomp", "style", acomp, engine="Basic Pitch + separación Voice/Style"),
        track("drums", "style", [], audio["drums"], transcribed=False),
    ]
    if res["singer"]:
        tracks.append(track("canto", "canto", clean(res["canto"]), audio["vocals"]))
    else:
        (job_dir / audio["vocals"]).unlink(missing_ok=True)

    with_notes = [t for t in tracks if t["notes"]]
    if with_notes:
        write_midi(with_notes, job_dir / "todo.mid")
    update("Listo", 1.0)
    return {
        "duration": duration,
        "tracks": tracks,
        "skipped": [],
        "mixes": mixes,
        "singer": res["singer"],
        "voice_thr": teclado.voice_style.threshold(),
        "combined_midi": "todo.mid" if with_notes else None,
    }


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)
