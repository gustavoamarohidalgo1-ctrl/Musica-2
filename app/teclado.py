"""Modo «Teclado con Style»: separa lo que suena en una grabación de teclado arranger en
  - Canto: la voz humana, si la hay (pista «vocals» de Demucs)
  - Voice: lo que toca la mano derecha
  - Style: el acompañamiento automático (bajo, acordes y pads, batería)

El algoritmo Voice/Style vive en voice_style.py y se eligió con un banco de pruebas de
canciones sintéticas con la verdad exacta. Aquí se le preparan los mismos datos que en
ese banco: pistas mono a 22 050 Hz, transcripciones de Basic Pitch, energía y pulsos.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from . import voice_style

SR = 22050          # la frecuencia a la que trabaja Basic Pitch
HOP = 0.05          # energía por pista cada 50 ms
MIN_NOTE_MS = 58    # nota mínima de Basic Pitch (por defecto descarta notas < 128 ms)
# Si la pista de voz de Demucs lleva más de esta fracción de la energía, hay alguien cantando.
SINGER_SHARE = 0.2
DEMUCS_STEMS = ("piano", "guitar", "bass", "vocals", "other", "drums")


def transcribe(path: Path) -> list[dict]:
    from basic_pitch.inference import predict

    from .pipeline import _basic_pitch

    _, _, events = predict(str(path), _basic_pitch(), minimum_note_length=MIN_NOTE_MS)
    notes = [{"start": float(s), "end": float(e), "pitch": int(p),
              "velocity": int(np.clip(round(a * 127), 1, 127)), "amp": float(a)}
             for s, e, p, a, _ in events]
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def analyze(mix_path: Path, stems: dict[str, Path], work: Path, on_progress) -> dict:
    """Devuelve {"voice", "style", "bass", "canto"} (listas de notas) y "singer" (bool).

    mix_path: audio completo; stems: pistas de Demucs (WAV); work: carpeta temporal.
    on_progress(fracción 0..1).
    """
    import librosa

    work.mkdir(parents=True, exist_ok=True)
    try:
        def mono(path):
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return librosa.resample(y.mean(1), orig_sr=sr, target_sr=SR)

        y = {name: mono(stems[name]) for name in DEMUCS_STEMS}
        n = min(len(v) for v in y.values())
        y = {k: v[:n] for k, v in y.items()}
        energy = {k: float(np.mean(v ** 2)) for k, v in y.items()}
        vocal_share = energy["vocals"] / (sum(energy.values()) + 1e-12)
        singer = vocal_share > SINGER_SHARE

        mix = mono(mix_path)[:n]
        y["mix"] = mix
        y["harm"] = y["piano"] + y["guitar"] + y["vocals"] + y["other"]
        y["inst"] = sum(y[k] for k in DEMUCS_STEMS if k != "vocals")

        stems_dir = work / "stems"
        stems_dir.mkdir(exist_ok=True)
        hop = int(SR * HOP)
        rms = {}
        for name, v in y.items():
            sf.write(stems_dir / f"{name}.wav", np.clip(v, -1, 1), SR, subtype="PCM_16")
            rms[name] = librosa.feature.rms(y=v, frame_length=2 * hop, hop_length=hop)[0].tolist()
        tempo, beats = librosa.beat.beat_track(y=mix, sr=SR, units="time")

        needed = list(dict.fromkeys([*voice_style.SOURCES, "bass", "vocals"]))
        notes = {}
        for i, name in enumerate(needed):
            notes[name] = transcribe(stems_dir / f"{name}.wav")
            on_progress((i + 1) / (len(needed) + 1))

        data = {
            "notes": notes,
            "feat": {"duration": n / SR, "tempo": float(np.atleast_1d(tempo)[0]),
                     "beats": [float(b) for b in beats], "rms_hop": HOP, "rms": rms,
                     "vocal_share": vocal_share},
            "stems_dir": stems_dir,
        }
        out = voice_style.separate(data)
        on_progress(1.0)
        return {
            "voice": out["voice"],
            "style": out["style"],
            "bass": notes["bass"],
            "canto": notes.get("vocals", []) if singer else [],
            "singer": singer,
            "vocal_share": vocal_share,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
