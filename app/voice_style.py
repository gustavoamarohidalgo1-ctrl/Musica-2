"""Separación Voice (mano derecha) / Style (acompañamiento) para el modo «Teclado con Style».

Origen: se eligió en un banco de pruebas con 82 canciones sintéticas de teclado arranger (Style
+ Voice con la verdad exacta; 18 con un cantante real mezclado), entre 8 algoritmos hechos por
agentes en dos rondas. Este fue el mejor en 22 canciones ocultas: F1 de las notas de la Voice
0,82 (0,81 sin canto, 0,84 con canto). Con grabaciones reales el acierto será algo menor.
La parte final («Uso en la app») procesa las canciones largas por trozos y guarda la
probabilidad de cada nota para el control «Separación Voice ↔ Style».

learned2: clasificador Voice / Style sobre notas candidatas (dos etapas de arboles poco profundos).

Entrada: data["notes"] (mix, inst, harm, vocals, bass), data["feat"] (beats, duration, vocal_share),
data["stems_dir"] (mix.wav e inst.wav para el nivel de la altura en el audio). SOURCES = mix, harm,
inst, vocals (el bajo siempre esta disponible).
  - Candidatas: union de las transcripciones mix, inst y harm (misma altura, inicio a <= 50 ms).
    inst = mezcla sin la pista de canto (Demucs); con canto (vocal_share > 0.2) el nivel en el audio
    se mide en inst.wav en lugar de mix.wav.
  - Canto: coincidencias con la transcripcion de vocals (solo se usan para bajar la probabilidad de
    Voice y para excluir del Style las notas que son canto cuando hay canto).
  - Rasgos (algo.FEATURES): altura, duracion, presencia en mix/inst/harm, cobertura por acorde,
    distancia al pulso y a la rejilla de semicorcheas, continuidad melodica (intervalo con la nota
    superior anterior y siguiente), bajo sonando, nivel relativo en el audio y canto.
  - Etapa 1 y etapa 2 (probabilidades de las notas vecinas): ensembles de HistGradientBoosting con
    3 niveles de profundidad, 150 arboles, exportados a model.json (junto a este archivo); la
    inferencia es numpy puro. Sin audio se usa el modelo de respaldo ("fallback" en model.json).
  - Voice = candidatas con probabilidad >= thr. Style = bajo + candidatas de harm que no son Voice
    ni canto.
Pesos: final.py (entrena con las 60 canciones DEV y escribe model.json; rows.py/cv.py validan).
"""
import json
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCES = ("mix", "harm", "inst", "vocals")
MERGE_TOL = 0.05     # fusionar mix/harm si la altura coincide y el inicio difiere <= 50 ms
ONSET_TOL = 0.05     # grupo de inicio (acorde)
VOC_TOL = 0.05       # coincidencia con una nota de canto
N_FFT = 4096
HOP = 512
SR = 22050
SING_THR = 0.2

FEATURES = [
    # base (etapa 1 de learned)
    "pitch_c", "pitch_pct", "dur_log", "dur_beats_log", "amp", "amp_pct", "in_mix", "in_inst", "in_harm",
    "n_onset", "top_onset", "below_top", "cov_above", "cov_below", "n_sound", "beat_dist",
    "rep_pitch_phase", "pitch_freq", "mix_lvl_rel", "mix_lvl_local", "amp_local",
    # nuevos
    "dev16_frac", "dev16_s", "int_prev", "int_next", "bass_pc", "bass_int",
    "lvl_win", "lvl_rank_on", "voc_match", "voc_ov", "voc_near",
    "voc_match_s", "voc_ov_s", "voc_near_s",
]

_MODEL_FILE = Path(__file__).resolve().parent / "voice_style_model.json"   # escrito por final.py
MODEL = None  # se carga de model.json la primera vez que se usa separate()

CTX_NAMES = [
    "q_self", "q_on_max", "q_ov_max", "q_above_max", "q_below_max", "q_rank_ov", "q_pct",
    "dp_prev_conf", "dp_next_conf", "q_win_mean",
]
STAGE2_NAMES = CTX_NAMES + FEATURES


def _read_wav(path):
    """WAV PCM mono o estereo -> float32 mono. None si no se puede leer."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, sw = w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (OSError, wave.Error, EOFError):
        return None
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def _power_spec(x):
    """Espectro de potencia (frames x bins); frame j -> j*HOP/SR."""
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad))
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    win = np.hanning(N_FFT).astype(np.float32)
    spec = np.fft.rfft(xp[idx] * win, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32), n_frames


@lru_cache(maxsize=256)
def _pitch_bins(p):
    """Bins de STFT alrededor del fundamental y del 2.o armonico de la altura p."""
    f0 = 440.0 * 2.0 ** ((p - 69) / 12.0)
    df = SR / N_FFT
    bins = []
    for h in (1, 2):
        f = h * f0
        kc = int(round(f / df))
        w = max(1, int(round(0.03 * f / df)))
        bins.extend(range(max(0, kc - w), min(N_FFT // 2, kc + w) + 1))
    return np.array(sorted(set(bins)), dtype=int)


def _stem_energy(P, n_frames, S, En, pitches):
    """Energia media de cada candidata (altura y 2.o armonico) en la ventana de la nota."""
    out = np.zeros(len(S))
    for i in range(len(S)):
        j0 = int(np.ceil(S[i] * SR / HOP))
        j1 = int(np.floor(En[i] * SR / HOP))
        j0 = min(max(j0, 0), n_frames - 1)
        j1 = min(max(j1, j0), n_frames - 1)
        cols = _pitch_bins(int(pitches[i]))
        out[i] = P[j0:j1 + 1][:, cols].sum(axis=1).mean()
    return out


def _candidates(sources):
    """Une las transcripciones (mezcla, mezcla sin canto, armonia) en candidatas unicas.

    sources: lista de (nombre, notas). Dos notas casan si tienen la misma altura y el inicio
    difiere <= MERGE_TOL; cada nota de una fuente casa como mucho con una candidata por fuente.
    """
    cands = []
    by_pitch = {}
    for src, notes in sources:
        for h in notes:
            p = int(h["pitch"])
            st = float(h["start"])
            best, bd = None, MERGE_TOL
            for j in by_pitch.get(p, []):
                c = cands[j]
                if c["in_" + src]:
                    continue
                d = abs(c["start"] - st)
                if d <= bd:
                    best, bd = j, d
            if best is None:
                c = {"start": st, "end": float(h["end"]), "pitch": p, "velocity": 0,
                     "in_mix": 0, "in_inst": 0, "in_harm": 0, "amps": []}
                cands.append(c)
                best = len(cands) - 1
                by_pitch.setdefault(p, []).append(best)
            c = cands[best]
            c["in_" + src] = 1
            c["amps"].append(float(h.get("amp", 0.5)))
            c["velocity"] = max(c["velocity"], int(h.get("velocity", 80)))
    for c in cands:
        c["amp"] = float(np.mean(c.pop("amps")))
    cands.sort(key=lambda c: (c["start"], c["pitch"]))
    return cands


def _pct(x):
    """Percentil (0..1) de cada valor dentro del vector."""
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / (n - 1)


def _beat_grid(beats, duration, biv):
    if len(beats) >= 2:
        return np.concatenate([[beats[0] - biv], beats, [beats[-1] + biv]])
    return np.arange(-biv, max(duration, 1.0) + 2 * biv, biv)


def _audio_path(data, singer):
    d = data.get("stems_dir")
    if d is None:
        return None
    d = Path(d)
    names = ("inst.wav", "mix.wav") if singer else ("mix.wav",)
    for nm in names:
        if (d / nm).exists():
            return d / nm
    return None


def _level(data, singer, S, En, P):
    """log10 de la energia de cada candidata en el audio (inst si hay canto). None si no hay audio."""
    path = _audio_path(data, singer)
    if path is None:
        return None
    x = _read_wav(path)
    if x is None:
        return None
    Pw, nf = _power_spec(x)
    lem = np.log10(_stem_energy(Pw, nf, S, En, P) + 1e-9)
    del Pw
    return lem


def build(data):
    """Devuelve (candidatas, matriz X [n x len(FEATURES)])."""
    notes = data["notes"]
    feat = data.get("feat", {}) or {}
    singer = float(feat.get("vocal_share", 0.0) or 0.0) > SING_THR
    mixn = notes.get("mix", []) or []
    instn = notes.get("inst", []) or []
    harm = notes.get("harm", []) or []
    voc = notes.get("vocals", []) or []
    bass = notes.get("bass", []) or []
    cands = _candidates([("mix", mixn), ("inst", instn), ("harm", harm)])
    n = len(cands)
    if n == 0:
        return cands, np.zeros((0, len(FEATURES)))

    S = np.array([c["start"] for c in cands])
    En = np.maximum(np.array([c["end"] for c in cands]), S + 0.02)
    P = np.array([c["pitch"] for c in cands], dtype=float)
    A = np.array([c["amp"] for c in cands])
    inm = np.array([c["in_mix"] for c in cands], dtype=float)
    ini = np.array([c["in_inst"] for c in cands], dtype=float)
    inh = np.array([c["in_harm"] for c in cands], dtype=float)
    dur = En - S
    F = {}

    # --- Altura y amplitud ---
    F["pitch_c"] = (P - 60.0) / 12.0
    F["pitch_pct"] = _pct(P)
    F["dur_log"] = np.log(dur + 0.05)
    F["amp"] = A
    F["amp_pct"] = _pct(A)
    F["in_mix"] = inm
    F["in_inst"] = ini
    F["in_harm"] = inh
    F["pitch_freq"] = np.array([np.sum(P == p) for p in P]) / n

    # --- Pulso: rejilla de beats y de semicorcheas ---
    beats = np.asarray(feat.get("beats", []), dtype=float)
    biv = float(np.median(np.diff(beats))) if len(beats) >= 3 else 0.5
    if not (0.2 <= biv <= 2.0):
        biv = 0.5
    duration = float(feat.get("duration", En.max()))
    grid = _beat_grid(beats, duration, biv)
    k = np.clip(np.searchsorted(grid, S, side="right") - 1, 0, len(grid) - 2)
    ivl = np.maximum(grid[k + 1] - grid[k], 1e-6)
    frac = np.clip((S - grid[k]) / ivl, 0.0, 1.0)
    F["beat_dist"] = 2.0 * np.minimum(frac, 1.0 - frac)
    F["dur_beats_log"] = np.log(dur / biv + 0.05)
    pos16 = grid[k][:, None] + (np.arange(5)[None, :] / 4.0) * ivl[:, None]
    d16 = np.abs(S[:, None] - pos16).min(1)
    F["dev16_frac"] = np.clip(d16 / (ivl / 4.0), 0.0, 1.0)
    F["dev16_s"] = np.minimum(d16, 0.1) / 0.1

    # --- Grupo de inicio y cobertura ---
    d_on = np.abs(S[:, None] - S[None, :])
    same_on = d_on <= ONSET_TOL
    np.fill_diagonal(same_on, False)
    F["n_onset"] = np.log1p(same_on.sum(1))
    cl_max = np.maximum(P, np.where(same_on, P[None, :], -1.0).max(1))
    top = P >= cl_max
    F["top_onset"] = top.astype(float)
    F["below_top"] = (cl_max - P) / 12.0

    sounding = (S[None, :] <= S[:, None]) & (En[None, :] > S[:, None])
    np.fill_diagonal(sounding, False)
    F["n_sound"] = np.log1p(sounding.sum(1))

    cov_a = np.zeros(n)
    cov_b = np.zeros(n)
    for i in range(n):
        ov = (S < En[i]) & (En > S[i])
        ov[i] = False
        hi = ov & (P > P[i])
        lo = ov & (P < P[i])
        cov_a[i] = _covered(S[i], En[i], S[hi], En[hi])
        cov_b[i] = _covered(S[i], En[i], S[lo], En[lo])
    F["cov_above"] = cov_a
    F["cov_below"] = cov_b

    # --- Repeticion del mismo patron (misma altura, misma posicion en el pulso) ---
    dph = np.abs(frac[:, None] - frac[None, :])
    dph = np.minimum(dph, 1.0 - dph)
    rep = ((P[:, None] == P[None, :]) & (dph <= 0.04) & (d_on > 0.5 * biv)).sum(1)
    F["rep_pitch_phase"] = np.log1p(rep)

    # --- Continuidad melodica: intervalo con la nota superior anterior y siguiente ---
    sky = np.where(top)[0]
    order = np.argsort(S[sky], kind="stable")
    sky = sky[order]
    ss, sp = S[sky], P[sky]
    int_prev = np.ones(n)
    int_next = np.ones(n)
    if len(sky):
        ib = np.searchsorted(ss, S - ONSET_TOL, side="left")      # cuantas notas superiores empiezan antes
        ia = np.searchsorted(ss, S + ONSET_TOL, side="right")     # primera que empieza despues
        has_p = ib > 0
        has_n = ia < len(sky)
        pp = sp[np.clip(ib - 1, 0, len(sky) - 1)]
        pn = sp[np.clip(ia, 0, len(sky) - 1)]
        int_prev = np.where(has_p, np.minimum(np.abs(P - pp), 12.0) / 12.0, 1.0)
        int_next = np.where(has_n, np.minimum(np.abs(P - pn), 12.0) / 12.0, 1.0)
    F["int_prev"] = int_prev
    F["int_next"] = int_next

    # --- Bajo sonando en el mismo momento ---
    bass_pc = np.zeros(n)
    bass_int = np.zeros(n)
    if len(bass):
        bS = np.array([b["start"] for b in bass], dtype=float)
        bE = np.array([b["end"] for b in bass], dtype=float)
        bP = np.array([b["pitch"] for b in bass], dtype=float)
        M = (bS[None, :] <= S[:, None] + ONSET_TOL) & (bE[None, :] > S[:, None] + ONSET_TOL)
        has = M.any(1)
        same_pc = ((P[:, None] - bP[None, :]) % 12 == 0) & M
        bass_pc = np.where(has, same_pc.any(1).astype(float), 0.0)
        last = np.where(M, bS[None, :], -np.inf).argmax(1)
        bass_int = np.where(has, np.clip((P - bP[last]) / 12.0, -2.0, 2.0) / 2.0, 0.0)
    F["bass_pc"] = bass_pc
    F["bass_int"] = bass_int

    # --- Nivel de la altura en el audio (mezcla sin canto si hay canto) ---
    lem = _level(data, singer, S, En, P)
    if lem is not None:
        F["mix_lvl_rel"] = np.clip(10.0 * (lem - lem.max()), -40.0, 0.0) / 40.0
        ov_all = (S[None, :] < En[:, None]) & (En[None, :] > S[:, None])
        loc_max = np.where(ov_all, lem[None, :], -np.inf).max(1)
        F["mix_lvl_local"] = np.clip(10.0 * (lem - loc_max), -40.0, 0.0) / 40.0
        near = d_on <= 0.25
        w_max = np.where(near, lem[None, :], -np.inf).max(1)
        F["lvl_win"] = np.clip(10.0 * (lem - w_max), -40.0, 0.0) / 40.0
        grp = same_on | np.eye(n, dtype=bool)
        below = (grp & (lem[None, :] < lem[:, None])).sum(1)
        gsize = grp.sum(1)
        F["lvl_rank_on"] = np.where(gsize > 1, below / np.maximum(gsize - 1, 1), 0.5)
    else:
        F["mix_lvl_rel"] = np.full(n, -1.0)
        F["mix_lvl_local"] = np.full(n, -1.0)
        F["lvl_win"] = np.full(n, -1.0)
        F["lvl_rank_on"] = np.full(n, 0.5)
    amp_max = np.where((S[None, :] < En[:, None]) & (En[None, :] > S[:, None]), A[None, :], -np.inf).max(1)
    F["amp_local"] = np.clip(A - amp_max, -1.0, 0.0)

    # --- Canto: coincidencias con la transcripcion de voces ---
    voc_match = np.zeros(n)
    voc_ov = np.zeros(n)
    voc_near = np.zeros(n)
    if len(voc):
        vS = np.array([v["start"] for v in voc], dtype=float)
        vE = np.array([v["end"] for v in voc], dtype=float)
        vP = np.array([v["pitch"] for v in voc], dtype=float)
        same_p = vP[None, :] == P[:, None]
        dstart = np.abs(vS[None, :] - S[:, None])
        voc_match = (same_p & (dstart <= VOC_TOL)).any(1).astype(float)
        ovl = np.minimum(En[:, None], vE[None, :]) - np.maximum(S[:, None], vS[None, :])
        ovl = np.where(same_p, np.clip(ovl / dur[:, None], 0.0, 1.0), 0.0)
        voc_ov = ovl.max(1)
        voc_near = ((np.abs(vP[None, :] - P[:, None]) <= 1.0) & (dstart <= 0.08)).any(1).astype(float)
    sg = 1.0 if singer else 0.0
    F["voc_match"] = voc_match
    F["voc_ov"] = voc_ov
    F["voc_near"] = voc_near
    F["voc_match_s"] = voc_match * sg
    F["voc_ov_s"] = voc_ov * sg
    F["voc_near_s"] = voc_near * sg

    X = np.column_stack([F[f] for f in FEATURES]).astype(float)
    return cands, X


def _covered(s, e, starts, ends):
    """Fraccion de la ventana [s, e] cubierta por la union de intervalos dados."""
    if e <= s or len(starts) == 0:
        return 0.0
    st = np.clip(starts, s, e)
    en = np.clip(ends, s, e)
    keep = en > st
    if not keep.any():
        return 0.0
    st, en = st[keep], en[keep]
    order = np.argsort(st)
    st, en = st[order], en[order]
    total = 0.0
    cs, ce = st[0], en[0]
    for a, b in zip(st[1:], en[1:]):
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    total += ce - cs
    return float(min(1.0, total / (e - s)))


def context_features(cands, q):
    """Rasgos de segunda etapa: probabilidades de la etapa 1 de las notas vecinas."""
    n = len(cands)
    out = np.zeros((n, len(CTX_NAMES)))
    if n == 0:
        return out
    q = np.asarray(q, dtype=float)
    S = np.array([c["start"] for c in cands])
    En = np.maximum(np.array([c["end"] for c in cands]), S + 0.02)
    P = np.array([c["pitch"] for c in cands], dtype=float)
    eye = np.eye(n, dtype=bool)
    d_on = np.abs(S[:, None] - S[None, :])
    same_on = (d_on <= ONSET_TOL) & ~eye
    ov = (S[None, :] < En[:, None]) & (En[None, :] > S[:, None]) & ~eye
    qm = q[None, :]
    qc = np.clip(q, 1e-4, 1 - 1e-4)
    out[:, 0] = np.clip(np.log(qc / (1 - qc)), -6, 6) / 6.0
    out[:, 1] = np.where(same_on, qm, 0.0).max(1)
    out[:, 2] = np.where(ov, qm, 0.0).max(1)
    out[:, 3] = np.where(ov & (P[None, :] > P[:, None]), qm, 0.0).max(1)
    out[:, 4] = np.where(ov & (P[None, :] < P[:, None]), qm, 0.0).max(1)
    cnt = ov.sum(1)
    lower = (ov & (q[None, :] < q[:, None])).sum(1)
    out[:, 5] = np.where(cnt > 0, lower / np.maximum(cnt, 1), 0.5)
    out[:, 6] = _pct(q)
    conf = q > 0.5
    for i in range(n):
        dp = [1.0, 1.0]
        prev = np.where(conf & (S < S[i] - ONSET_TOL))[0]
        if prev.size:
            j = prev[np.argmax(S[prev])]
            dp[0] = min(abs(P[i] - P[j]) / 12.0, 2.0) / 2.0
        nxt = np.where(conf & (S > S[i] + ONSET_TOL))[0]
        if nxt.size:
            j = nxt[np.argmin(S[nxt])]
            dp[1] = min(abs(P[i] - P[j]) / 12.0, 2.0) / 2.0
        out[i, 7], out[i, 8] = dp
    win = (d_on <= 1.0) & ~eye
    nwin = win.sum(1)
    out[:, 9] = np.where(nwin > 0, (win * qm).sum(1) / np.maximum(nwin, 1), 0.0)
    return out


def stage2_matrix(cands, X, q):
    return np.hstack([context_features(cands, q), X])


def _load_model():
    """Carga model.json (junto a este archivo) una sola vez; arrays de numpy por arbol."""
    global MODEL
    if MODEL is None and not _MODEL_FILE.exists():
        # Sin el modelo la Voice saldría vacía sin avisar: mejor un error claro en el trabajo.
        raise FileNotFoundError(f"Falta el modelo de separación Voice/Style: {_MODEL_FILE}")
    if MODEL is None:
        raw = json.loads(_MODEL_FILE.read_text())
        for part in [raw] + ([raw["fallback"]] if "fallback" in raw else []):
            for key in ("stage1", "stage2"):
                st = part[key]
                st["base"] = float(st["base"])
                st["trees"] = [{k: np.asarray(v) for k, v in t.items()} for t in st["trees"]]
        MODEL = raw
    return MODEL


def _tree_proba(stage, X):
    """Probabilidad de un ensemble de arboles (log-odds = base + suma de hojas; regla x <= umbral)."""
    n = X.shape[0]
    raw = np.full(n, stage["base"], dtype=float)
    for t in stage["trees"]:
        feat, thr, left, right = t["feature"], t["thr"], t["left"], t["right"]
        val, leaf, mleft = t["value"], t["leaf"], t["missing_left"]
        node = np.zeros(n, dtype=int)
        out = np.zeros(n)
        active = np.ones(n, dtype=bool)
        while active.any():
            idx = np.where(active)[0]
            nd = node[idx]
            lf = leaf[nd]
            out[idx[lf]] = val[nd[lf]]
            active[idx[lf]] = False
            ii = idx[~lf]
            if ii.size == 0:
                break
            n2 = node[ii]
            x = X[ii, feat[n2]]
            go_left = np.where(np.isnan(x), mleft[n2], x <= thr[n2])
            node[ii] = np.where(go_left, left[n2], right[n2])
        raw += out
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))


def _stage_cols(stage, all_names):
    return [all_names.index(nm) for nm in stage["names"]]


def _separate_core(data):
    notes = data["notes"]
    bass = notes.get("bass", []) or []
    feat = data.get("feat", {}) or {}
    singer = float(feat.get("vocal_share", 0.0) or 0.0) > SING_THR
    M = _load_model()
    if M is None:
        return {"voice": [], "style": [dict(b) for b in bass]}
    cands, X = build(data)
    if len(cands) == 0:
        return {"voice": [], "style": [dict(b) for b in bass]}
    # Sin audio de la mezcla (ni inst.wav) se usa el modelo de respaldo sin rasgos de nivel
    use = M if (_audio_path(data, singer) is not None or "fallback" not in M) else M["fallback"]
    q = _tree_proba(use["stage1"], X[:, _stage_cols(use["stage1"], FEATURES)])
    X2 = stage2_matrix(cands, X, q)
    prob = _tree_proba(use["stage2"], X2[:, _stage_cols(use["stage2"], STAGE2_NAMES)])
    thr = float(use["thr"])
    i_voc = FEATURES.index("voc_match")
    voice, rest = [], []
    for c, p, x in zip(cands, prob, X):
        note = {"start": c["start"], "end": c["end"], "pitch": c["pitch"], "velocity": c["velocity"],
                "p": round(float(p), 3)}   # probabilidad de ser Voice (para el control de la app)
        if p >= thr:
            voice.append(note)
        elif c["in_harm"] and not (singer and x[i_voc] > 0):
            rest.append(note)   # acompanamiento de harm; las notas del canto no son Style
    style = [dict(b) for b in bass] + rest
    return {"voice": voice, "style": style}


# --------------------------------------------------------------------------- #
# Uso en la app: canciones largas por trozos
# --------------------------------------------------------------------------- #
CHUNK = 40.0    # se entrenó con canciones de 30-50 s; las comparaciones entre notas son n²
OVERLAP = 4.0   # contexto extra a cada lado de cada trozo


def threshold() -> float:
    """Umbral de probabilidad por defecto para decidir que una nota es Voice."""
    m = _load_model()
    return float(m["thr"]) if m else 0.5


def _write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _shift(n, dt):
    return dict(n, start=float(n["start"]) + dt, end=float(n["end"]) + dt)


def separate(data):
    """data: {"notes": {fuente: [notas]}, "feat": {..., "vocal_share"}, "stems_dir": con mix.wav e inst.wav}.

    Devuelve {"voice": [...], "style": [...]}; cada nota lleva "p" (probabilidad de ser Voice),
    salvo las del bajo.
    """
    import tempfile

    notes, feat = data["notes"], data["feat"]
    sdir = Path(data["stems_dir"])
    audio = {name: _read_wav(sdir / name) for name in ("mix.wav", "inst.wav")}
    duration = float(feat["duration"])
    n_chunks = max(1, round(duration / CHUNK))
    bounds = [duration * i / n_chunks for i in range(n_chunks + 1)]

    voice, style = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for core_a, core_b in zip(bounds, bounds[1:]):
            a, b = max(0.0, core_a - OVERLAP), min(duration, core_b + OVERLAP)
            for name, x in audio.items():
                if x is not None:
                    _write_wav(Path(tmp) / name, x[int(a * SR):int(b * SR)])
            chunk = {
                "notes": {k: [_shift(n, -a) for n in v if a <= n["start"] < b] for k, v in notes.items()},
                "feat": {"duration": b - a, "vocal_share": feat.get("vocal_share", 0.0),
                         "beats": [t - a for t in feat.get("beats", []) if a <= t < b]},
                "stems_dir": Path(tmp),
            }
            out = _separate_core(chunk)
            for kind, dst in (("voice", voice), ("style", style)):
                dst.extend(_shift(n, a) for n in out[kind] if core_a <= n["start"] + a < core_b)
    return {"voice": voice, "style": style}
