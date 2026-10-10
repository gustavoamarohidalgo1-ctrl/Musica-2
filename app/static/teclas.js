"use strict";
// Reduce las notas de la Voice a las teclas que se presionaron de verdad.
//
// Con Harmony o Dual activados, el teclado suena más notas de las que toca la mano derecha:
//  - Harmony Duet / Trio / Block / Octave: notas por debajo de la melodía que empiezan a la vez.
//  - Dual con desplazamiento de octava: la misma nota una octava arriba o abajo.
//  - Harmony Echo / Tremolo: la nota se repite sola mientras se mantiene la tecla.
// pressedKeys(notas) -> { keys: [...], auto: [...] }   (auto = notas que añadió el teclado)
(function (root) {
  const ONSET = 0.06;        // notas que empiezan con menos de 60 ms de diferencia = un mismo golpe
  const TREMOLO_IOI = 0.16;  // repeticiones de la misma nota más rápidas que esto = Tremolo
  const REPEAT_MAX = 0.6;    // el Echo / Tremolo más lento que se considera (s entre repeticiones)

  function byPitch(notes) {
    const m = new Map();
    for (const n of notes) {
      if (!m.has(n.pitch)) m.set(n.pitch, []);
      m.get(n.pitch).push(n);
    }
    for (const seq of m.values()) seq.sort((a, b) => a.start - b.start);
    return [...m.values()];
  }

  // ¿Echo / Tremolo activo? Si muchas notas repiten la anterior de su misma altura a un ritmo
  // fijo, ese ritmo es el del efecto (una melodía tocada a mano no repite tanto ni tan parejo).
  function repeatRate(seqs, total) {
    const iois = [];
    for (const seq of seqs) {
      for (let k = 1; k < seq.length; k++) {
        const d = seq[k].start - seq[k - 1].start;
        if (d <= REPEAT_MAX) iois.push(d);
      }
    }
    let best = null, count = 0;
    for (const c of iois) {
      const n = iois.filter((d) => Math.abs(d - c) <= 0.12 * c + 0.01).length;
      if (n > count) { best = c; count = n; }
    }
    return best !== null && count >= 8 && count >= 0.3 * total ? best : null;
  }

  // Echo / Tremolo: la misma altura repetida sola mientras se mantiene la tecla.
  function repeats(notes) {
    const auto = new Set();
    const held = new Map();  // primera nota de la cadena -> fin de la última repetición
    const seqs = byPitch(notes);
    const rate = repeatRate(seqs, notes.length);
    // Una repetición sigue la cadena si llega a 1 o 2 pasos del ritmo (Basic Pitch a veces pierde una).
    const step = (d) => rate !== null && [1, 2].some((k) => Math.abs(d - k * rate) <= 0.15 * rate + 0.015);
    for (const seq of seqs) {
      let i = 0;
      while (i < seq.length) {
        let j = i + 1;
        const ioi = j < seq.length ? seq[j].start - seq[i].start : Infinity;
        const fast = (d) => ioi <= TREMOLO_IOI && Math.abs(d - ioi) <= 0.25 * ioi + 0.02;
        while (j < seq.length && (step(seq[j].start - seq[j - 1].start) || fast(seq[j].start - seq[j - 1].start))) j++;
        const chain = seq.slice(i, j);
        if (chain.length >= (rate !== null ? 2 : 3)) {
          chain.slice(1).forEach((n) => auto.add(n));
          held.set(chain[0], Math.max(...chain.map((n) => n.end)));
          i = j;
        } else {
          i += 1;
        }
      }
    }
    return { auto, held };
  }

  function pressedKeys(notes) {
    const all = [...notes].sort((a, b) => a.start - b.start || b.pitch - a.pitch);
    const { auto, held } = repeats(all);
    const rest = all.filter((n) => !auto.has(n));

    // Golpes: notas que empiezan juntas
    const groups = [];
    for (const n of rest) {
      const g = groups[groups.length - 1];
      if (g && n.start - g[0].start <= ONSET) g.push(n);
      else groups.push([n]);
    }

    // De cada golpe queda la nota de arriba. Con un Dual a la octava no se puede saber desde el
    // audio cuál de las dos capas es la tecla (suenan juntas); se toma la de arriba y el control
    // «Octava» de la app baja o sube toda la Voice si el Dual iba una octava arriba.
    const keys = [];
    for (const g of groups) {
      const top = Math.max(...g.map((n) => n.pitch));
      const key = g.find((n) => n.pitch === top);
      keys.push(key);
      for (const n of g) if (n !== key) auto.add(n);
    }

    // Notas que empiezan mientras suena una tecla más aguda y terminan dentro de ella:
    // armonía que el teclado vuelve a tocar, no una tecla nueva.
    const final = [];
    for (const n of keys) {
      const inside = keys.some((k) => k !== n && k.pitch > n.pitch && k.start < n.start - ONSET
        && k.end >= n.end - 0.05);
      if (inside) auto.add(n);
      else final.push(n);
    }
    const order = (a, b) => a.start - b.start || a.pitch - b.pitch;
    // La tecla del Echo / Tremolo dura todo lo que se mantuvo (copia: no se tocan las originales).
    const out = final.map((n) => (held.has(n) ? { ...n, end: held.get(n) } : n));
    return { keys: out.sort(order), auto: [...auto].sort(order) };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { pressedKeys };
  else root.pressedKeys = pressedKeys;
})(this);
