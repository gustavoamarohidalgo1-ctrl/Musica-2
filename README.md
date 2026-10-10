# AppMusica — Audio a MIDI

Sube una canción y la app detecta **qué teclas se están tocando**: las muestra en un
piano roll, las ilumina en un teclado mientras suena y te deja descargar el **MIDI**.

Todo corre **en tu Mac, gratis y sin internet** (después de instalar).

## Cómo abrirla

Doble clic en **`iniciar.command`**. Se abre el navegador en http://127.0.0.1:8765.
Para cerrarla, cierra la ventana de Terminal que se abrió.

> La primera vez macOS puede decir que no puede verificar el archivo:
> clic derecho → **Abrir** → **Abrir**.

## Modos

| Modo | Qué hace | Úsalo para |
|---|---|---|
| **Teclado con Style** | Separa lo que toca la mano derecha (**Voice**) del acompañamiento automático (**Style**: bajo, acordes y pads, batería). Si alguien canta, el **canto** sale aparte | Grabaciones y videos de teclados arranger (Yamaha PSR, etc.) |
| **Canción completa** | Demucs separa piano, guitarra, bajo, voz, batería y otros; luego saca las notas de cada uno | Canciones con banda |
| **Solo piano** | Transcribe directo con el modelo de piano de ByteDance | Grabaciones de piano solo (lo más preciso) |
| **Un solo instrumento** | Transcribe directo con Basic Pitch | Guitarra, voz, flauta, violín… |

### Cómo funciona «Teclado con Style»

En el audio, la Voice y el Style llegan mezclados, así que la separación es una
estimación, no algo exacto. Pistas que usa:
- La batería y el bajo del Style los separa Demucs.
- La Voice suele ir por encima del acompañamiento, sonar más fuerte y tener las
  pequeñas variaciones de tiempo de una persona; el Style se repite y va exacto al pulso.
- Si la pista de voz de Demucs lleva mucha energía, hay alguien cantando: el canto se
  muestra aparte y se quita antes de buscar la Voice.

**Harmony y Dual:** con Harmony (Duet, Trio, Block, Octave, Echo, Tremolo) o Dual activados,
el teclado suena notas que no tocaste. **Solo teclas presionadas** (activado por defecto)
deja en la Voice solo la tecla que presionas: de las notas que empiezan juntas queda la de
arriba, se quita la duplicación a la octava del Dual y se juntan las repeticiones del
Echo / Tremolo. Las notas que añadió el teclado pasan a **Harmony y Dual (automático)**,
oculta por defecto (ver `app/static/teclas.js`). Con un Dual a la octava no se puede saber desde
el audio cuál capa es la tecla: queda la de arriba, y **Octava de la Voice ▼ ▲** la corrige.

Acierto medido en canciones de prueba con Harmony (teclas de la mano derecha, F1): Duet 0,86,
Trio 0,80, Tremolo 0,74, Echo 0,69, Block 0,60, Octave 0,55; sin Harmony 0,80–0,84.

Con **Ver: Todo / Solo Voice / Solo Style / Solo voz humana** eliges qué notas mirar, y con
**Sonido MIDI** escuchas solo esas notas. El algoritmo se eligió midiendo varias opciones
con canciones de prueba de las que se conoce cada nota (ver `app/voice_style.py`).

## Recortar antes de convertir

Al elegir un archivo (de hasta 150 MB) aparece su forma de onda. Arrastra sobre ella para
marcar solo la parte que quieres (o escribe **Desde** / **Hasta**, por ejemplo `1:05` y
`1:45`) y pulsa **▶ Escuchar** para comprobarla. En archivos más grandes, o en formatos
que el navegador no puede leer, escribe los tiempos a mano. Solo se procesa ese tramo, así que es mucho más rápido;
las notas y el MIDI empiezan en 0:00 al inicio del recorte. Sin recorte se procesa la
canción completa.

## Qué puedes hacer con el resultado

- **Piano roll:** cada barra es una nota. Clic para saltar a ese momento; rueda o arrastre para moverte; Ctrl + rueda para hacer zoom.
- **Teclado:** se iluminan las teclas presionadas, con el color de cada instrumento.
- **Secuencia de notas:** lista con tiempo, nota (C4 / Do4), duración y fuerza. Clic en una fila para saltar.
- **Escuchar:** la canción original o solo un instrumento separado. Sube **Sonido MIDI** para oír las notas detectadas y comparar.
- **Ocultar notas débiles:** quita notas falsas de poca fuerza. También se aplica al MIDI que descargas.
- **Descargar MIDI:** todo junto (una pista por instrumento) o cada instrumento por separado.
- **Barra espaciadora:** reproducir / pausar.

## Modelos de IA (todos gratuitos y de código abierto)

- **Demucs `htdemucs_6s`** (Meta): separación de instrumentos. Usa la GPU del Mac.
- **Basic Pitch** (Spotify): transcripción de cualquier instrumento.
- **piano_transcription** (ByteDance): transcripción de piano de alta precisión.

## Archivos

```
app/pipeline.py      separación + transcripción
app/server.py        servidor local (FastAPI)
app/static/          interfaz web
models/              modelo de piano de ByteDance
data/jobs/           tus canciones procesadas (audios separados, MIDI, notas)
instalar.sh          reinstala todo desde cero
iniciar.command      abre la app
```

## Limitaciones

- No es 100% perfecto: en partes muy cargadas aparecen notas falsas o faltan algunas.
- Piano y guitarra se separan peor que voz, bajo y batería.
- La batería se separa, pero no se convierte en notas.
