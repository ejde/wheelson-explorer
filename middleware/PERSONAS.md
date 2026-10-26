# Custom personas

Built-in personas (`benson`, `sir_david`, `klaus`, `zog7`) live in
`config.py`. You do not need to edit Python to add your own: define them
in a YAML file and the middleware merges them in.

## Quickstart

```sh
cp middleware/personas.example.yaml middleware/personas.yaml
# edit personas.yaml: rename the id, rewrite the voice, tune the movement
```

Then either restart the middleware or reload live:

```sh
curl -X POST http://localhost:8000/personas/reload
curl http://localhost:8000/personas   # list everything, built-in + custom
curl -X POST http://localhost:8000/persona -H 'Content-Type: application/json' \
  -d '{"persona": "finch"}'
```

`personas.yaml` is gitignored: your personas are yours and never get
committed. `personas.example.yaml` is the tracked template.

## Schema

One mapping key per persona. The id must be lowercase letters, digits,
or underscore. Unknown fields are rejected, so a typo fails loudly
instead of being silently ignored.

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Display name |
| `system_prompt` | yes | The voice: who this Muse is, in 2-4 sentences |
| `base_speed` | yes | `slow`, `medium`, or `fast` |
| `safety_weight` | yes | Number >= 0; higher = more cautious |
| `curiosity_weight` | yes | Number >= 0; higher = chases novelty |
| `perimeter_weight` | yes | Number >= 0; higher = hugs walls |
| `preferred_wall` | yes | `none`, `left`, or `right` |
| `forward_burst_ms` | yes | Positive integer, milliseconds of one forward burst |
| `hold_bias` | yes | 0.0-1.0; higher = pauses more often |
| `emoji` | no | Display emoji (default: robot) |
| `color` | no | Dashboard accent as `#rrggbb` |
| `move_bias` | no | One-line movement signature for the dashboard |
| `salience` | no | What the VLM should prioritize for this persona |
| `commentary_style` | no | Tone label for VLM commentary |
| `turn_duration_ms` | no | Positive integer, milliseconds of one turn (default 600) |

Reference points from the built-ins: benson is slow, safety 1.7,
hold 0.8; sir_david is curiosity 1.6; zog7 is fast, perimeter 1.7,
hold 0.1. Start from the persona closest to what you want and tune one
knob at a time.

## Rules

- A custom persona may also override a built-in key (e.g. your own
  slower `zog7`). The override is logged, and removing it from the file
  restores the built-in on the next reload.
- Reload refuses to remove the currently active persona (HTTP 409).
  Switch to another persona first, then reload.
- A malformed file never wipes the running set: the reload fails with
  HTTP 400 and the previous personas stay live.
- Switching works exactly like built-ins (`POST /persona`), including
  the validation error that lists valid keys.
- The dashboard's persona buttons still list built-ins only; use the
  API for custom personas.

## Notes

- A middleware persona is movement + voice presets. It is not the
  firmware face: the on-screen face (generic vs the Finch owl) is a
  separate firmware build option (`-DFACE_FINCH`).
- The `finch` persona in `personas.example.yaml` is the test-bed
  example: a curious, companionable Muse. Copy it, rename it, and make
  it yours.
