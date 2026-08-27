# TrackmaniaRL diagrams

Each diagram is stored as one reproducible set:

- `.spec.json` — compact source for deterministic regeneration;
- `.excalidraw` — canonical editable source that can be imported into Excalidraw;
- `-preview.png` — GitHub/PyPI-compatible raster preview;
- `-preview.svg` — scalable preview embedded in the detailed guides;
- `-preview.html` — local preview with a download of the editable scene;

The spec stores the semantic colors, zones, nodes, routed edges and notes used
by the repository renderer. After editing a spec, deterministically regenerate
the editable scene, SVG and HTML preview with:

```bash
uv run python -m docs.diagrams.render
```

The renderer intentionally does not depend on a platform-specific SVG
rasterizer. Regenerate a PNG from the resulting SVG at the spec's exact canvas
size whenever the raster preview is published in the root README or release
archive. Manual Excalidraw adjustments must be reflected back in the spec.
Validate every scene and visually inspect both SVG and PNG at normal
documentation width before committing all rendered forms.

## Diagram set

| Subject | Editable source | Preview |
| --- | --- | --- |
| Runtime architecture | [runtime-architecture.excalidraw](runtime-architecture.excalidraw) | [PNG](runtime-architecture-preview.png) · [SVG](runtime-architecture-preview.svg) · [HTML](runtime-architecture-preview.html) |
| Checkpoint and resume | [checkpoint-resume.excalidraw](checkpoint-resume.excalidraw) | [PNG](checkpoint-resume-preview.png) · [SVG](checkpoint-resume-preview.svg) · [HTML](checkpoint-resume-preview.html) |
| Model composition and unified learner | [model-composition.excalidraw](model-composition.excalidraw) | [PNG](model-composition-preview.png) · [SVG](model-composition-preview.svg) · [HTML](model-composition-preview.html) |
| Imitation learning and RL handoff | [imitation-learning.excalidraw](imitation-learning.excalidraw) | [PNG](imitation-learning-preview.png) · [SVG](imitation-learning-preview.svg) · [HTML](imitation-learning-preview.html) |
| Demonstration timing and action labels | [demonstration-timing.excalidraw](demonstration-timing.excalidraw) | [PNG](demonstration-timing-preview.png) · [SVG](demonstration-timing-preview.svg) · [HTML](demonstration-timing-preview.html) |
| Reward decomposition and terminal PBRS | [reward-decomposition.excalidraw](reward-decomposition.excalidraw) | [PNG](reward-decomposition-preview.png) · [SVG](reward-decomposition-preview.svg) · [HTML](reward-decomposition-preview.html) |
| Recurrent replay, n-step targets and PER | [replay-sequence.excalidraw](replay-sequence.excalidraw) | [PNG](replay-sequence-preview.png) · [SVG](replay-sequence-preview.svg) · [HTML](replay-sequence-preview.html) |
| Local and remote deployment | [distributed-security.excalidraw](distributed-security.excalidraw) | [PNG](distributed-security-preview.png) · [SVG](distributed-security-preview.svg) · [HTML](distributed-security-preview.html) |
| Trackmania and Openplanet integration | [trackmania-integration.excalidraw](trackmania-integration.excalidraw) | [PNG](trackmania-integration-preview.png) · [SVG](trackmania-integration-preview.svg) · [HTML](trackmania-integration-preview.html) |

The committed HTML previews are local-only and expose a download of the
editable scene. They do not upload repository diagrams or their contents.
