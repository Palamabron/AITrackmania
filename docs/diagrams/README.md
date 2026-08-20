# TrackmaniaRL diagrams

Each diagram is stored as one reproducible set:

- `.spec.json` — compact source for deterministic regeneration;
- `.excalidraw` — canonical editable source that can be imported into Excalidraw;
- `-preview.png` — GitHub-compatible preview embedded in Markdown documentation;
- `-preview.svg` — scalable preview for local use;
- `-preview.html` — local preview with download and optional Open in Excalidraw
  controls;

Edit the spec and regenerate the scene for structural changes. Manual
Excalidraw adjustments are allowed, but must be reflected back in the spec or
documented as intentional. Validate every scene and visually inspect the PNG at
normal documentation width before committing all four rendered forms.

## Diagram set

| Subject | Editable source | Preview |
| --- | --- | --- |
| Runtime architecture | [runtime-architecture.excalidraw](runtime-architecture.excalidraw) | [PNG](runtime-architecture-preview.png) · [SVG](runtime-architecture-preview.svg) · [HTML](runtime-architecture-preview.html) |
| Model composition and unified learner | [model-composition.excalidraw](model-composition.excalidraw) | [PNG](model-composition-preview.png) · [SVG](model-composition-preview.svg) · [HTML](model-composition-preview.html) |
| Imitation learning and RL handoff | [imitation-learning.excalidraw](imitation-learning.excalidraw) | [PNG](imitation-learning-preview.png) · [SVG](imitation-learning-preview.svg) · [HTML](imitation-learning-preview.html) |
| Extension workflow | [extension-workflow.excalidraw](extension-workflow.excalidraw) | [PNG](extension-workflow-preview.png) · [SVG](extension-workflow-preview.svg) · [HTML](extension-workflow-preview.html) |
| Distributed security and durability | [distributed-security.excalidraw](distributed-security.excalidraw) | [PNG](distributed-security-preview.png) · [SVG](distributed-security-preview.svg) · [HTML](distributed-security-preview.html) |

The committed HTML previews are local-only. Clicking their Open in Excalidraw
button performs an upload from the browser; merely opening the preview does not
send the scene anywhere.
