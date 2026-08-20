# TrackmaniaRL diagrams

Each diagram is stored as one reproducible set:

- `.spec.json` — compact source for deterministic regeneration;
- `.excalidraw` — canonical editable source that can be imported into Excalidraw;
- `-preview.png` — GitHub-compatible preview embedded in Markdown documentation;
- `-preview.svg` — scalable preview for local use;
- `-preview.html` — local preview with a download of the editable scene;

The spec stores the semantic colors, zones, nodes, routed edges and notes used
by the repository renderer. After editing a spec, deterministically regenerate
the editable scene, SVG and HTML preview with:

```bash
uv run python docs/diagrams/render.py
```

The renderer intentionally does not depend on a platform-specific SVG rasterizer.
When a diagram embedded as PNG in the root README changes, export its generated
SVG to the same `1600`-pixel canvas as `-preview.png`. Manual Excalidraw
adjustments must be reflected back in the spec. Validate every scene and
visually inspect both SVG and PNG at normal documentation width before
committing all rendered forms.

## Diagram set

| Subject | Editable source | Preview |
| --- | --- | --- |
| Runtime architecture | [runtime-architecture.excalidraw](runtime-architecture.excalidraw) | [PNG](runtime-architecture-preview.png) · [SVG](runtime-architecture-preview.svg) · [HTML](runtime-architecture-preview.html) |
| Model composition and unified learner | [model-composition.excalidraw](model-composition.excalidraw) | [PNG](model-composition-preview.png) · [SVG](model-composition-preview.svg) · [HTML](model-composition-preview.html) |
| Imitation learning and RL handoff | [imitation-learning.excalidraw](imitation-learning.excalidraw) | [PNG](imitation-learning-preview.png) · [SVG](imitation-learning-preview.svg) · [HTML](imitation-learning-preview.html) |
| Extension workflow | [extension-workflow.excalidraw](extension-workflow.excalidraw) | [PNG](extension-workflow-preview.png) · [SVG](extension-workflow-preview.svg) · [HTML](extension-workflow-preview.html) |
| Distributed security and durability | [distributed-security.excalidraw](distributed-security.excalidraw) | [PNG](distributed-security-preview.png) · [SVG](distributed-security-preview.svg) · [HTML](distributed-security-preview.html) |

The committed HTML previews are local-only and expose a download of the
editable scene. They do not upload repository diagrams or their contents.
