# TrackmaniaRL diagrams

Each diagram is stored in three forms:

- `.excalidraw` — canonical editable source that can be imported into Excalidraw;
- `-preview.svg` — lightweight preview embedded in Markdown documentation;
- `-preview.html` — local preview with download and optional Open in Excalidraw
  controls;

Edit the `.excalidraw` file first. When the scene changes, export its matching
SVG and replace the preview, then regenerate the local HTML preview with the
Excalidraw preview tool. This avoids losing manual layout adjustments.

## Diagram set

| Subject | Editable source | Preview |
| --- | --- | --- |
| Runtime architecture | [runtime-architecture.excalidraw](runtime-architecture.excalidraw) | [SVG](runtime-architecture-preview.svg) · [HTML](runtime-architecture-preview.html) |
| Extension workflow | [extension-workflow.excalidraw](extension-workflow.excalidraw) | [SVG](extension-workflow-preview.svg) · [HTML](extension-workflow-preview.html) |
| Distributed security and durability | [distributed-security.excalidraw](distributed-security.excalidraw) | [SVG](distributed-security-preview.svg) · [HTML](distributed-security-preview.html) |

The committed HTML previews are local-only. Clicking their Open in Excalidraw
button performs an upload from the browser; merely opening the preview does not
send the scene anywhere.
