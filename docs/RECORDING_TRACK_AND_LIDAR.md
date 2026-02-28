# Recording points on the map (reward & track) — where and how

## Two ways to get “points in space”

| Source | What it is | Sees after the corner? | Best for |
|--------|-------------|------------------------|----------|
| **Screenshot “LIDAR”** | 19 beams from the **current frame image** (road vs non-road in view). | **No** — only what’s visible on screen. | Fast, no recording; view-dependent. |
| **Pre-recorded track left/right** | **World-space 3D polylines** of the track boundaries (recorded once per map). | **Yes** — full track geometry ahead. | Knowing the track layout ahead (including after corners). |

**Recommendation:** If you care about “points in space” and what’s **after the corner**, **pre-recorded track left/right is better.** Screenshot LIDAR cannot see beyond the current camera view.

---

## Screenshot “LIDAR” (no recording)

**LIDAR in this codebase is not from pre-recorded map points.** It is computed in real time from the **game window screenshot**:

- **Where:** `tmrl/custom/tm/utils/tools.py` — class `Lidar`, method `lidar_20(img)`.
- **How:** 19 beams are derived from the current frame image (road vs non-road). View-dependent; **cannot see around corners**.
- **Recording:** None. Set `ENV.RTGYM_INTERFACE` to e.g. `LIDARPROGRESS` or `LIDARPROGRESSIMAGES` to use it.

---

## Reward trajectory (progress / checkpoints)

Used for **progress-based reward** (e.g. LIDARPROGRESS, LIDARPROGRESSIMAGES): the agent is rewarded for advancing along a reference trajectory.

- **Script:** `tmrl/tools/record_reward.py`
- **How:** Drive the track in the game; press **E** to start recording, **Q** to stop (or finish the track). The script samples positions from the OpenPlanet client, subsamples to a spaced trajectory, and saves it.
- **Where to put the file:**  
  **`TmrlData/reward/reward_<MAP_NAME>.pkl`**  
  (`MAP_NAME` comes from `ENV.MAP_NAME` in your config, e.g. `tmrl-test` → `reward_tmrl-test.pkl`).
- **Config:** The path is set in `tmrl/config/config_constants.py` as `REWARD_PATH` (from `REWARD_FOLDER` and `MAP_NAME`). Ensure `TmrlData/reward/` exists and that the filename matches your `MAP_NAME`.

### Denser trajectory (e.g. for a difficult turn)

To get a **finer progress signal** on the track (e.g. more points through a turn), you can interpolate the existing reward trajectory to add more points: run `scripts/interpolate_reward_trajectory.py` (see script help). It loads a `reward_<MAP_NAME>.pkl`, interpolates along the polyline (e.g. 10× more points), checks that the reward scale stays consistent, and overwrites or saves the pkl. No re-recording in-game needed.

### Centerline from track left + right (SOTA: środek trasy)

Zamiast nagrywać przejazd dla reward trajectory możesz **wygenerować oś referencyjną** ze środka toru: `(track_left + track_right) / 2`. To podejście (Gran Turismo Sophy, F1Tenth) daje gładką oś, symetryczne granice i pozwala agentowi samodzielnie odkryć optymalną linię. Granice toru są wtedy definiowane przez left/right (dotknięcie „funkcji” tych punktów = ocieranie o barierki); **MAX_TRACK_WIDTH** można wtedy zastąpić sprawdzaniem „poza pasem” (odległość do left/right).

- **Skrypt:** `scripts/build_centerline_reward.py`
- **Wejście:** `track_<MAP_NAME>_left.pkl`, `track_<MAP_NAME>_right.pkl` (lub ścieżki `--left` / `--right`). Obsługiwane też pliki `.csv` (N,2) x,z.
- **Wyjście:** `reward_<MAP_NAME>.pkl` (lub `--out`). Opcja `--smooth` wygładza centerline splajnami (stabilna styczna → stabilny cos(θ_error) w nagrodzie projected velocity).
- **Użycie:**  
  `python scripts/build_centerline_reward.py` (ścieżki z TmrlData/config)  
  lub  
  `python scripts/build_centerline_reward.py --left path/left.pkl --right path/right.pkl --out path/reward_map.pkl --smooth`

---

## Track boundaries (left / right) — “points in space” including after corners

Pre-recorded **left and right** track boundaries are **world-space 3D polylines**. They describe the full track layout, so the agent can know what’s ahead **including after the corner**.

- **Script:** `tmrl/tools/record_track.py`
- **How:** Drive the track; press **E** to start recording, **Q** to stop (or finish). Run **twice**: once for the left boundary (default), once for the right (see script options).
- **Where to put the files:**  
  - **`TmrlData/track/track_<MAP_NAME>_left.pkl`**  
  - **`TmrlData/track/track_<MAP_NAME>_right.pkl`**  
  `MAP_NAME` = `ENV.MAP_NAME` in config.
- **Who uses them:**  
  - **Reward function**: loads from `TRACK_PATH_LEFT` / `TRACK_PATH_RIGHT` for distance/geometry.  
  - **TRACKMAP** interface: track only (no images).  
  - **TRACKMAPIMAGES** interface: **images + track left/right** (see below).

---

## Using images + track left/right (TRACKMAPIMAGES)

To use **camera images and pre-recorded track boundaries** together (points in space ahead, including after corners):

### 1. Where to put your track files

You already recorded left/right. Place the files here (paths are under `TmrlData`, usually `~/TmrlData`):

| File | Path |
|------|------|
| Left boundary  | **`TmrlData/track/track_<MAP_NAME>_left.pkl`**  |
| Right boundary | **`TmrlData/track/track_<MAP_NAME>_right.pkl`** |

Replace `<MAP_NAME>` with the **exact** map name you use in config (e.g. `tmrl-test` → `track_tmrl-test_left.pkl`).

### 2. Reward trajectory (required for progress reward)

The interface also needs a **reward trajectory** for progress-based reward. If you don’t have it yet:

- Run **`tmrl/tools/record_reward.py`** (E to start, Q to stop), then put the file at:  
  **`TmrlData/reward/reward_<MAP_NAME>.pkl`**

### 3. Config

In your `config.json` (e.g. `TmrlData/config/config.json`):

- **`ENV.RTGYM_INTERFACE`**: set to **`"TRACKMAPIMAGES"`**
- **`ENV.MAP_NAME`**: same name as in the filenames above (e.g. `"tmrl-test"`)
- **`ALG.ALGORITHM`**: **`"SAC"`** (required for this interface)
- Optionally: **`ENV.IMG_GRAYSCALE": true`**, **`ENV.IMG_WIDTH`** / **`ENV.IMG_HEIGHT`** (e.g. 64)

The observation is then **(speed, progress, track_information, images)**. Track left/right are loaded from the `.pkl` paths above; no CSV or extra conversion needed.

---

## Summary

| What              | Record? | Script                | Output path (under `TmrlData`)        | Sees after corner? |
|-------------------|--------|------------------------|---------------------------------------|--------------------|
| **Screenshot LIDAR** | No   | —                      | Computed from image each step        | No                 |
| **Track left/right** | Yes  | `tmrl/tools/record_track.py`   | `track/track_<MAP_NAME>_left.pkl`, `_right.pkl` | **Yes** (world-space geometry) |
| **Reward trajectory** | Yes | `tmrl/tools/record_reward.py`  | `reward/reward_<MAP_NAME>.pkl`         | — (used for progress reward) |

- For **images + points in space (including after the corner)** → use **TRACKMAPIMAGES**: put track files in `TmrlData/track/` as above, set `ENV.RTGYM_INTERFACE` to `"TRACKMAPIMAGES"` and `ALG.ALGORITHM` to `"SAC"`.
- For **track only** (no images) → **TRACKMAP** interface.
- Use the same `MAP_NAME` in `ENV.MAP_NAME` as in the filenames so the loader finds the correct reward and track files.
