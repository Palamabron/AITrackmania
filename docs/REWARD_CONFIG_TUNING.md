# Dobór parametrów REWARD_CONFIG (SOTA)

## Założenia
- **Suma nagród max ~750** (bez gigantycznych zwrotów).
- **Szybsze przejazdy promowane** przez istniejący bonus za czas (TIME_BONUS_SCALE), bez dodawania nowych nagród ani kar.
- Tylko **konfiguracja** (bez zmian w kodzie).

---

## Parametry w docs/config_sophy_run_o.json

| Parametr | Wartość | Rola |
|----------|---------|------|
| **REWARD_SCALE** | 0.46 | Skala całej nagrody; łącznie z PROJECTED_VELOCITY trzyma sumę w okolicach 650–720. |
| **PROJECTED_VELOCITY_SCALE** | 0.5 | Waga prędkości w kierunku trasy; mniejsza niż 1.0 ogranicza sumę. |
| **TIME_BONUS_SCALE** | 95000 | Bonus na mecie = 95000/kroki × REWARD_SCALE; mniej kroków (szybszy przejazd) = wyższy bonus, suma nadal <750. |
| **STEERING_DELTA_PENALTY** | 0.08 | Lekka kara za szarpanie kierownicą. |

Przy typowym przejeździe ~730–750 kroków: baza (progress + projected velocity) ~600–650, bonus za czas ~58–62, **suma ~660–710 (max <750)**. Szybszy przejazd (mniej kroków) dostaje wyższy bonus, więc łącznie wygrywa.

---

## Wklejka REWARD_CONFIG (spójna z config_sophy_run_o.json)

```json
"REWARD_CONFIG": {
  "CONSTANT_PENALTY": 0,
  "CHECK_FORWARD": 500,
  "CHECK_BACKWARD": 10,
  "FAILURE_COUNTDOWN": 9,
  "MIN_STEPS": 70,
  "MAX_STRAY": 50,
  "SPEED_SAFE_DEVIATION_RATIO": 0.15,
  "WALL_HUG_SPEED_THRESHOLD": 10,
  "WALL_HUG_PENALTY_FACTOR": 0.005,
  "REWARD_SCALE": 0.46,
  "SPEED_TERMINAL_SCALE": 0,
  "PROJECTED_VELOCITY_SCALE": 0.5,
  "STEERING_DELTA_PENALTY": 0.08,
  "MAX_TRACK_WIDTH": 50,
  "BOUNDARY_PENALTY_WEIGHT": 2.0,
  "BOUNDARY_CRASH_PENALTY": 10.0,
  "REWARD_CLIP_FLOOR": 10.0,
  "TIME_BONUS_SCALE": 95000,
  "CONDITIONAL_PENALTY_WHEN_BRAKING": false,
  "BRAKE_THRESHOLD": 0.3,
  "TRACK_LOOK_AHEAD_PCT": 5.0,
  "TRACK_POINT_SPACING_M": 2.5
}
```
