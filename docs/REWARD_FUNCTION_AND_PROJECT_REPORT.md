# Raport: funkcja nagrody (reward function) i tło projektu AITrackmania / TMRL

Dokument opisuje **obecną implementację funkcji nagrody** (SOTA) w projekcie, **kontekst projektu** oraz **znany problem**: najszybsze przejazdy powinny mieć wyższą sumę nagród niż wolniejsze, a w praktyce nie zawsze tak jest.

---

## 1. Tło projektu

### 1.1 Czym jest TMRL / AITrackmania

**TMRL** (TrackMania Reinforcement Learning) to framework do uczenia ze wzmocnieniem (RL) w czasie rzeczywistym. W tym repozytorium używany jest do trenowania agentów w **TrackMania 2020**.

- **Środowisko:** Gymnasium + rtgym (krok co ~0,05 s, 20 Hz).
- **Sterowanie:** wirtualny gamepad (analogowy gaz / hamulec / kierownica).
- **Dane z gry:** plugin OpenPlanet (np. TQC_GrabData) wysyła do Pythona pozycję, prędkość, heading (aim_yaw), ster, checkpointy, stan toru itd.

### 1.2 Algorytmy i pipeline

| Element | Opis |
|--------|------|
| **Algorytm** | **TQC** (Truncated Quantile Critic) – actor-critic z kwantylowym krytykiem i odrzucaniem górnych kwantyli. |
| **Model** | **Sophy residual** – sieć na wejściu API (m.in. 20 floatów z pluginu) oraz opcjonalnie „track look-ahead”. |
| **Interfejs** | **TQCGRAB** – `TM2020InterfaceTQC`: obserwacje z API; reward z `RewardFunction` w `tmrl/custom/tm/utils/compute_reward.py`. |

Cel: polityka ma **pokonywać tor w jak najkrótszym czasie**, przy sensownej trajektorii (bez wypadania z toru, bez niepożądanych skrótów).

---

## 2. Obecna funkcja nagrody – logika (SOTA)

Klasa **`RewardFunction`** w `tmrl/custom/tm/utils/compute_reward.py` liczy nagrodę **w każdym kroku** na podstawie pozycji, trajektorii referencyjnej, prędkości, kąta ustawienia (aim_yaw), steru oraz parametrów z **`REWARD_CONFIG`**. **Nie ma już** post-epizodowego bonusu K/T² w buforze – sygnał „szybkość” jest wbudowany w nagrodę krokową.

Składniki (w kolejności logicznej):

---

### 2.1 Postęp wzdłuż trajektorii (progress)

- **Trajektoria:** ładowana z pliku (np. `TmrlData/reward/reward_<MAP_NAME>.pkl`) – jedna demonstracja trasy (pokrywa tor, nie musi być optymalna czasowo).
- **Postęp:** w każdym kroku wyznaczany jest najbliższy punkt na trajektorii (z ograniczeniem „do przodu” przez `nb_obs_forward`). Postęp = przyrost **dystansu wzdłuż łuku** (cumulative arc length).
- **Nagroda (raw):** `distance_gained * (100.0 / _total_traj_length)` – **pełne okrążenie** daje łącznie **100** w jednostkach raw (przed skalowaniem). Przy braku postępu następuje „rewind” wstecz (bardziej Markowowska nagroda).

Jeśli samochód jest bardzo daleko od trajektorii (`min_dist > 2 * max_dist_from_traj`), nagroda za postęp w tym kroku = 0.

---

### 2.2 Projected velocity (gęsta nagroda za prędkość)

**Wzór (per krok):**  
`reward += (v_kmh/3.6) * cos(θ_error) * dt * PROJECTED_VELOCITY_SCALE`  
gdzie:
- `v_kmh` – prędkość w km/h,
- `θ_error` = aim_yaw − kąt stycznej do trajektorii w bieżącym indeksie (precomputowane tangenty),
- `dt` = 0,05 s.

- **Efekt:** nagroda rośnie, gdy jedziesz **szybko i w kierunku trasy** (cos ≈ 1). Cofanie (θ > π/2) i jazda prostopadle do trasy (np. w ścianę) dają cos ≈ 0 lub ujemne – brak lub kara bez dodatkowych warunków.
- **Konfiguracja:** `REWARD_CONFIG.PROJECTED_VELOCITY_SCALE` (np. 0,5). Sygnał jest **gęsty** (co krok), w przeciwieństwie do dawnego bonusu K/T² po epizodzie.

---

### 2.3 Kary i warunki zakończenia (termination)

| Składnik | Opis |
|----------|------|
| **constant_penalty** | Stała kara co krok (np. 0); opcjonalnie wyłączana przy hamowaniu. |
| **Wall-hug** | Ruch bez postępu wzdłuż trajektorii i daleko od toru – kara narastająca; po wielu krokach możliwe `terminated` (wall_hug_no_progress). |
| **Low speed / stall** | Długotrwała bardzo niska prędkość (< 5 km/h) → failure_counter; po przekroczeniu progu → terminated. |
| **Crash** | Jeśli `crashed` → odejmowane `crash_penalty`. |
| **Boundary (SOTA)** | Gdy `min_dist > _deviation_threshold`: **kara kwadratowa** od odległości od osi; gdy `min_dist > MAX_TRACK_WIDTH` → twarde `terminated` + kara `BOUNDARY_CRASH_PENALTY`. Kara **nie** jest osłabiana przy wysokiej prędkości (zapobiega wall-banging / noseboost). |
| **Steering delta (jerk)** | Kara `STEERING_DELTA_PENALTY * |steer_t − steer_{t−1}|` – ogranicza szarpanie kierownicą (bang-bang). |

---

### 2.4 Bonusy (lap, checkpoint, koniec trasy)

- **Lap / checkpoint:** `LAP_REWARD`, `CHECKPOINT_REWARD` (z cooldownami).
- **Near finish:** bonus zbliżony do `END_OF_TRACK_REWARD` przy zbliżaniu do mety.
- **End of track:**  
  - dodawana **pozostała odległość** do końca trajektorii (pełna trasa ≈ 100 raw);  
  - mały bonus za ukończenie: `10.0 / _reward_scale`;  
  - **Time bonus (opcjonalny):** jeśli `TIME_BONUS_SCALE > 0`, na mecie: `reward += TIME_BONUS_SCALE / step_counter` (mniej kroków = wyższy bonus).

---

### 2.5 Skalowanie i obcięcie

- **Skala:** `reward *= _reward_scale` (np. 0,46).
- **Clip ujemnych:** `reward = max(-REWARD_CLIP_FLOOR, reward)` (np. -10). Dodatnia nagroda nie jest obcinana z góry.

**Uwaga:** Nie ma już floory „dla kroków z postępem” (max(0, reward)) – kroki z projected velocity mogą być ujemne przy złym kierunku.

---

## 3. Parametry konfiguracyjne (REWARD_CONFIG) – aktualne

| Klucz | Znaczenie |
|-------|-----------|
| **REWARD_SCALE** | Mnożnik końcowy (np. 0,46); łącznie z PROJECTED_VELOCITY trzyma sumę w ok. 550–750. |
| **PROJECTED_VELOCITY_SCALE** | Waga składnika v·cos(θ)·dt (np. 0,5). |
| **TIME_BONUS_SCALE** | Bonus na mecie = TIME_BONUS_SCALE / step_counter; 0 = wyłączony. Większa wartość silniej promuje szybsze przejazdy. |
| **STEERING_DELTA_PENALTY** | Kara za zmianę kąta kierownicy (np. 0,08). |
| **MAX_TRACK_WIDTH** | Powyżej tej odległości od osi: terminated + BOUNDARY_CRASH_PENALTY. |
| **BOUNDARY_PENALTY_WEIGHT** | Współczynnik kary kwadratowej za odchylenie od osi. |
| **BOUNDARY_CRASH_PENALTY** | Kara przy terminated „off_track”. |
| **REWARD_CLIP_FLOOR** | Dolne obcięcie nagrody krokowej (np. 10). |
| **SPEED_TERMINAL_SCALE** | Dawny bonus K/T² – **domyślnie 0** (wyłączony); bonus za czas jest w TIME_BONUS_SCALE i projected velocity. |
| **TRACK_LOOK_AHEAD_PCT**, **TRACK_POINT_SPACING_M** | Track look-ahead (punkty przed samochodem) – do obserwacji, nie do samej nagrody. |

Pozostałe (failure, crash, lap, checkpoint, end_of_track) – w sekcji ENV.

---

## 4. Znany problem: najszybsze przejazdy vs suma nagród

**Oczekiwane zachowanie:** najszybsze przejazdy (mniejsza liczba kroków, krótszy czas) powinny mieć **wyższą sumę nagród** niż przejazdy wolniejsze.

**Obecny problem:** w praktyce **nie zawsze tak jest**. Może się zdarzyć, że przejazd z **większą** liczbą kroków (wolniejszy) dostanie **wyższą** łączną nagrodę niż przejazd szybszy (mniej kroków). Przykłady z logów:

- Szybszy (np. 733 kroki, 36,65 s) → 560,6 pkt  
- Wolniejszy (np. 767 kroki, 38,35 s) → 555,9 pkt  

albo odwrotnie:

- Szybszy (747 kroki) → 560,6 pkt  
- Wolniejszy (765 kroki) → 560,9 pkt  

**Przyczyny:**

1. **Projected velocity** jest sumowana **per krok**. Teoretycznie ∫ v·cos(θ) dt na tej samej trasie powinna być podobna (ta sama droga), ale w praktyce suma zależy od liczby kroków i rozkładu prędkości/kąta – przy większej liczbie kroków suma składowych może być nieco wyższa lub niższa w zależności od stylu jazdy.
2. **Kary** (steering delta, boundary, wall-hug) zależą od konkretnej trajektorii; szybszy przejazd może mieć więcej ostrych skrętów i wyższe kary.
3. **Time bonus** (TIME_BONUS_SCALE / steps) **wyrównuje** ten efekt: mniej kroków → wyższy bonus. Przy małym TIME_BONUS_SCALE różnica bonusu nie zawsze przewyższa różnicę z punktów 1–2, więc wolniejszy przejazd może nadal wygrać sumą.

**Dlatego:** aby „najszybsze = wyższa suma” było spełnione stabilnie, w configu ustawia się **TIME_BONUS_SCALE** na wartość wystarczającą (np. 95000 przy REWARD_SCALE 0,46), tak aby przewaga bonusu dla szybszego przejazdu przeważyła ewentualną przewagę wolniejszego z projected velocity i kar. Dobór opisany jest w `docs/REWARD_CONFIG_TUNING.md`.

---

## 5. Logowanie i wandb

- Przy zakończeniu epizodu (`log_model_run`) wypisywane jest „Total reward of the run”, liczba kroków i szacowany czas (kroki × 0,05 s).
- W WandB logowane są m.in. `run/Run reward`, `run/Steps`, `run/Run time` oraz (opcjonalnie) statystyki rozkładu nagród krokowych.

---

## 6. Podsumowanie

- **Obecna nagroda (SOTA):** postęp wzdłuż trajektorii (~100 raw na okrążenie) + **projected velocity** (v·cos(θ)·dt) co krok + kary (boundary kwadratowa, steering delta, wall-hug, stall, crash) + bonusy (lap, checkpoint, end_of_track, **time bonus** na mecie). Wszystko skalowane przez `REWARD_SCALE` i obcinane od dołu.
- **Bonus za szybkość:** nie ma już post-epizodowego K/T² w buforze; szybkość jest w projected velocity i w **TIME_BONUS_SCALE** na mecie (im mniej kroków, tym wyższy bonus).
- **Znany problem:** najszybsze przejazdy powinny mieć wyższą sumę nagród niż wolniejsze; nie zawsze tak jest z powodu sumy projected velocity per krok i kar. TIME_BONUS_SCALE w configu służy do wzmocnienia promowania szybszych przejazdów i utrzymania sumy w docelowym zakresie (np. max ~750).
