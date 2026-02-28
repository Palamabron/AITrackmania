# Raport: SophyResidual RUN_L – analiza WandB, pipeline, problem ceilingu nagród

**Run WandB:** `tmrl/tmrl/SophyResidual_runv23_RUN_L TRAINER`  
**Stan runa:** failed  
**Czas trwania:** ~56 643 s (~15,7 h)  
**Liczba rund (kroków logowanych):** 500  
**Zakres _step:** 0 – 1439  

Dokument ma służyć jako raport handoff dla innego modelu/specjalisty: opisuje, co widać w metrykach, jak wygląda pipeline i konfiguracja, jaki jest problem (ceiling nagród ~53 przez nieumiejętność przejazdu zakrętu) oraz co już próbowano.

---

## 1. Co widać w WandB – wykresy i obserwacje

### 1.1 Nagroda i długość epizodu

| Metryka | Min | Max | Średnia | Początek (śr. pierwsze 10%) | Koniec (śr. ostatnie 10%) | Trend |
|--------|-----|-----|---------|------------------------------|----------------------------|--------|
| **return_train** | 0 | **59,99** | 20,83 | 14,14 | 18,64 | ↑ lekki wzrost |
| **episode_length_train** | 0 | 2207 | 718,9 | 510,5 | 654,6 | ↑ wzrost |

- **return_train:** W trakcie runu średni return lekko rośnie (z ~14 do ~19), ale **maksymalna osiągnięta nagroda to ~60** i run się przy tym zatrzymuje (ceiling). Ostatni zlogowany punkt to 13,93 – duża wariancja, wiele epizodów kończy się wcześniej.
- **episode_length_train:** Średnia długość epizodu rośnie (510 → 655), max 2207 kroków. Świadczy to o tym, że agent czasem jedzie dalej, ale **nie przejeżdża consistently trudnego zakrętu** – stąd plateau nagrody w okolicy ~53–60.

**Wniosek:** Wykresy `return_train` i `episode_length_train` pokazują częściowy postęp (dłuższe epizody, lekki wzrost średniego returnu), ale wyraźny **sufit ~53–60** i brak dalszego wzrostu po osiągnięciu tego poziomu. To spójne z tym, że model „zatrzymuje się” na jednym zakręcie.

---

### 1.2 Wartości Q i backup (target Q)

| Metryka | Min | Max | Początek (śr.) | Koniec (śr.) | Trend |
|--------|-----|-----|----------------|--------------|--------|
| **debug/q_a1** | 1,53 | 48,20 | 38,32 | 7,66 | ↓ silny spadek |
| **debug/q1** | 1,51 | 48,20 | 38,32 | 7,66 | ↓ |
| **debug/q2** | 1,53 | 48,22 | 38,32 | 7,66 | ↓ |
| **debug/backup** | 1,61 | 48,18 | 38,29 | 7,67 | ↓ |
| **debug/q_a1_targ** | 0,45 | 48,07 | 37,99 | 7,61 | ↓ |

- Wszystkie wartości Q i backup **mocno spadają** z ~38 na początku do ~7,6 na końcu.
- Odpowiada to **obniżaniu się estymaty „jak dobra jest obecna polityka”** w miarę napełniania bufora nowymi (słabszymi) danymi i mniejszego udziału demo.
- Nie ma tu ekstremalnej eksplozji (jak w Run Hv2 w eksperymentach), ale **trend ujemny jest wyraźny** – krytyk dostosowuje się do niższych returnów, które agent faktycznie osiąga (ceiling ~53).

**Wniosek:** Wykresy Q/backup pokazują **systematyczny spadek wartości** w czasie runu. To spójne z plateau nagród: agent nie poprawia się ponad ~53, więc targety i Q się obniżają.

---

### 1.3 Straty (losses)

| Metryka | Min | Max | Początek (śr.) | Koniec (śr.) | Trend |
|--------|-----|-----|----------------|--------------|--------|
| **losses/loss_actor** | -48,22 | -1,55 | -38,34 | -7,68 | ↑ (mniej ujemna) |
| **losses/loss_critic** | 0,016 | 0,293 | 0,127 | 0,057 | ↓ |

- **loss_actor** (ujemna w TQC – celem jest maksymalizacja): na początku ~-38, na końcu ~-7,7. „Trend w górę” oznacza **zmniejszanie się wielkości ujemnej** – polityka przestaje być tak agresywnie optymalizowana pod wysokie Q, bo Q spadły.
- **loss_critic** maleje (0,13 → 0,06) – krytyk się zbiega do niższych wartości.

**Wniosek:** Wykresy lossów są spójne z opisanym wyżej obrazem: brak eksplozji, ale **dostosowanie się do ceilingu nagród** (niższe Q → mniej ujemny loss aktora, mniejszy loss krytyka).

---

### 1.4 Demo i bufor

| Metryka | Min | Max | Początek (śr.) | Koniec (śr.) | Trend |
|--------|-----|-----|----------------|--------------|--------|
| **debug/demo_fraction_in_batch** | 0 | 1,0 | **0,931** | 0,131 | ↓ |
| **debug/demo_sampling_weight** | 1,0 | 1,15 | 1,11 | 1,0 | ↓ |
| **memory_len** | 5411 | 1 000 000 | 55 595 | 985 551 | ↑ |

- **demo_fraction_in_batch:** Na początku ~93% batchy z demo, na końcu ~13%. **Demo weight decay działa** – udział demo w treningu maleje w czasie.
- **demo_sampling_weight:** Z ~1,11 schodzi do 1,0 (bez biasu) – zgodnie z konfiguracją `DEMO_WEIGHT_DECAY_SAMPLES`.
- **memory_len:** Bufor rośnie z ~55k do ~986k próbek – trening na pełnym buforze w drugiej połowie runu.

**Wniosek:** Wykresy demo/bufora potwierdzają, że **imitation bias jest włączony i decay działa**. Mimo 20 ludzkich przejazdów (w tym przez problematyczny zakręt) agent **nie nauczył się tego zakrętu na tyle, by przełamać ceiling ~53**.

---

### 1.5 Krok nagrody (debug/r), gradienty, akcje

| Metryka | Początek (śr.) | Koniec (śr.) | Trend |
|--------|----------------|--------------|--------|
| **debug/r** | 0,382 | 0,078 | ↓ |
| **debug/critic_grad_norm** | 5,21 | 2,97 | ↓ |
| **debug/actor_grad_norm** | 0,018 | 0,012 | ↓ |
| **debug/a_0** (np. gaz) | 0,763 | 0,106 | ↓ |

- **debug/r:** Średnia nagroda per krok **spada** (0,38 → 0,08). W drugiej połowie treningu agent częściej zbacza / kończy wcześniej, więc średnia nagroda per krok jest niższa.
- **Gradienty** maleją – bez oznak eksplozji; gradient clipping prawdopodobnie nie był mocno angażowany (wartości < 1).
- **debug/a_0** (prawdopodobnie akcja „gaz”): Silny spadek (0,76 → 0,11). Może to oznaczać **mniej agresywne gazowanie** w późniejszej polityce (np. ostrożniejsza jazda, która i tak nie przejeżdża zakrętu).

---

### 1.6 Podsumowanie obserwacji z WandB

1. **Return_train** ma wyraźny **ceiling ~53–60**; średni return lekko rośnie, ale ostatni punkt i rozkład pokazują, że agent nie przełamuje tego progu.
2. **Q i backup** systematycznie spadają (~38 → ~7,6) – brak eksplozji, ale **obniżanie się wartości** w zgodzie z plateau nagród.
3. **Lossy** są stabilne; brak oznak rozjazdu treningu (w przeciwieństwie do wcześniejszych runów bez gradient/backup clipping).
4. **Demo** są używane (wysoki udział na początku), decay działa; mimo to **imitation nie wystarcza**, by nauczyć zakrętu.
5. **Długość epizodu** rośnie (śr. 510 → 655, max 2207), co pokazuje postęp, ale **niewystarczający** do pełnego przejazdu po problematycznym zakręcie.

---

## 2. Pipeline i pełna konfiguracja (parametry)

Poniżej opis pipeline’u i **wszystkie parametry** z podanego configu (RUN_M – spójny z serią RUN_L).

### 2.1 Ogólne (root)

| Parametr | Wartość | Opis |
|----------|---------|------|
| RUN_NAME | SophyResidual_runv23_RUN_M | Nazwa runu |
| RESET_TRAINING | false | Nie resetuj treningu przy starcie |
| BUFFERS_MAXLEN | 5 000 000 | Maks. łączna długość buforów |
| RW_MAX_SAMPLES_PER_EPISODE | 28 000 | Maks. próbek na epizod w R2D2 |
| CUDA_TRAINING | true | Trening na GPU |
| CUDA_INFERENCE | false | Inferencja na CPU |
| VIRTUAL_GAMEPAD | true | Wirtualny gamepad |
| DCAC | false | Bez DCAC |
| LOCALHOST_* | true | Trainer/worker/server na localhost |
| PUBLIC_IP_SERVER | 127.0.0.1 | IP serwera |
| TLS | false | Bez TLS |
| NB_WORKERS | -1 | Auto liczba workerów |
| WANDB_PROJECT | tmrl | Projekt WandB |
| WANDB_ENTITY | tmrl | Entity WandB |
| WANDB_WORKER | true | Logowanie z workera |
| WANDB_DEBUG_REWARD | true | Debug reward w WandB |
| PORT | 55555, LOCAL_PORT_* 55556–55558 | Porty sieciowe |
| BUFFER_SIZE | 536870912 | Rozmiar bufora (bajty) |
| Różne SOCKET_TIMEOUT_*, ACK_TIMEOUT_*, RECV_TIMEOUT_* | 30–7200 s | Timeouty połączeń |

### 2.2 MODEL

| Parametr | Wartość | Opis |
|----------|---------|------|
| MAX_EPOCHS | 10 000 | Maks. liczba epok |
| ROUNDS_PER_EPOCH | 10 | Rund na epokę |
| TRAINING_STEPS_PER_ROUND | 300 | Kroków treningu na rundę |
| MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP | 6 | Stosunek trening : dane (ograniczenie) |
| ENVIRONMENT_STEPS_BEFORE_TRAINING | 5000 | Warmup przed treningiem |
| UPDATE_MODEL_INTERVAL | 1000 | Co ile kroków aktualizować model (worker) |
| UPDATE_BUFFER_INTERVAL | 1000 | Co ile wysyłać bufor |
| SAVE_MODEL_EVERY | 0 | Nie zapisuj co N epok (używany best checkpoint) |
| MEMORY_SIZE | 1 000 000 | Rozmiar pamięci R2D2 |
| BATCH_SIZE | 512 | Wielkość batcha |
| SCHEDULER | NAME: "" | Bez schedulera LR |
| NOISY_LINEAR_CRITIC/ACTOR | false | Bez noisy layers |
| OUTPUT_DROPOUT, RNN_DROPOUT | 0 | Bez dropoutu |
| CNN_FILTERS | [32, 64, 64, 64] | Filtrów w warstwach CNN |
| CNN_OUTPUT_SIZE | 256 | Wymiar po CNN |
| RNN_LENS | [1], RNN_SIZES | [64] | LSTM 1 warstwa, 64 jednostki |
| API_MLP_SIZES | [256, 256] | MLP po API |
| API_LAYERNORM, MLP_LAYERNORM | true | LayerNorm włączony |
| **USE_RESIDUAL_SOPHY** | **true** | **Residual Sophy włączony** |
| RESIDUAL_MLP_HIDDEN_DIM | 256 | Wymiar ukryty residual MLP |
| RESIDUAL_MLP_NUM_BLOCKS | 3 | Liczba bloków residual |
| USE_FROZEN_EFFNET | false | Bez zamrożonego EffNet |

### 2.3 ALG (TQC)

| Parametr | Wartość | Opis |
|----------|---------|------|
| ALGORITHM | TQC | Truncated Quantile Critic |
| LEARN_ENTROPY_COEF | false | Stałe alpha |
| LR_ACTOR | 1e-5 | Learning rate aktora |
| LR_CRITIC | 5e-5 | Learning rate krytyka |
| LR_ENTROPY | 5e-5 | (niewykorzystywane przy LEARN_ENTROPY_COEF=false) |
| GAMMA | 0.9925 | Dyskonto |
| POLYAK | 0.995 | Tau dla target network |
| TARGET_ENTROPY | -3.0 | Docelowa entropia (przy learn=true) |
| ALPHA | 0.01 | Stały współczynnik entropii |
| REDQ_* | 10, 2, 20 | Parametry REDQ (jeśli używane) |
| TOP_QUANTILES_TO_DROP | 8 | Ile górnych kwantyli odrzucać (TQC) |
| QUANTILES_NUMBER | 25 | Liczba kwantyli |
| N_STEPS | 3 | n-step returns |
| R2D2_REWIND | 0.5 | Rewind w R2D2 |
| OPTIMIZER_ACTOR/CRITIC | adam | Optymalizatory |
| BETAS_ACTOR/CRITIC | [0.997, 0.997] | Betas Adama |
| L2_ACTOR/CRITIC | 0 | Bez L2 |
| NUMBER_OF_POINTS | 10 | (kontekst nagrody) |
| SPEED_BONUS | 0.0008 | Bonus za prędkość |
| SPEED_MIN_THRESHOLD | 30 | Próg prędkości (min) |
| SPEED_MEDIUM_THRESHOLD | 20 | Próg prędkości (medium) |

(W configu mogą być też GRAD_CLIP_ACTOR/CRITIC, BACKUP_CLIP_RANGE – patrz docs/EXPERIMENTS_AND_HYPERPARAMETERS_REPORT.md.)

### 2.4 PLAYER_RUNS (demo)

| Parametr | Wartość | Opis |
|----------|---------|------|
| ONLINE_INJECTION | true | Demo wstrzykiwane na bieżąco |
| SOURCE_PATH | (ścieżka Windows) | Katalog z nagraniami gracza |
| CONSUME_ON_READ | true | Pliki „skonsumowane” po wczytaniu |
| MAX_FILES_PER_UPDATE | 6 | Maks. plików demo na update |
| DEMO_INJECTION_REPEAT | 1 | Powtórzenie iniekcji demo |
| DEMO_SAMPLING_WEIGHT | 1.15 | Waga próbkowania demo (na początku) |
| DEMO_WEIGHT_DECAY_SAMPLES | 200 000 | Po tylu próbkach w buforze waga demo → 1.0 |

### 2.5 ENV i nagroda

| Parametr | Wartość | Opis |
|----------|---------|------|
| RTGYM_INTERFACE | TQCGRAB | Interfejs rtgym |
| INIT_GAS_BIAS | 0.8 | Początkowy bias gazu |
| MAP_NAME | test-3 | Mapa |
| END_OF_TRACK_REWARD | 5.0 | Nagroda za koniec toru |
| WINDOW_WIDTH/HEIGHT | 2048×1024 | Okno gry |
| IMG_WIDTH/HEIGHT | 64×64 | Obraz dla sieci |
| USE_IMAGES | false | **Bez obrazów** (tylko API?) |
| IMG_GRAYSCALE | true | Skala szarości (gdy USE_IMAGES) |
| SLEEP_TIME_AT_RESET | 1.5 | Czekanie przy resecie |
| IMG_HIST_LEN | 4 | Długość historii obrazów |
| RTGYM_CONFIG | time_step 0.05, ep_max_length 15000, act_buf_len 2, wait_on_done true, itp. | Konfiguracja rtgym |

**REWARD_CONFIG:**

| Parametr | Wartość | Opis |
|----------|---------|------|
| CONSTANT_PENALTY | 0.0003 | Stała kara per krok |
| CHECK_FORWARD | 500 | Sprawdzanie postępu do przodu |
| CHECK_BACKWARD | 10 | Sprawdzanie wstecz |
| FAILURE_COUNTDOWN | 9 | Kroki do failure |
| MIN_STEPS | 70 | Min. kroków przed failure |
| MAX_STRAY | 50.0 | Maks. odchylenie od trasy |
| SPEED_SAFE_DEVIATION_RATIO | 0.15 | Bezpieczny stosunek odchylenia do prędkości |
| RECKLESS_SPEED_THRESHOLD | 45 | Próg „reckless” (km/h?) |
| RECKLESS_PENALTY_FACTOR | 0.006 | Kara za reckless |
| WALL_HUG_SPEED_THRESHOLD | 10.0 | Próg prędkości przy wall-hug |
| WALL_HUG_PENALTY_FACTOR | 0.005 | Kara za wall-hug |
| PROXIMITY_REWARD_SHAPING | 0.5 | Kształtowanie nagrody za bliskość trasy |
| REWARD_SCALE | 2.0 | Skala nagrody przed tanh |

---

## 3. Sformułowanie problemu (dla handoffu)

- **Obserwacja:** Nagroda treningowa (`return_train`) ma **ceiling w okolicy ~53**. Agent nie poprawia się powyżej tego poziomu (maks. w runie ~60, typowo plateau ~53).
- **Przyczyna (z perspektywy użytkownika):** Model **nie potrafi przejechać konkretnego zakrętu** na mapie. Po tym zakręcie mógłby zdobywać dalszą nagrodę (np. do końca toru), więc ceiling wynika z **wczesnego końca epizodu na tym zakręcie**.
- **Kontekst:** W **replay bufferze są 20 własnych przejazdów użytkownika**, w których **cała trasa jest przejechana**, **w tym ten zakręt**. Mimo to polityka **nie wykorzystuje tych demo na tyle, by nauczyć się tego manewru** i przełamać ceiling.
- **Pytanie do rozwiązania:** Dlaczego imitation z 20 pełnymi przejazdami (z zakrętem) nie wystarcza, by policy nauczyła się tego zakrętu? Jak zmienić pipeline / algorytm / reward / architekturę / sposób użycia demo, żeby **przełamać ceiling ~53** i jechać dalej (wyższy return, dłuższe epizody)?

---

## 4. Co już próbowano (na podstawie EXPERIMENTS_AND_HYPERPARAMETERS_REPORT.md)

Poniżej skrót eksperymentów i zmian z `docs/EXPERIMENTS_AND_HYPERPARAMETERS_REPORT.md`, żeby kolejny model nie powtarzał tych samych kierunków bez potrzeby.

1. **Baseline (pre–Run A):** LEARN_ENTROPY_COEF=true, TARGET_ENTROPY=-0.5, wysokie LR_ENTROPY, duży train ratio → eksplozja entropy, Q i loss_actor; q1==q2 (twin critics nie różnicowane). **Fix:** Różne seedy dla q1/q2, TARGET_ENTROPY=-3, mniejszy train ratio, dłuższy warmup, TOP_QUANTILES_TO_DROP=5.
2. **Run A:** Zamrożenie alpha (LEARN_ENTROPY_COEF=false, ALPHA=0.027). Stabilność bez eksplozji alpha; Q i tak dryfowały ujemnie.
3. **Run B:** Niższe alpha (0.0125), GAMMA 0.99, MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP=6. **Duża poprawa:** mniejszy dryf Q, stabilne lossy, lepszy trend returnu i długości epizodu.
4. **Run C:** TOP_QUANTILES_TO_DROP=8. Stabilnie, return do ~11.8, episode_length do ~964.
5. **Run D:** ALPHA=0.01. Stabilnie, ale plateau returnu i wiele krótkich epizodów.
6. **Run E:** SPEED_BONUS=0.0008, MAX_FILES_PER_UPDATE=5. Nieco lepszy sygnał, nadal plateau i krótkie epizody.
7. **Run F:** Zaostrzenie REWARD_CONFIG (MAX_STRAY 50, SPEED_SAFE_DEVIATION_RATIO 0.15, RECKLESS_* 45/0.008, FAILURE 7, CONSTANT_PENALTY 0.001). **Efekt negatywny:** słabszy sygnał nagrody (debug/r spadł), większa wariancja; zalecany rollback (Run G).
8. **Run G (sugerowany):** RECKLESS_PENALTY 0.005, CONSTANT_PENALTY 0.0005, FAILURE_COUNTDOWN 9 – złagodzenie Run F.
9. **Imitation bias (kod + config):** Tagowanie demo (`is_demo`), DEMO_INJECTION_REPEAT, DEMO_SAMPLING_WEIGHT, demo_fraction_in_batch. Rekomendowane DEMO_SAMPLING_WEIGHT 2.5, DEMO_INJECTION_REPEAT 2. **Nie ocenione w długim runie** jako osobny eksperyment.
10. **Final reward overhaul (kod):** Proximity reward shaping, wall-hug penalty/termination, off-track speed penalty, REWARD_SCALE, nowe klucze REWARD_CONFIG. **Nie uruchomione** w raporcie.
11. **Training stability (Run Hv2 i fixy):** Przy 40 demo i fazie demo-only Q i loss_actor eksplodowały; demo_fraction ~0.93 przez cały czas. **Wprowadzone:** gradient clipping (actor/critic), backup (target Q) clipping, **demo weight decay** (DEMO_WEIGHT_DECAY_SAMPLES), zapis **best checkpoint** (best_actor.pth). Te mechanizmy są w kodzie i w RUN_L (decay i stabilność widać w metrykach).

**Podsumowanie dla handoffu:** Próbowano m.in. stabilizacji alpha/gamma/train ratio, TQC (top quantiles), reward shaping (speed bonus, wall-hug, proximity), zaostrzania i łagodzenia kar, imitation bias z demo oraz **demo weight decay + gradient/backup clipping + best checkpoint**. Run RUN_L jest już z tymi ostatnimi fixami; ceiling ~53 przy 20 demo z pełnym przejazdem (w tym zakręt) **nadal występuje**. Szukane jest rozwiązanie, które **wykorzysta demo do nauczenia konkretnego zakrętu** i podbije return powyżej ~53.

---

## 5. Rekomendacje z raportu eksperymentów (do rozważenia)

- Użyć „final reward overhaul” (proximity shaping, wall-hug, REWARD_SCALE) z zalecanymi wartościami.
- Zachować ustawienia algorytmu z Run B/C/D: LEARN_ENTROPY_COEF=false, ALPHA=0.01, GAMMA=0.99, MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP=6, TOP_QUANTILES_TO_DROP=8.
- Włączyć imitation bias i dodać 10–20 zróżnicowanych przejazdów (czyste lapty + recovery).
- Gradient/backup clipping włączone; monitorować critic/actor_grad_norm i demo_sampling_weight.
- W razie collapse’u ładować `TmrlData/weights/best_actor.pth`.

---

## 6. Metryki WandB – lista (do odtworzenia wykresów)

- **Nagroda / długość:** return_train, episode_length_train, return_test, episode_length_test  
- **Q / backup:** debug/q_a1, debug/q1, debug/q2, debug/backup, debug/q_a1_targ, *_std  
- **Lossy:** losses/loss_actor, losses/loss_critic  
- **Demo / bufor:** debug/demo_fraction_in_batch, debug/demo_sampling_weight, memory_len  
- **Krok:** debug/r, debug/r_std  
- **Gradienty:** debug/critic_grad_norm, debug/actor_grad_norm  
- **Akcje (przykłady):** debug/a_0, debug/a_1, debug/a_2, debug/a1_*, debug/a2_*  
- **Inne:** debug/log_pi, debug/logp_a2, lrs/actor_lr, lrs/critic_lr, round_time, training_step_duration, idle_time, sampling_duration  

---

*Raport wygenerowany na podstawie runu WandB `tmrl/tmrl/SophyResidual_runv23_RUN_L TRAINER`, konfiguracji RUN_M oraz dokumentu `docs/EXPERIMENTS_AND_HYPERPARAMETERS_REPORT.md`.*
