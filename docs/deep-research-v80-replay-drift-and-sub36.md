# Prompt do Deep Research: kryminalistyczna diagnoza replayu i droga do stabilnego 35–36 s w TrackmaniaRL

> Historyczny prompt badawczy dla konkretnego eksperymentu v80. Nie jest
> dokumentacją aktualnego API. W RunSpec 2.0 IQN/FQF korzystają z kompozycji
> modelu i `DiscreteValueLearner`; stare checkpointy IQN służą wyłącznie jako
> jawny warm-start.

Jesteś głównym badaczem i architektem systemu sterowania/RL dla TrackManii. Otrzymujesz ZIP aktualnego repozytorium `trackmaniarl` oraz ten prompt. Nie masz dostępu do W&B, uruchomionej gry ani komputera autora. Masz wykonać kryminalistyczny audyt kodu, zweryfikować kontrakty danych i zaprojektować możliwie najkrótszą, mierzalną drogę do agenta przejeżdżającego mapę stabilnie w 35–36 sekund. Liczy się wynik na tej jednej mapie, nie ogólność rozwiązania.

Nie traktuj wcześniejszych diagnoz jako prawdy. Sprawdź każdą z nich bezpośrednio w kodzie. Wyraźnie oddziel:

1. błędy udowodnione z kodu;
2. bardzo prawdopodobne przyczyny zgodne z obserwacjami;
3. hipotezy wymagające eksperymentu w grze;
4. pomysły algorytmiczne, które nie rozwiązują aktualnego problemu kontraktu wykonawczego.

## Cel i twarde kryteria

Docelowy release gate:

- co najmniej 27/30 ukończonych przejazdów;
- mediana czasu `<= 36.0 s`;
- docelowo najlepszy czas `35.x s`;
- brak błędów telemetry/controller;
- brak ręcznej ingerencji podczas przejazdu;
- wynik musi pochodzić z polityki reagującej na stan, kontrolera trajektorii albo połączenia obu — nie musi być klasycznym „czystym RL”, jeśli inne rozwiązanie daje lepszy rezultat.

Ekspert przejechał mapę klawiaturą w `35.855 s`, więc przestrzeń binarnych wejść jest fizycznie wystarczająca.

## Najnowszy stan i najważniejszy materiał dowodowy

Nowy recorder utworzył demonstrację:

lokalny artefakt demonstracyjny `demo-01-35.855s.npz` (nie jest częścią repozytorium)

Parametry nagrania:

- format: `trackmaniarl-trackmania-demo-v4`;
- `control_alignment=transition_end`;
- 3585 przejść i 3586 ramek;
- pierwsza ramka: `10 ms`;
- ostatnia ramka: `35860 ms`;
- interwał median/p95/max: dokładnie `10/10/10 ms`;
- 99 zmian sterowania;
- najkrótszy impuls: `10 ms`;
- kontrola dyskretna: gaz, hamulec i kierownica `{-1, 0, +1}`.

Rozkład surowych akcji demonstracji:

- raw `0`: 203 próbki, `[gas=0, brake=0, steer=-1]`;
- raw `1`: 13 próbek, `[0, 1, -1]`;
- raw `3`: 770 próbek, `[1, 0, -1]`;
- raw `36`: jedna próbka terminalna `[0, 0, 0]`, wynik wyzerowania wejść przez ekran Finish;
- raw `39`: 1320 próbek, `[1, 0, 0]`;
- raw `72`: 215 próbek, `[0, 0, +1]`;
- raw `73`: 175 próbek, `[0, 1, +1]`;
- raw `75`: 888 próbek, `[1, 0, +1]`.

Pierwsze przełączenia sterowania w demonstracji:

- wejście w pierwszy krótki skręt: około `580 ms`, neutral → `steer +1`;
- wyjście: około `670 ms`, `+1` → neutral;
- kolejny istotny skręt: około `2770 ms`, neutral → `steer -1`.

Po kolejnych poprawkach czysty open-loop replay działa następująco:

```text
Open-loop replay interval: 10.000 ms
trial=0 finished=False progress=24.8%
telemetry_error=- controller_error=-
```

Samochód jedzie we właściwą stronę i pokonuje początek trasy, ale małe błędy kumulują się i kończy około 24,8% mapy.

Wcześniejsze wyniki diagnostyczne:

- przy odwróconym znaku kierownicy replay kończył przy około 3,5%;
- po poprawieniu znaku osiąga 24,8%;
- wcześniejszy trajectory tracker z agresywnym feedbackiem osiągał około 40,7%, ale wykonywał slalom i odbijał się od ścian;
- po kolejnych ręcznych zmianach regulatora niektóre warianty kończyły przy 2–7%, co pokazuje, że znak, faza i feedback były wielokrotnie wzajemnie maskowane;
- starsza polityka IQN/R2D2 kończyła 30/30 z medianą około 37,4–38,3 s, więc środowisko i pełny rollout potrafią działać stabilnie, ale polityka nie uzyskuje docelowego tempa;
- ekspercka demonstracja `35.855 s` dowodzi, że cel jest osiągalny na tej mapie i tym typem wejścia.

## Aktualna konwencja sterowania

Po weryfikacji wizualnej w grze backend klawiatury został ustawiony na:

- `InputSteer -1 -> A -> fizycznie w lewo`;
- `InputSteer +1 -> D -> fizycznie w prawo`.

Sprawdź implementację w `trackmaniarl/trackmania/control.py`, zwłaszcza `KeyboardController.apply()` oraz `_apply_without_timer()`. Nie zakładaj, że nazwa osi, znak `yaw` lub układ współrzędnych wystarczają do rozstrzygnięcia kierunku — najnowszy dowód pochodzi z bezpośredniej obserwacji replayu. Zaproponuj trwały test integracyjny kalibrujący znak na podstawie zmiany pozycji/yaw po kontrolowanym impulsie, aby podobna regresja nie mogła wrócić.

## Najbardziej podejrzany kontrakt przyczynowy

Recorder v4 w `trackmaniarl/trackmania/demonstrations.py` działa obecnie w przybliżeniu tak:

1. zachowuje początkową ramkę `frame[i]`;
2. odczytuje następną ramkę `frame[i+1]`;
3. wyciąga `InputGas/InputBrake/InputSteer` z `frame[i+1]`;
4. zapisuje ten input jako akcję przejścia `frame[i] -> frame[i+1]`;
5. ustawia `control_alignment="transition_end"`.

Natomiast `DemonstrationReplayPolicy.from_path()` w `trackmaniarl/trackmania/guidance.py` obecnie paruje:

```python
demonstration.frames[:-1, 3]
demonstration.actions
```

i podczas online replayu wybiera akcję na podstawie aktualnego `race_time_ms`.

To może oznaczać, że akcja obserwowana na końcu przejścia przy `t+10 ms` jest emitowana już przy `t`, czyli nadal o jeden tick za wcześnie. Z drugiej strony `api.InputSteer` może być wejściem obowiązującym dla nadchodzącego ticka, zależnie od kolejności aktualizacji OpenPlanet/ScriptAPI/physics. Tego nie wolno rozstrzygnąć intuicyjnie. To centralne pytanie audytu:

> Czy `CSmScriptPlayer.InputSteer/InputGasPedal/InputIsBraking` odczytane wraz z pozycją i `RaceTime` opisują input, który spowodował aktualnie raportowany stan, input aktywny w bieżącym ticku, czy input przeznaczony dla następnego ticka fizyki?

Znajdź najbardziej autorytatywne informacje w dokumentacji OpenPlanet/Nadeo/ManiaPlanet albo wywnioskuj kolejność z implementacji pluginu. Jeżeli dokumentacja nie rozstrzyga, zaprojektuj minimalny eksperyment identyfikacyjny, który ustali odpowiedź bez zgadywania.

## Kod wymagający obowiązkowego audytu

Przejrzyj co najmniej:

- `trackmaniarl/project/openplanet/TrackmaniaRL_GrabData_IQN.as`
  - pętla `Main()`;
  - kolejność odczytu `RaceTime`, pozycji, prędkości oraz inputów;
  - częstotliwość `yield()` i związek z render tickiem/fizyką;
- `trackmaniarl/trackmania/telemetry.py`
  - `OpenPlanetClient.read()` versus `read_next()`;
  - bufor TCP, kompletność ramek, reconnect i możliwe opóźnienia;
- `trackmaniarl/trackmania/demonstrations.py`
  - `_wait_for_new_run()`;
  - `record_demonstration()`;
  - `_advance_demonstration_frame()`;
  - `control_alignment`;
  - terminalne wyzerowanie inputów;
  - bramka jakości 100 Hz;
  - `resample_demonstration()`;
- `trackmaniarl/trackmania/guidance.py`
  - `DemonstrationReplayPolicy`;
  - `TrajectoryTrackingDemonstrationPolicy`;
  - indeksowanie referencji;
  - koszt nearest-state;
  - action lead;
  - znaki błędów bocznych/heading/lateral velocity;
  - histereza, cooldown i impulsy recovery;
- `trackmaniarl/trackmania/control.py`
  - sposób wysyłania `SendInput`;
  - kolejność key-up/key-down;
  - koszt czasowy i jitter;
  - reset;
- `trackmaniarl/trackmania/environment.py`
  - `decision_interval_ms`;
  - kiedy akcja jest aplikowana względem oczekiwania na następną ramkę;
  - `_last_race_time_ms`, `step_race_time_ms` i ewentualne pomijanie ticków;
- `trackmaniarl/trackmania/evaluation.py`
  - reset policy/pipeline;
  - moment pierwszego `act()`;
  - czy pierwsza akcja jest emitowana przy 0, 10 czy 20 ms;
- `trackmaniarl/cli.py`
  - `demo-benchmark`, `--open-loop`, `--trajectory-tracking`, `--action-lead-ms`;
- testy dotyczące demonstrations, telemetry, keyboard control i trajectory guidance.

Podawaj konkretne pliki, funkcje i aktualne numery linii z przesłanego ZIP-a. Jeśli aktualny kod różni się od opisu, kod ma pierwszeństwo.

## Pytania, na które raport musi odpowiedzieć

### A. Czy demonstracja v4 jest rzeczywiście przyczynowo wyrównana?

1. Czy `transition_end` jest poprawną semantyką?
2. Z jakim timestampem powinna być zapisana akcja: `frame[i].time`, `frame[i+1].time`, środek przedziału czy osobny timestamp zdarzenia wejściowego?
3. Czy open-loop powinien używać `searchsorted(..., side="left")`, `side="right"`, przesunięcia o jeden indeks albo empirycznego opóźnienia?
4. Czy 10 ms raportowane przez `RaceTime` jest tickiem fizyki, renderowania czy jedynie kwantyzacją timera?
5. Czy `yield()` w OpenPlanet gwarantuje dokładnie jedną próbkę na tick, czy może generować wiele/przegapiać próbki?
6. Czy pierwsza akcja demonstracji powinna być zastosowana przed pierwszą dodatnią ramką czasu wyścigu?
7. Czy terminalna akcja ma jakiekolwiek znaczenie dla treningu/replayu?

### B. Dlaczego open-loop dryfuje do 24,8% mimo idealnego 10 Hz/100 Hz kontraktu próbek?

Rozważ i uszereguj na podstawie kodu:

- off-by-one w action timestamp;
- fazę startu epizodu;
- jitter Windows `SendInput` i przełączanie klawiszy poza tickiem gry;
- różnicę między czasem rejestracji inputu a czasem jego zastosowania przez grę;
- zmienny framerate i kolejność OpenPlanet callbacków;
- niedeterministyczność fizyki/pozycji początkowej;
- `read()` opróżniające bufor w środowisku online;
- pomijanie ticków przez pętlę środowiska;
- emisję key-up i key-down jako dwóch oddzielnych zdarzeń;
- błędne odwzorowanie krótkich 10 ms impulsów;
- różnice między ręcznym sterowaniem podczas nagrania a programowym `SendInput` podczas replayu;
- fakt, że nagranie klawiatury opisuje stan klawiszy, a nie dokładny czas zdarzenia key-down/key-up.

Wyjaśnij, które z tych czynników open-loop może naprawić, a które czynią idealny replay otwartej pętli nierealnym.

### C. Jak zbudować właściwy instrument pomiarowy?

Zaprojektuj wersję telemetry/replay trace zapisującą co tick do NPZ/JSONL co najmniej:

- numer próbki producenta i konsumenta;
- `RaceTime`;
- lokalny `perf_counter_ns` przy odebraniu ramki;
- pozycję, prędkość i heading;
- input raportowany przez OpenPlanet;
- akcję żądaną przez policy;
- akcję faktycznie wysłaną przez backend;
- czasy key-down/key-up;
- indeks demonstracji wybrany przez replay;
- planowany timestamp akcji;
- liczbę pominiętych ramek;
- opóźnienie policy → controller → następna obserwowana zmiana inputu;
- lateral/heading/speed error względem demonstracji;
- pierwszy moment przekroczenia błędu 0,1 m, 0,25 m, 0,5 m i 1 m.

Wskaż dokładne miejsca w kodzie, gdzie logować te wartości. Zaproponuj automatyczny raport porównujący demonstrację z replayem i lokalizujący pierwszy punkt rozjazdu zamiast raportowania wyłącznie końcowego progressu.

### D. Jak powinien wyglądać stabilny kontroler zamknięty dla binarnej klawiatury?

Dotychczasowy prosty PD wybierał pełne `-1/0/+1` co 10 ms, powodując slalom. Zaprojektuj konkretny tracker odpowiedni dla tej mapy i danych:

- feed-forward z eksperckiej sekwencji akcji;
- phase/state synchronization bez cofania indeksu;
- lookahead trajektorii zależny od prędkości, nie tylko najbliższy punkt;
- błąd poprzeczny i heading w układzie Freneta;
- preview curvature / pure pursuit / Stanley / LQR / MPC — wybierz najbardziej opłacalny wariant dla binarnej kierownicy;
- stan dyskretnego automatu kierownicy;
- histereza Schmitta osobno dla włączenia i wyłączenia skrętu;
- minimum hold time oraz minimum neutral time;
- zakaz natychmiastowego `left → right` bez neutralnej fazy;
- korekta tylko wtedy, gdy przewidywany błąd za 100–300 ms rośnie;
- harmonogram progów zależny od prędkości i krzywizny;
- oddzielna kontrola gazu/hamulca oparta o prędkość referencyjną;
- ograniczenie feedbacku tak, aby nie niszczył feed-forward eksperta;
- tryb reacquisition po dużym odchyleniu;
- logika finish segmentu i ostatnich 50 m.

Podaj równania, znaki, jednostki i pseudokod. Zaproponuj początkowe wartości parametrów oraz mały grid do strojenia. Wyjaśnij, jak zweryfikować każdy znak kontrolera testem fizycznym, a nie założeniem o układzie współrzędnych.

Oceń również, czy lepiej:

1. pozostać przy klawiaturze i automacie dyskretnym;
2. użyć wirtualnego gamepada z analogową korekcją, mimo że demonstracje pochodzą z klawiatury;
3. wysyłać wejścia bezpośrednio z pluginu/OpenPlanet, jeśli API i zasady środowiska lokalnego na to pozwalają;
4. użyć trajectory controller jako bezpiecznego nauczyciela dla DAgger, a następnie wytrenować szybką politykę reaktywną.

### E. Jak dojść od eksperta 35.855 s do modelu 35–36 s?

Dopiero po rozwiązaniu wykonania i trackera oceń pipeline uczenia. Aktualny projekt ma implementacje BC, IQN/R2D2, prioritized replay, n-step, demonstration margin/cross-entropy/TD, synthetic recovery, DAgger oraz offline pretrain. Przeanalizuj je z kodu.

Porównaj konkretnie:

- BC na pojedynczej demonstracji `35.855 s`;
- BC na kilku przejazdach v4 w zakresie 35.8–36.8 s;
- oversampling okien przed zmianą akcji;
- klasyfikację następnej akcji kontra przewidywanie czasu do kolejnego switcha;
- model hybrydowy: `expert phase action + learned residual/switch timing`;
- sequence model GRU versus model bez pamięci z jawnie podanym previous action/hold duration;
- Transformer/attention wyłącznie jeśli jego koszt i korzyść są uzasadnione;
- DAgger z interwencjami człowieka;
- synthetic recovery oparte na fizycznie realistycznych perturbacjach, a nie arbitralnym przesunięciu ramki;
- DQfD/IQN offline pretraining;
- conservative Q-learning/AWAC/IQL offline — tylko jeśli pasują do dyskretnych danych i istniejącego kodu;
- online fine-tuning z bardzo niskim epsilonem i stałą frakcją demonstracji;
- self-imitation tylko z przejazdów szybszych od aktualnej kotwicy;
- trajectory optimization / optymalizacja czasów switchy jako alternatywa dla kosztownego RL.

Nie rekomenduj nowej architektury tylko dlatego, że jest nowocześniejsza. Dla każdej propozycji podaj:

- dlaczego pasuje do tej jednej mapy;
- co dokładnie trzeba zmienić w obecnym kodzie;
- koszt wdrożenia;
- ryzyko;
- oczekiwany wpływ na finish rate i czas;
- kryterium odrzucenia eksperymentu.

## Oczekiwany plan eksperymentalny

Zaprojektuj sekwencję bramek, w której nie wolno przejść dalej bez spełnienia poprzedniej:

### Gate 0 — kalibracja znaku i opóźnienia

Minimalny kontrolowany impuls lewo/prawo/gaz, pomiar input echo, yaw i displacement. Wynik ma automatycznie ustalić znak i opóźnienie kontrolera.

### Gate 1 — identyfikacja alignmentu

Nagraj prosty eksperyment z kilkoma długimi impulsami w znanych timestampach. Porównaj warianty start/end/midpoint oraz offsety np. `-30…+30 ms`. Kryterium: jednoznaczne minimum błędu trajektorii i zgodność w wielu restartach.

### Gate 2 — open-loop fidelity

Nie wymagaj ukończenia pełnej trasy, jeśli open-loop jest fundamentalnie niestabilny. Zdefiniuj mierzalne kryteria: pierwszy czas do błędu 0,25 m, RMS przez pierwsze 5/10 s, zgodność switchy i liczba pominiętych ramek.

### Gate 3 — closed-loop trajectory tracker

Minimum 9/10 ukończeń; początkowo czas `<40 s`, następnie `<37 s`; brak slalomu mierzony liczbą przeciwstawnych switchy w krótkim oknie.

### Gate 4 — behavior cloning

BC nie może być oceniane tylko accuracy/loss. Kryterium rollout: najpierw 9/10 ukończeń `<40 s`, potem mediana `<37 s`. Walidacja musi być dzielona po całych epizodach/segmentach, nie losowych sąsiednich ramkach.

### Gate 5 — offline-to-online improvement

Model startuje z działającej polityki. Stop natychmiast, jeśli finish rate spada poniżej 90% lub mediana degraduje się przez kilka ewaluacji. Checkpoint wybierany leksykograficznie: finish rate, mediana, najlepszy czas, segmentowy time debt.

Podaj dokładne komendy PowerShell/`uv run trackmaniarl ...` dla zaproponowanych testów, bazując na istniejącym CLI. Jeśli obecne CLI nie wystarcza, pokaż minimalne nowe subcommandy/flagę oraz konkretne zmiany w kodzie.

## Wymagany format odpowiedzi

Raport ma mieć strukturę:

1. **Executive summary** — maksymalnie 10 najważniejszych wniosków.
2. **Najbardziej prawdopodobny łańcuch przyczynowy** — od nagrania inputu do rozjazdu przy 24,8%.
3. **Udowodnione błędy w kodzie** — plik, funkcja, linie, dowód, wpływ, poprawka.
4. **Nierozstrzygnięte hipotezy** — dokładny eksperyment falsyfikujący każdą z nich.
5. **Audyt timestampów i causal alignment** — diagram czasu dla `frame[i]`, inputu, fizyki, odczytu pluginu i `SendInput`.
6. **Projekt trace/diagnostics v2** — schema i miejsca instrumentacji.
7. **Projekt stabilnego kontrolera trajektorii** — równania, pseudokod, parametry i testy znaków.
8. **Ocena klawiatura vs gamepad vs bezpośredni input**.
9. **Pipeline uczenia do 35–36 s** — tylko po rozwiązaniu wcześniejszych bramek.
10. **Eksperymenty w kolejności** — hipoteza, zmiana, komenda, metryki, sukces, odrzucenie.
11. **Minimalny plan patchy P0/P1/P2** — w kolejności wdrożenia.
12. **Lista rzeczy, których teraz nie robić** — aby nie wrócić do wielomiesięcznego strojenia na uszkodzonym kontrakcie.

## Standard dowodowy i źródła

- Opieraj wnioski o zachowanie systemu przede wszystkim na przesłanym kodzie.
- Dla semantyki OpenPlanet/TrackMania używaj dokumentacji pierwotnej, kodu pluginu lub wiarygodnych źródeł technicznych; zaznacz, czego nie da się potwierdzić publicznie.
- Nie cytuj ogólnych artykułów o autonomous driving jako dowodu na konkretną kolejność ticków TrackManii.
- Nie zakładaj, że dokładne `10 ms` w NPZ oznacza dokładne event timing klawiatury.
- Nie uznawaj open-loop finish za konieczny warunek trenowania, jeśli wykażesz, że real-time input injection jest niedeterministyczny; w takim przypadku zaproponuj lepszy miernik fidelity oraz kontrolę zamkniętą.
- Nie obiecuj 35 s. Podaj najbardziej prawdopodobną drogę, przedziały ryzyka oraz warunki, po których należy zmienić kierunek.
- Każda rekomendacja ma wskazywać konkretny element aktualnego codebase’u, nie abstrakcyjny „system RL”.

Najważniejsze pytanie końcowe brzmi:

> Jaki jest minimalny zestaw poprawek w kontrakcie czasu, instrumentacji i sterowaniu zamkniętym, który wykorzysta istniejącą demonstrację 35.855 s do uzyskania polityki kończącej mapę stabilnie w 35–36 sekund — oraz jak eksperymentalnie udowodnić poprawność każdego etapu przed rozpoczęciem kosztownego treningu?
