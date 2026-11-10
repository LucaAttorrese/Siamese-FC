# Piano: SiamFC in Python come sorgente di tracking per C-UASP (camera IDS)

Piano approvato il 2026-10-06 e aggiornato fino al 2026-11-06. Legenda: ✅ fatto · 🟡 in corso · ⏳ da fare.

## Contesto

`C-UASP/control/cuasp.py` riceve da `suncubes.camera.CameraAPI` una `CameraCorrection` (errore di puntamento az/po in rad) prodotta dalla pipeline
`IDS -> KLT -> ArUco async -> PnP -> target point -> frame laser -> angoli`. Questo richiede marker ArUco montati sul drone.
Obiettivo: sostituire la parte ArUco/PnP con il tracker **SiamFC**, lasciando invariati il contratto `CameraCorrection`, il supervisore PBVS e il dispatcher motori (`MotorCommandDispatcher`, `abs_rotate`, stop su perdita del target).

## Decisioni prese con l'utente

| Tema | Decisione |
|---|---|
| Camera | Resta la **IDS** (Mono8, 2592x1944 in `settings.py`, target 70 Hz, teleobiettivo). Niente OAK. |
| Calcolo | Tutto sull'host. Sviluppo: portatile con RTX 4060 (8 GB). **Training ed esercizio: portatile con RTX 5060 Laptop (8 GB)**, Ryzen 7 260, 31 GB di RAM. Sostituisce il PC con RTX 5090 previsto il 2026-11-04 (decisione del 2026-11-06). Stessa memoria video del portatile di sviluppo: batch 3 a 1280 px, training più lunghi che sulla 5090. |
| Python | **3.13 o successivo.** Su Windows, con 3.12, `time.monotonic()` avanza a scatti di 16 ms, più di un frame a 70 Hz (vedi Rischi). |
| Pesi SiamFC | L'originale `2016-08-17.net.mat` non è più scaricabile. Si usa **CFNet "Baseline-conv5"**, variante addestrata in grigio `baseline-conv5_gray_e100.mat` (Valmadre et al., CVPR 2017). La RGB `baseline-conv5_e55.mat` serve solo per i confronti. |
| Inizializzazione e recupero | **Detector YOLO** (Ultralytics YOLO11) addestrato su droni, asincrono a bassa frequenza, come oggi l'ArUco. |
| Licenza Ultralytics | AGPL-3.0, **accettata** perché il sistema serve per ora solo a validare l'approccio (2026-11-06). Per un prodotto non open source servirebbe una licenza Enterprise o un detector con licenza permissiva. |
| Ambiente operativo | **Al coperto e illuminato** (2026-11-09). Per ora non è previsto l'uso all'esterno di notte con luci urbane: il punto debole notturno del detector (Fase 2, punto 2) non è prioritario. |
| Bersaglio e distanza | **DJI Mavic Mini o simile** (circa 245 × 290 × 56 mm aperto con eliche) a **20–22 m** (2026-11-09). Dimensione attesa nel frame IDS, con fx ≈ 21 060 / 26 760 px: lato √(w·h) circa 130–160 px visto di lato, circa 270–340 px visto da sotto. Campo visivo orizzontale a 21 m: circa 2,6 / 2,0 m. |
| Distanza Z (parallasse camera→laser, `t_lc` ≈ −18 cm) | **Z fissa configurabile** in `settings.py`; valore proposto `CV_SIAMFC_TARGET_RANGE_M = 21` (centro dell'intervallo 20–22 m), con errore di parallasse massimo circa 0,43 mrad (circa 9–11 px) per Z vera tra 20 e 22 m. |
| Repository | **C-UASP non si modifica mai.** I file sono copiati in `Siamese-FC/cuasp_siamfc/` (provenienza e hash in `cuasp_siamfc/PROVENANCE.md`) e modificati solo lì. |
| Git | **Nessuna operazione git senza permesso esplicito dell'utente**: commit e push li fa lui. |

## Architettura (obiettivo)

```
ids-latest-frame (invariato)  ->  camera-api-worker: SiamFC.update() su ogni nuovo frame (GPU, 3 scale in batch)
                                    |  centro del bbox in px -> undistort -> raggio * Z_fissa -> R_lc, t_lc -> (d_az, d_po)
                                    |  confidenza = f(picco, APCE)  -> CorrectionCandidate
                                  drone-detector (executor a 1 thread): YOLO, asincrono
                                    - SEARCHING: scansione del frame intero a missing_hz -> init di SiamFC
                                    - TRACKING: verifica in una ROI attorno al bbox a redetect_hz -> re-init se c'è deriva
```
Macchina a stati dell'adapter: `SEARCHING -> TRACKING -> LOST (score basso per N frame o oltre max_lost_s) -> SEARCHING`.
In LOST/SEARCHING l'adapter emette `reliable=False` (reason `siamfc_searching` / `siamfc_lost`). `cuasp.py` invia già `stop[0]=1` sulla transizione da tracciato a perso.

## Fase 0 — Copia della baseline ✅

File C-UASP copiati in `cuasp_siamfc/` (commit `474450c`). Baseline dei test: tutto verde tranne
`test_multi_marker_fusion.py::test_settings_default_target_offsets_match_restored_rig_values`, che fallisce già in C-UASP (offset del marker 21 contro 18 cm).

## Fase 1 — Port SiamFC ✅ (`cuasp_siamfc/control/suncubes/siamfc/`)

- `convert_matconvnet.py`: DagNN MatConvNet → `.pth`. Layout `H×W×Cin×Cout` → `Cout×Cin×H×W`; conv2/4/5 con group 2; BN fusa nelle conv, con `moments = [mean, sigma]` (sigma è una deviazione standard); `fin_adjust_bn` scalare conservata.
- `net.py`: backbone con stride totale 4 e conv5 a 32 canali. Exemplar 127 → 17×17, instance 255 → 49×49, score 33×33. Un ingresso a un canale equivale esattamente a 3 canali replicati.
- `tracker.py`: port 1:1 di CFNet `tracker.m`, `tracker_step.m` e `make_scale_pyramid.m`. Include `s_x = 255/127·s_z`, la piramide di scala a due stadi, l'aggiornamento del template (`zLR`) e gli iperparametri della Tab. 2 del paper. Riproduce i dettagli del riferimento: crop centrato 1 px prima, `imresize` bicubica esatta di MATLAB (`imresize.py`), ricerca del picco in ordine column-major.
- Crop copiati sulla GPU in uint8 (`_crop_to_device`), con valori identici al riferimento NumPy.
- Risultati: V1, V2 e V5 nella tabella di validazione.

## Fase 2 — Detector YOLO 🟡

Fatto:
- dataset scaricati, con link, SHA256 e procedura di ricostruzione in `datasets/MANIFEST.md`: DUT-Anti-UAV (detection e tracking) e Anti-UAV300 (solo la parte visibile);
- dataset YOLO in grigio `datasets/yolo_uav_gray` (train 12699 / val 5709 / test 6480), costruito con `tools/prepare_yolo_dataset.py`;
- script di training `tools/train_detector.py` (YOLO11s, imgsz 1280, augmentation di colore disattivata, seed fisso), con stop pulito a fine epoca (`--request-stop`) e ripresa (`--resume`), verificati end-to-end (2026-11-06).

Da fare sul PC di training (RTX 5060 Laptop):
1. ✅ Primo training, run `yolo11s_uav_gray_1280_e40`: 40 epoche a 1280 px, batch 3, `--workers 2`, eseguito a tappe con stop e ripresa dal 2026-11-06 al 2026-11-09. Comando e avvertenze in `cuasp_siamfc/README.md` §7. Circa 10 minuti per epoca, costanti per tutto il run. `best.pt` = epoca 36 (val: mAP50 0,954, mAP50-95 0,649, P 0,947, R 0,907), SHA256 `3a7c1732…8040d5`. Nelle epoche 38–40 il val scende leggermente (mAP50-95 0,636 all'epoca 40), mentre il recall sale a 0,916. **Test complessivo** su `best.pt` (6480 immagini): **mAP50 0,949, mAP50-95 0,593, P 0,962, R 0,869**.
2. ✅ Valutazione sul test, separata per sorgente (prefissi `dut_*` e `auav_*`) e per dimensione del drone: `tools/eval_detector.py` (2026-11-09), con output in `runs/detector/yolo11s_uav_gray_1280_e40_test_breakdown/breakdown.json`.
   - Per sorgente (validazione di Ultralytics): DUT mAP50 0,941, mAP50-95 0,670, P 0,958, R 0,876; Anti-UAV300 mAP50 0,959, mAP50-95 0,546, P 0,937, R 0,925.
   - Recall a IoU 0,5 con soglia 0,25, per lato del box in px originali. DUT è uniforme: 0,90–0,95 in tutte le fasce, anche sotto i 32 px. **Anti-UAV300 cala sui box medio-piccoli: 16–32 px 0,17 (46 box), 32–64 px 0,76 (797 box)**, contro 0,93–0,99 sopra i 64 px. Il punto debole è il dominio Anti-UAV300 sotto i 64 px, non i droni piccoli in generale.
   - **Diagnosi (2026-11-09).** Dei 229 box Anti-UAV300 tra 16 e 64 px non rilevati con soglia 0,25:
     - **191 sono detezioni corrette ma con confidenza bassa** (IoU ≥ 0,5, confidenza mediana 0,11);
     - 18 sono box disallineati (0,1 ≤ IoU < 0,5);
     - 20 sono veri mancati.

     Il recall a IoU 0,3 è quasi uguale a quello a IoU 0,5, quindi l'imprecisione delle annotazioni non spiega il calo (solo pochi casi dubbi, come un box sopra il testo dell'overlay). Nella fascia 16–64 px, abbassando la soglia il recall passa da 0,73 (soglia 0,25) a 0,81 (0,15), 0,86 (0,10) e 0,90 (0,05).
   - **Causa.** I casi sono concentrati in 2 sessioni notturne (`20190925_193610`, `20190925_200805`): il drone è visibile solo come punti luminosi (i LED) su cielo scuro, e viene perso o rilevato con confidenza bassa quando si trova vicino a lampioni o strutture illuminate. Le sessioni notturne non sono sottorappresentate nel training (43 % delle immagini Anti-UAV300 di train contro il 19 % del test): è il caso "notte con luci urbane" a essere intrinsecamente difficile.
   - **Implicazioni.** L'ambiente operativo è al coperto e illuminato, quindi il caso notturno non è prioritario. In scene illuminate il detector è uniforme a tutte le dimensioni (DUT: recall 0,90–0,95). Resta valida per `detector.py` una soglia configurabile con conferma temporale. Lo script di diagnosi è stato eseguito una tantum (non è in repo).
   - **Divario di dominio.** I dataset di training sono tutti all'aperto, con droni su sfondo di cielo, mentre l'ambiente reale è al coperto: pareti, strutture, lampade e riflessi, con droni vicini e quindi grandi in pixel (con il teleobiettivo, fx ≈ 21 000–27 000 px, un drone di 35 cm a 10–20 m occupa circa 370–950 px). Le prestazioni misurate sul test non garantiscono quelle al coperto: i **frame IDS registrati ed etichettati nell'ambiente reale (punto 4) diventano la priorità** per valutare il detector e, se serve, per il fine-tuning.
   - Falsi allarmi sulle immagini senza drone (276, tutte Anti-UAV300): 14 con soglia 0,25, 6 con soglia 0,517.
3. ✅ (2026-11-10) `control/suncubes/siamfc/detector.py`, con test in `tests/test_siamfc_detector.py` (31 test, uno con i pesi veri, saltato se i pesi mancano). Contiene:
   - `Detection` e il protocollo `DroneDetector`;
   - `UltralyticsDroneDetector`: frame intero o ROI, ridimensionamento del singolo canale Mono8 con `INTER_AREA` prima della replica su 3 canali, verifica SHA256 dei pesi, fp16 tramite `quantize=16`;
   - `DetectionConfirmer`: conferma dopo N detezioni consecutive coerenti, misurate come spostamento del centro relativo alla dimensione del box, rapporto tra le dimensioni e intervallo di tempo massimo.

   Coordinate in convenzione pixel-centro (come SiamFC e `cv2.undistortPoints`), convertite da quelle a bordi di Ultralytics. Latenza sul PC di training con frame da 2592x1944: **29 ms** sul frame intero (erano 55 ms replicando prima i canali sul frame da 5 MP) e 22 ms su una ROI 800x800; circa 8 ms sono overhead di `predict` di Ultralytics, eliminabile chiamando il modello direttamente se V5 mostrasse contesa sulla GPU con SiamFC.

   Specifica originale: interfaccia `DroneDetector` (Protocol) `detect(gray, roi_xywh | None) -> list[Detection(bbox_xywh, score)]`, sostituibile e simulabile nei test. Implementazione Ultralytics in PyTorch (niente ONNX per ora). Con il bersaglio previsto (Mavic Mini a 20–22 m, lato circa 130–340 px nel frame da 2592 px, quindi circa 65–170 px a imgsz 1280) basta l'inferenza sul frame intero ridimensionato. L'inferenza a tasselli (tile 640 con sovrapposizione, ispirata a `MultiMarkerTiledPreprocessor` / `_next_scout_roi` in `camera.py`) non serve, salvo bersagli molto più lontani o piccoli. Soglia di confidenza configurabile e conferma su N frame prima dell'inizializzazione di SiamFC (vedi diagnosi al punto 2).
4. **Prioritario** (vedi divario di dominio al punto 2): frame IDS registrati ed etichettati nell'ambiente operativo al coperto, prima come test set del detector e di SiamFC, poi eventualmente per il fine-tuning.

## Fase 3 — Adapter camera ✅ (`control/suncubes/siamfc_source.py`, 2026-11-10)

**Fatto.** `siamfc_source.py` con tre parti testabili separatamente:
- `pixel_to_laser_angles`: geometria con Z fissa;
- `SiamFCTrackingSession`: macchina a stati senza thread;
- `IdsSiamFCCorrectionSourceAdapter`: producer IDS, thread `drone-detector`, gating e debug.

Routing `ids_siamfc` in `camera.py::_build_source` (unica modifica a `camera.py`, più il commento sui modi in `CameraConfig`). Test in `tests/test_siamfc_source_adapter.py` (37 test, solo componenti simulati): un pixel sull'asse laser dà angoli nulli anche con rotazione, braccio di leva e distorsione; tutte le transizioni di stato; adapter end-to-end; routing e validazione della configurazione. Suite completa: 209 passati + 1 fallimento noto.

Scelte rispetto alla specifica sotto, motivate da prove con i modelli veri:
- **Confidenza**: media geometrica di `picco/picco_rif` e `APCE/APCE_rif`, ciascuno limitato a [0, 1]. I riferimenti sono le mediane dei primi 5 frame dopo ogni (re)inizializzazione, non il solo primo frame, che è gonfiato. Il picco grezzo può essere negativo fuori dal bersaglio. Sulla demo: circa 0,7 sul bersaglio, circa 0,07 fuori.
- **Detection ritardate**: il tracker si inizializza sul frame su cui ha girato il detector e poi si aggiorna sul frame corrente.
- **Verifica sul frame intero**, non su una ROI ritagliata. Una ROI di circa 300 px ingrandita a 1280 produceva falsi positivi enormi; reinizializzare su di essi faceva crescere box e ROI fino a seguire lo sfondo con confidenza alta. Ora la "ROI" è un raggio di ricerca, con controllo della dimensione (rapporto ≤ 1,5) e reinizializzazione solo dopo 2 verifiche concordi (`reinit_confirm`).
- **Perdita non recuperabile dal tracker** se causata da verifica fallita o bersaglio fuori dal frame: in quel caso riparte solo un'acquisizione confermata. Il recupero autonomo resta solo per i cali di confidenza.
- **Filtri anti-falsi positivi**: `acquire_min_score` 0,5 (da tarare in V4) e, nel detector, box scartati se più grandi di 0,3 del lato del frame (`max_box_frac`) o più piccoli di 4 px.
- **Verifica a 5 Hz con 2 mancate** (`redetect_hz`, `verify_miss_limit`): un tracker scivolato sullo sfondo resta "affidabile" al massimo circa 0,4 s, contro 1,5 s con 2 Hz e 3 mancate.

**Riproduzione con modelli veri** (SiamFC + YOLO, frame a 50 fps in tempo reale, sequenze DUT-Anti-UAV tracking). Frame affidabili sul drone giusto, su quelli con drone presente:

| Video | Drone | Corretti | Bersaglio sbagliato | Drone assente |
|---|---|---|---|---|
| video01 | ~95 × 41 px | 81 % | 20 | 0 |
| video05 | ~51 × 37 px | 91 % | 0 | 0 |
| video10 | ~112 × 44 px | 94 % | 12 | 22 |
| video15 | ~58 × 47 px | 73–76 % | 0–46 | 6–56 |

Aggancio in 0,24–0,5 s. `read_candidate`: mediana circa 7,4 ms, p99 15,6–17,5 ms con il detector attivo, quindi oltre i 14,3 ms del budget a 70 Hz ma entro i 20 ms a 50 fps (da approfondire in V5). Il ritardo residuo, circa 0,4 s di frame sbagliati per ogni deriva, è il tempo della rete di sicurezza.

**Strumento di riproduzione** `tools/replay_siamfc_source.py`. Fa girare l'intera sorgente `ids_siamfc` su una sequenza registrata (cartella di immagini o video, ground truth opzionale, calibrazione vera opzionale) a ritmo reale, e produce:
- le metriche di sicurezza: frame affidabili corretti, sul bersaglio sbagliato e con drone assente, segmento pericoloso più lungo;
- la copertura, l'errore del centro e i tempi di `read_candidate`;
- un CSV per frame e un riepilogo JSON.

L'opzione `--empty-scene` misura i falsi allarmi di una ripresa senza drone. Sarà lo strumento di V4 sulle registrazioni IDS.

**Limite noto**: un falso positivo *persistente* del detector è indistinguibile, per la logica di tracking, da un drone in volo stazionario. In video15 un oggetto fisso con score 0,39–0,57 viene agganciato quando il drone è fuori inquadratura, a seconda che lo score superi 0,5. Si risolve al livello del detector: frame IDS dell'ambiente reale (anche a scena vuota, per misurarne i falsi positivi), soglia tarata su quei dati ed eventualmente zone da ignorare configurabili.

Specifica originale (2026-10-06):

Nuova classe `IdsSiamFCCorrectionSourceAdapter` che implementa `CorrectionSourcePort` (`start/stop/read_candidate/get_debug_snapshot/fast_stats`, `mode="ids_siamfc"`). **Riusa** da `suncubes/camera.py`, senza duplicarli:
- `_IdsLatestFrameProducer`, `_load_vs_calibration_for_camera`, `_scale_intrinsics_to_runtime`;
- `_point_c_to_laser`, `_angles_from_point_laser`, `_laser_axis_point_in_camera`, `_project_camera_point_to_pixel`;
- `_clone_debug_payload`, `_setting`, `GatingConfig`.

Geometria: centro del bbox (più un offset px opzionale) → `cv2.undistortPoints` con K scalata al runtime → `p_C = [x_n·Z, y_n·Z, Z]` con `Z = CV_SIAMFC_TARGET_RANGE_M` → `p_L = R_lc p_C + t_lc` → angoli. Il `z_m` riportato è la Z fissa e il debug segnala `z_mode=fixed`. `PbvsZResolver` in modalità "fixed" non va usato così com'è, perché richiede uno `z_raw` misurato valido.

Confidenza: `peak_t / peak_init` combinato con APCE, con soglie in `settings.py` tarate in V4. Vedi anche la proposta P4 più sotto.

Debug snapshot: stesse chiavi già lette da `cuasp._draw_debug_frame` (`frame_gray`, `principal_point_px`, `laser_axis_pixel_px`, `laser_axis_range_m/_source`, `pbvs_target_pixel_px`, `markers=[]`, `expected_marker_ids=()`, `tile_state`, `tracking_backend`, `target_point=None`) più le nuove `siamfc_bbox_xywh`, `siamfc_score`, `tracker_state`, `detections`.

Thread: SiamFC nel `camera-api-worker`, YOLO solo nel thread `drone-detector`; i risultati passano tramite `Future`, come per `_detect_job` dell'ArUco. Nessuno stato mutabile condiviso nuovo.

Routing: in `camera.py::_build_source` aggiungere il ramo `cfg.mode == "ids_siamfc"`, con import lazy. È l'unica modifica a `camera.py`.

## Fase 4 — Integrazione in `cuasp.py` e `settings.py` ⏳

- `settings.py`:
  - `CV_TARGET_TRACKER = "aruco" | "siamfc"`;
  - `CV_SIAMFC_WEIGHTS_PATH`, `CV_SIAMFC_WEIGHTS_SHA256`;
  - `CV_SIAMFC_PARAMS` (`paper` oppure il JSON tarato in V4b);
  - `CV_SIAMFC_DEVICE`, `CV_SIAMFC_TARGET_RANGE_M`, `CV_SIAMFC_CONF_MIN`, `CV_SIAMFC_LOST_FRAMES`, `CV_SIAMFC_MAX_LOST_S`;
  - `CV_DETECTOR_*`: path dei pesi, sha, score minimo, tile, redetect_hz, missing_hz, IoU di re-init;
  - estendere `settings.validate()`.
- `cuasp.py`:
  - `_build_camera_config()` sceglie `mode="ids_siamfc"` e passa i dict `siamfc`/`detector`;
  - `_validate_configuration()` salta i controlli ArUco in modalità siamfc;
  - il banner di avvio riporta gli hash dei pesi;
  - `_draw_debug_frame` disegna il bbox SiamFC e le detezioni;
  - il messaggio `[MOTOR] marker lost` diventa `target lost`;
  - **nessuna modifica** a PBVS, segni, clamp e dispatcher.
- `pyproject.toml`: dipendenze opzionali aggiornate (torch cu130 sulle GPU Blackwell, ultralytics).
- Documentazione: README e note sulla nuova pipeline, i thread e il limite della Z fissa.

## Fase 5 — Validazione (livelli di evidenza separati)

| Livello | Cosa | Criterio | Stato |
|---|---|---|---|
| V1 conversione | Rete fusa contro semantica MatConvNet non fusa (DagNN sintetico); grigio uguale a RGB replicato | Errore entro la precisione float32 | ✅ Manca un confronto tensoriale con MatConvNet vero: non c'è MATLAB |
| V2 sequenza demo | `tools/eval_sequence.py` su `demo-sequences/vot15_bag` | Nessuna deriva | ✅ IoU 0.796, errore medio 5.9 px. Non indipendente: `bag` è nel validation set su cui il paper ha tarato gli iperparametri (vedi P2) |
| V3 benchmark | OTB-2013/OTB-100 (toolkit `got10k`), rete RGB su RGB e rete grigia su OTB convertito in grigio | Vedi P1 | ⏳ |
| V4 dominio drone | DUT tracking, Anti-UAV300 test e sequenze IDS registrate (registratore Mono8 con `ts_capture_ns`) | Success/precision, errore del centro in px **e in mrad** (÷fx), tempo di riaggancio, falsi riagganci | ⏳ |
| V4b ricerca iperparametri | Random search, procedura del paper App. F (sezione sotto) | I parametri tarati devono battere quelli della Tab. 2 sul test held-out, altrimenti restano quelli del paper | ⏳ Da fare sul PC di training |
| V5 latenza | `tools/bench_siamfc.py` su frame 2592x1944 | Entro il budget del frame (14.3 ms a 70 Hz, 20 ms a 50 fps) con il detector attivo | 🟡 4060: 6.9–8.9 ms medi fino a target di 400 px. 5060 Laptop (PC di training): 5.9–8.1 ms medi, p99 ≤ 9.2 ms. Manca la misura con il detector attivo sulla stessa GPU |
| V6 pytest | Suite in `cuasp_siamfc/tests` | Tutto verde; senza torch i test SiamFC vengono saltati, non falliscono | 🟡 Verde, a parte il test ArUco già rotto in C-UASP |
| V7 banco, motori spenti | IDS dal vivo con `CV_CUAS_ENABLE_MOTOR_COMMANDS=0`. Drone **con ArUco montati**: la pipeline ArUco registrata fa da riferimento offline per SiamFC sugli stessi frame | Differenza angolare entro un budget da concordare; segni az/po corretti | ⏳ |
| V8 banco, motori accesi | Gain ridotto, poi nominale; prova di perdita e riaggancio → `stop[0]=1` | Nessuna oscillazione nuova, stop su perdita verificato | ⏳ |

### V4b — Ricerca degli iperparametri del tracker

**Perché.** Il paper CFNet (App. F) mostra che gli iperparametri del tracker incidono molto. Li sceglie con una random search di 300 combinazioni su un validation set separato dal test. I valori della Tab. 2 sono tarati su sequenze generiche (VOT/Temple-Color) con la rete RGB; il nostro dominio è diverso: droni piccoli su cielo, teleobiettivo, grigio, 50–70 fps.

**Cosa si cerca.** Solo i parametri dell'algoritmo, senza riaddestrare la rete. Campionamento uniforme intorno ai valori del paper:

| Parametro | Paper (Tab. 2) | Intervallo |
|---|---|---|
| `scale_step` | 1.0470 | 1.02 – 1.07 |
| `scale_penalty` | 0.9825 | 0.960 – 0.995 |
| `scale_lr` | 0.68 | 0.40 – 0.80 |
| `w_influence` | 0.175 | 0.10 – 0.35 |
| `z_lr` | 0.0102 | 0.000 – 0.020 |

`num_scale` = 3 e `response_up` = 8 restano fissi.

**Dati.**
- Taratura: Anti-UAV300 train + val (227 video, in grigio).
- Test held-out: Anti-UAV300 test (91 video) e DUT tracking (20 video), usati una sola volta.
- Conferma: sequenze IDS.
- Attenzione: Anti-UAV300 è a 20 fps contro i 50–70 della IDS. I parametri vanno riconfermati sulle sequenze IDS.

**Protocollo (paper §4.1).**
- 3 partenze equispaziate per sequenza.
- Quando l'IoU scende a 0 il tracker si ferma e i frame restanti contano 0.
- I frame con `exist = 0` sono esclusi dall'IoU.
- Metrica principale: overlap medio (AUC). Metriche secondarie: precisione a 20 px, errore in mrad, numero di perdite.
- Inizializzazione dal groundtruth.

**Strategia.**
1. 300 configurazioni su circa 40 video, una partenza. Sulla 5060 Laptop la stima è di circa 20 ore con un processo (circa 7 ms per frame, decodifica inclusa); più processi in parallelo possono aiutare perché SiamFC non satura la GPU. Il passo 2 aggiunge circa 8 ore. Stime da confermare con il run pilota.
2. Le 10 migliori sul set completo, con 3 partenze.
3. Prima di tutto, un run pilota di 5 configurazioni per misurare i tempi.

**Implementazione.**
- `tools/tune_hyperparams.py`: seed fisso; CSV aggiunto riga per riga, così il run è riprendibile; output `best_params.json` con parametri, metriche, seed, hash dei pesi e lista dei video.
- Metriche in un modulo condiviso con `eval_sequence.py`, con i relativi test.
- `CV_SIAMFC_PARAMS`: il default resta `paper` finché V4b non è superata.

**Dipendenze.** Dopo V3; prima della taratura delle soglie di perdita e confidenza.

### Test previsti

- ✅ `test_siamfc_geometry.py`, `test_siamfc_convert.py`, `test_siamfc_tracker_synthetic.py`, `test_siamfc_crop_device.py`.
- ⏳ `test_siamfc_source_adapter.py`: producer, tracker e detector simulati per verificare la macchina a stati SEARCHING/TRACKING/LOST, `reliable` e reason; un pixel sull'asse laser proiettato deve dare `d_az ≈ d_po ≈ 0` a Z fissa.
- ⏳ Estensioni di `test_camera_config_routing.py` (`ids_siamfc` → nuovo adapter) e di `test_settings_validate.py`.
- ⏳ E2E: `CameraAPI` siamfc simulata → `cuasp.main()` → `FakePCMMServer`.

## Rischi e punti aperti

- **Focale ambigua** (ereditato da C-UASP): i YAML di calibrazione sono a 2040x1536 (fx ≈ 21 060 px), `settings.py` acquisisce a 2592x1944 e `camera.py` riscala K (fx ≈ 26 760 px). Il commento nel YAML ammette che la risoluzione reale non è chiara. Va risolto prima di V7: incide del 27 % su pixel e angoli.
- **Esposizione e frame rate**: `CV_IDS_EXPOSURE_TIME_US = 20000` con target di 70 Hz. 20 ms di esposizione limitano la camera a circa 50 fps; verificare sul banco.
- **Moto relativo e gimbal**: SiamFC tollera uno spostamento di circa la dimensione del bersaglio per frame, cioè circa L/D rad per frame (drone di 35 cm a 50 fps: circa 10°/s a 100 m, circa 2°/s a 500 m). Un `abs_rotate` ampio, per esempio al riaggancio con il clamp a 30°, fa uscire il target dalla regione di ricerca. Vedi P3.
- **Grigio contro RGB**: mitigato dalla rete addestrata in grigio. La perdita va misurata in V3/V4.
- **Target piccoli e grandi**: sotto circa 16 px (indicativo; un drone di 35 cm ci arriva a circa 460–585 m) degradano tracker e detector. Sopra circa 400 px la latenza cresce, problema mitigato dalla copia in uint8.
- **Z fissa**: errore di parallasse ≈ |t_lc|·|1/Z_vera − 1/Z_fissa| rad. Va documentato e loggato; serve un hook per una futura Z da sensore esterno.
- **Memoria in training su Windows**: i worker del dataloader impegnano circa 1.7 GB di memoria virtuale ciascuno e la validazione ne usa il doppio. Usare `--workers 2`.
- **Risoluzione dell'orologio su Windows** (trovato il 2026-11-06): con Python 3.12 `time.monotonic()` usa `GetTickCount64()` e avanza a scatti di 16 ms, più di un frame a 70 Hz (14.3 ms). C-UASP usa `monotonic_ns` per età e freschezza dei campioni, e `test_read_candidate_reuses_cached_result_for_unchanged_sequence` fallisce a intermittenza. Da Python 3.13 l'orologio è `QueryPerformanceCounter()` (0.1 µs). Ogni macchina che esegue il runtime deve usare Python ≥ 3.13.
- **Sicurezza**: nessun cambio a limiti, KAS o dispatcher. Le prove al banco partono sempre con i motori disabilitati.
- **Piano B, fine-tuning di SiamFC**: se V4/V4b non bastano, fine-tuning dei pesi sui video di droni (ricetta in App. A del paper: SGD, batch 8, coppie a distanza massima di 100 frame, loss logistica). Richiede un training PyTorch da scrivere: in repo ci sono solo il training MATLAB della rete del 2016 e un convertitore che fonde le BN.

## Proposte dalla revisione del paper CFNet (2026-11-06), non ancora approvate

L'utente ha approvato solo V4b. Le seguenti restano proposte: chiedere prima di applicarle.

- **P1 — Criterio V3**: usare i valori di Baseline-conv5 (Tab. 1) invece di SiamFC-3s: OTB-2013 61.8 OPE / 64.0 TRE, OTB-100 58.8 OPE / 61.6 TRE. Per la rete grigia il paper non ha valori di riferimento: si misura lo scarto rispetto alla RGB.
- **P2 — V2 non indipendente**: segnalarlo esplicitamente nel README.
- **P3 — Compensazione del moto del gimbal come requisito dell'adapter**: predizione dello spostamento dagli encoder oppure ripartenza del detector durante gli spostamenti ampi.
- **P4 — Confidenza `sigmoid(picco)`**: la rete è addestrata con loss logistica e lo score dopo `fin_adjust_bn` ne è l'ingresso, quindi `sigmoid(picco)` più APCE è una misura più fondata del rapporto con il picco iniziale, che deriva con l'aggiornamento del template.
- Nota: il template update va lasciato attivo. La Fig. 4 del paper mostra che aiuta sempre; `zLR = 0` è solo un dato di latenza.

## Ordine di esecuzione (sul PC di training, RTX 5060 Laptop)

1. ✅ Setup secondo `cuasp_siamfc/README.md` (venv cu130 con Python 3.13, pesi, test, `bench_siamfc.py`) e registrazione della latenza V5 (2026-11-06).
2. ✅ Ricostruzione dei dataset (`datasets/MANIFEST.md`) il 2026-11-06: hash e conteggi uguali al MANIFEST.
3. 🟡 Training del detector YOLO11s ✅ (completato il 2026-11-09, test complessivo mAP50 0,949 / mAP50-95 0,593). Valutazione per sorgente e per dimensione ✅ (vedi Fase 2, punto 2).
4. `detector.py` + adapter `ids_siamfc` + routing, settings e `cuasp.py` (Fasi 3–4) → V6 pytest + E2E.
5. V3 (OTB), poi V4b (ricerca iperparametri), poi V4 (dati drone e IDS) e taratura delle soglie.
6. V7, poi V8 al banco.
