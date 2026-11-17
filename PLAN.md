# Piano: SiamFC in Python come sorgente di tracking per C-UASP (camera IDS)

Piano approvato il 2026-10-06 e aggiornato fino al 2026-11-06. Legenda: ✅ fatto · 🟡 in corso · ⏳ da fare.

## Contesto

`C-UASP/control/cuasp.py` riceve da `suncubes.camera.CameraAPI` una `CameraCorrection` (errore di puntamento az/po in rad) prodotta dalla pipeline
`IDS -> KLT -> ArUco async -> PnP -> target point -> frame laser -> angoli`. Questo richiede marker ArUco montati sul drone.
Obiettivo: sostituire la parte ArUco/PnP con il tracker **SiamFC**, lasciando invariati il contratto `CameraCorrection`, il supervisore PBVS e il dispatcher motori (`MotorCommandDispatcher`, `abs_rotate`, stop su perdita del target).

## Decisioni prese con l'utente

| Tema | Decisione |
|---|---|
| Camera | Resta la **IDS**: U3-33DxXCP-M, Sony IMX900 global shutter, 2064x1552, pixel da 2,25 µm, teleobiettivo, Mono8. Niente OAK. Dalla prova del 2026-11-16: frame **2040x1536** (ROI di `UserSet1`; il 2592x1944 ereditato veniva rifiutato, ora `settings.py` chiede 2040x1536). Con esposizione a 15 ms e banda della camera a 300 MB/s: **62,6 fps**. |
| Calcolo | Tutto sull'host. Sviluppo: portatile con RTX 4060 (8 GB). **Training ed esercizio: portatile con RTX 5060 Laptop (8 GB)**, Ryzen 7 260, 31 GB di RAM. Sostituisce il PC con RTX 5090 previsto il 2026-11-04 (decisione del 2026-11-06). Stessa memoria video del portatile di sviluppo: batch 3 a 1280 px, training più lunghi che sulla 5090. |
| Python | **3.13 o successivo.** Su Windows, con 3.12, `time.monotonic()` avanza a scatti di 16 ms, più di un frame a 70 Hz (vedi Rischi). |
| Pesi SiamFC | L'originale `2016-08-17.net.mat` non è più scaricabile. Si usa **CFNet "Baseline-conv5"**, variante addestrata in grigio `baseline-conv5_gray_e100.mat` (Valmadre et al., CVPR 2017). La RGB `baseline-conv5_e55.mat` serve solo per i confronti. |
| Inizializzazione e recupero | **Detector YOLO** (Ultralytics YOLO11) addestrato su droni, asincrono a bassa frequenza, come oggi l'ArUco. |
| Licenza Ultralytics | AGPL-3.0, **accettata** perché il sistema serve per ora solo a validare l'approccio (2026-11-06). Per un prodotto non open source servirebbe una licenza Enterprise o un detector con licenza permissiva. |
| Ambiente operativo | **Al coperto e illuminato** (2026-11-09). Per ora non è previsto l'uso all'esterno di notte con luci urbane: il punto debole notturno del detector (Fase 2, punto 2) non è prioritario. |
| Bersaglio e distanza | **DJI Mavic Mini o simile** (circa 245 × 290 × 56 mm aperto con eliche) a **20–22 m** (2026-11-09). Dimensione attesa nel frame IDS da 2040x1536, con fx ≈ 21 060 / 24 120 px (le due calibrazioni): lato √(w·h) circa 130–145 px visto di lato, circa 265–305 px visto da sotto. Campo visivo a 21 m: circa 2,0 / 1,8 m in larghezza e 1,5 / 1,3 m in altezza. |
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

## Fase 4 — Integrazione in `cuasp.py` e `settings.py` ✅ (2026-11-10 / 11)

**Fatto.**
- **`settings.py`**:
  - `CV_TARGET_TRACKER = "siamfc"` di default (decisione del 2026-11-10);
  - tutti i parametri `CV_SIAMFC_*` e `CV_DETECTOR_*`: pesi con SHA256, Z fissa 21 m, `CV_SIAMFC_CONF_MIN` 0,15 come soglia di gating dedicata (quella ArUco, 0,0005, è tarata sull'area del marker), macchina a stati, filtri del detector;
  - controlli di tipo e intervallo in `validate()`;
  - **`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0`** di default (era 1 in C-UASP; decisione del 2026-11-10).
- **`cuasp.py`**:
  - `_build_camera_config()` sceglie `ids_siamfc` e passa i dict `siamfc`/`detector`, con i percorsi risolti rispetto a `cuasp_siamfc/`;
  - `_validate_configuration()` salta i controlli ArUco e valida la configurazione completa della sorgente, inclusi i controlli incrociati;
  - nel banner: modalità, Z fissa, percorsi dei pesi e SHA256 effettivamente caricati;
  - nel debug: box SiamFC colorato per stato, detezioni, riga di stato al posto di "PBVS target camera: N/A", laser "fixed @21 m";
  - messaggio `target lost`;
  - **nessuna modifica** a PBVS, segni, clamp e dispatcher.
- `pyproject.toml`: extra `siamfc` con torchvision e ultralytics.
- Test: routing e configurazione SiamFC, controllo che ogni parametro `CV_SIAMFC_*`/`CV_DETECTOR_*` arrivi alla sorgente, validazione, frame di debug, nuovi controlli di `settings.validate()`. Test end-to-end `tests/test_cuasp_siamfc_e2e.py`: sorgente SiamFC con componenti simulati → `CameraAPI` → `cuasp.main()` → dispatcher → `FakePCMMServer`, con comandi `abs[0,1]` di segno e valore attesi e stop su perdita.
- Avvio reale della sorgente dai settings con i pesi veri: percorsi risolti e SHA256 corrispondenti.

**Risolto (2026-11-11): corsa nel dispatcher ereditato da C-UASP.**
- **Problema** (trovato dal test end-to-end, 2 esecuzioni su 10): quando la perdita arrivava mentre un `abs_rotate` era già in corso, lo stop partiva da un task separato (`asyncio.create_task(self.motors.stop)` in `MotorCommandDispatcher.submit`). Se vinceva la corsa sul lock TCP, sul cavo passava `stop[0]=1` seguito dall'`abs[0,1]` già in volo, calcolato dall'ultimo campione valido, e lo stop veniva annullato. Valeva anche in modalità ArUco.
- **Decisione dell'utente**: opzione A, cioè modificare il dispatcher della copia `cuasp_siamfc` e documentare la differenza dal funzionamento originale.
- **Modifica**:
  - `submit()` segna solo `_stop_pending`;
  - il worker `_run()` invia lo stop (`_send_stop_on_loss()`) all'inizio del giro successivo, prima di ogni altro invio, quindi sempre dopo un comando già in volo;
  - `_send()` scarta il comando se la perdita arriva durante la lettura degli encoder;
  - uno stop fallito viene registrato come un invio fallito.
- **Verifica**:
  - ordine riprodotto in modo deterministico: originale `enc, abs_begin, stop, abs_end`, ora `enc, abs_begin, abs_end, stop`;
  - nuovo `tests/test_motor_dispatcher_stop_order.py` (6 test);
  - i test originali del dispatcher passano invariati;
  - test end-to-end 30 esecuzioni su 30;
  - suite completa 251 passati + 1 fallimento noto.
- **Documentazione**: `cuasp_siamfc/PROVENANCE.md` ("Local changes to the copied files", tra i cambiamenti di comportamento rispetto a C-UASP) e README ("Behaviour changes with respect to C-UASP"). In C-UASP la corsa resta, perché quel repository non si modifica.

Specifica originale:

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
| V5 latenza | `tools/bench_siamfc.py` su frame 2592x1944 | Entro il budget del frame con il detector attivo: 16,0 ms a 62,6 fps (camera dal 2026-11-16) | 🟡 **Al limite** (2026-11-16, 5060 Laptop, frame 2040x1536 a 62,6 fps):<br>• SiamFC da solo: 6,0–7,8 ms medi, p99 ≤ 8,6;<br>• sorgente completa con il detector: mediana 6,3 ms, p95 12,4, **p99 17,6**, max 33.<br>Il 2,3 % dei frame sfora i 16 ms, mai due di fila, a circa 5 Hz: è la contesa con la verifica del detector. Nessun effetto pratico: comandi motore a 50 Hz al massimo, campioni validi 250 ms. Rimedio noto, non applicato: chiamare YOLO senza `predict` di Ultralytics. Dettagli nel README |
| V6 pytest | Suite in `cuasp_siamfc/tests` | Tutto verde; senza torch i test SiamFC vengono saltati, non falliscono | 🟡 Verde, a parte il test ArUco già rotto in C-UASP (2026-11-11: 251 passati + 1; e2e `cuasp.main()` → server motori simulato stabile, 30/30) |
| V7 banco, motori spenti | IDS dal vivo con `CV_CUAS_ENABLE_MOTOR_COMMANDS=0`. Drone **con ArUco montati**: la pipeline ArUco registrata fa da riferimento offline per SiamFC sugli stessi frame | Differenza angolare entro un budget da concordare; segni az/po corretti | ⏳ |
| V8 banco, motori accesi | Gain ridotto, poi nominale; prova di perdita e riaggancio → `stop[0]=1`. **Promemoria (richiesto dall'utente il 2026-11-10): i comandi motore sono disabilitati di default (`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0`); vanno abilitati esplicitamente solo per questa prova, ricordandolo all'utente prima di iniziare.** La corsa stop/`abs` del dispatcher è risolta (Fase 4, 2026-11-11): in V8 verificare sul banco che dopo lo stop su perdita non arrivino altri comandi | Nessuna oscillazione nuova, stop su perdita verificato | ⏳ |

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

- **Focale** (ereditato da C-UASP). L'ambiguità del 27 % è risolta dalla prova del 2026-11-16: la camera consegna 2040x1536, la stessa risoluzione delle calibrazioni, e il runtime riscala K con le dimensioni reali, quindi con fattore 1. Decisione dell'utente (2026-11-16): si tiene l'override `calibration_legacy_20260921_x_test.yaml` (fx 24 119, cx 1140) e non il più recente `calibration.yaml` (fx 21 062, cx 697), accettando le incoerenze:
  - con pixel da 2,25 µm le fx corrispondono a 47–54 mm di focale, mentre i YAML indicano un obiettivo da 16 mm;
  - la calibrazione vale solo con il ROI 2040x1536 a offset (8, 8) salvato in `UserSet1`, perché il runtime non imposta gli offset.

  V7 confronterà comunque gli angoli con il riferimento ArUco.
- **Esposizione e frame rate** (misurati il 2026-11-16). All'inizio 43,2 fps, limitati da `DeviceLinkThroughputLimit` = 162 MB/s salvato nella camera. Con 300 MB/s la camera concede 47,7 fps a 20 ms, 62,6 a 15 ms, 67,9 a 12 ms o meno; i 70 Hz non sono raggiungibili con questo ROI.

  **Decisione dell'utente (2026-11-16): 15 ms**, con il runtime che imposta la banda a 300 MB/s (`CV_IDS_LINK_THROUGHPUT_LIMIT_BPS`). Verificato: 62,6 fps, 0 perdite. Le immagini sono il 25 % più scure che a 20 ms; a 21 m il mosso è di circa 15–17 px per ogni m/s di moto laterale del drone.
- **Unità del guadagno**: corretto il 2026-11-16 (`camera.py`, `_set_gain_db()`). Il valore in dB viene convertito nel fattore lineare della camera (1,0–15,85); un valore fuori range fa fallire l'avvio della camera. Per tornare alla luminosità dei 20 ms bastano +2,5 dB.
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

## Proposta P5 — Normalizzazione della luminosità prima del detector (2026-11-17), da valutare in seguito

Non approvata: decisione dell'utente rimandata. Per ora il rimedio è il guadagno della camera a +6 dB (`CV_IDS_GAIN_DB`, 2026-11-17). Le registrazioni restano valide in ogni caso, perché la normalizzazione si applica a valle e si valuta offline su qualsiasi registrazione.

**Motivazione** (punto 4b, blocco B): a 0 dB, con cielo nuvoloso, il detector perde il drone scuro davanti a sfondi scuri. Il guadagno fisso funziona con questa luce, ma con il sole schiarirebbe molto la scena. La normalizzazione invece si adatta alla luce.

**Progetto:**
- Dove: solo sull'immagine che va al detector, nel thread del detector (`UltralyticsDroneDetector.detect()`).
  - SiamFC continua a ricevere il frame originale: una luminosità che cambia durante il tracking non corrisponderebbe più al template preso all'aggancio.
  - Geometria, freschezza, clamp e stop su perdita restano invariati.
- Cosa: guadagno digitale globale per frame, applicato con una tabella di 256 valori (`cv2.LUT`, circa 1 ms su 2040x1536).
  - g = minimo fra: livello obiettivo / media (obiettivo: la luminosità tipica del dataset di training, da misurare); 250 / p99,5, per non saturare; un massimo fisso, ad esempio ×4.
  - Statistiche calcolate su una griglia sottocampionata.
  - **g ≥ 1**: l'immagine non viene mai scurita, quindi con scena chiara il comportamento resta quello attuale.
- Senza memoria tra frame, quindi il replay è riproducibile.
- Nuove impostazioni `CV_DETECTOR_NORMALIZE` (spenta di default finché non è validata), `CV_DETECTOR_NORM_TARGET`, `CV_DETECTOR_NORM_MAX_GAIN`, con controlli in `validate()`.
- Guadagno applicato registrato nel debug e nei CSV del replay; test unitari; documentazione in `PROVENANCE.md` e README.
- Limite: il guadagno globale aiuta poco se il frame è quasi tutto chiaro con una zona molto scura dietro il drone. Variante locale (CLAHE) solo come seconda opzione: altera l'immagine in modi non visti in training e amplifica il rumore.
- Si può combinare con un guadagno della camera moderato, e con un nuovo training del detector che includa frame IDS etichettati (soluzione di fondo al divario di dominio).

**Validazione** (quasi tutta offline, con un'opzione di `replay_siamfc_source.py`):
1. Scena vuota, tutti i 12 spezzoni del blocco A. Requisito: come oggi, zero agganci e zero frame affidabili. Caso critico: A10 a luci spente (riceve il guadagno più alto).
2. Blocco B a 0 dB, tutti gli spezzoni:
   - gli sfondi scuri (B2 r2/r4, B3 r2, B6 r2–r4) devono avvicinarsi ai risultati a +6 dB;
   - gli spezzoni già buoni (B1, B2 r1/r3, B6 r1, B7) non devono peggiorare;
   - controllo visivo dei casi dubbi, per escludere falsi bersagli.
3. Spezzoni a +6 dB con la normalizzazione: guadagno digitale vicino a 1, nessun effetto negativo.
4. Non regressione sui video DUT con le annotazioni (`--gt`): i bersagli sbagliati non devono aumentare e la copertura non deve calare.
5. Latenza (come in V5): `read_candidate` invariato; detector circa +1–2 ms.
6. Dal vivo in V7, al banco con motori spenti, con il guadagno applicato visibile nel debug.

Per numeri solidi nei tratti senza tracking servono etichette su un campione di frame, pre-generate e corrette dall'utente (già previste per V4). Stima: 1–2 ore di implementazione e test, 1–1,5 ore di calcolo in background.

## Ordine di esecuzione (sul PC di training, RTX 5060 Laptop)

1. ✅ Setup secondo `cuasp_siamfc/README.md` (venv cu130 con Python 3.13, pesi, test, `bench_siamfc.py`) e registrazione della latenza V5 (2026-11-06).
2. ✅ Ricostruzione dei dataset (`datasets/MANIFEST.md`) il 2026-11-06: hash e conteggi uguali al MANIFEST.
3. ✅ Training del detector YOLO11s (completato il 2026-11-09, test complessivo mAP50 0,949 / mAP50-95 0,593). Valutazione per sorgente e per dimensione ✅ (vedi Fase 2, punto 2).
4. ✅ `detector.py` + adapter `ids_siamfc` + routing, settings e `cuasp.py` (Fasi 3–4) → V6 pytest + E2E (2026-11-10 / 11; corsa stop/`abs` del dispatcher risolta).
4b. ⏳ Registrazione di frame IDS nell'ambiente reale, al coperto, anche a scena vuota, e analisi con `tools/replay_siamfc_source.py` (V4, falsi positivi, taratura delle soglie); V5 con detector e SiamFC insieme. Pronti (2026-11-11): il protocollo `RECORDING_PROTOCOL.md` (blocchi A scena vuota, B drone a 20–22 m, C con marker ArUco, D camera in movimento; circa 35–40 min di riprese, 100–180 GB) e il registratore `cuasp_siamfc/tools/record_ids.py` (ogni frame con timestamp, impostazioni del runtime; provato in modalità sintetica a 5 MP: 0 perdite a 50 e a 70 fps).

   **Prima prova con la camera (2026-11-16):**
   - Mono8 corretto, 0 perdite, 43,2 fps, frame 2040x1536, circa 0,3–0,5 MB per frame in JPEG;
   - immagini scure sulla scena di prova (livello medio 26–84 su 255);
   - registratore portato alla v2: dimensione reale, livelli di grigio, rilettura dei nodi della camera e controlli in `meta.json`.

   Decisioni dell'utente dello stesso giorno, applicate e verificate con la camera:
   - risoluzione richiesta 2040x1536;
   - guadagno in unità esplicite;
   - esposizione a 15 ms con banda a 300 MB/s, quindi 62,6 fps;
   - calibrazione: resta l'override `_x_test`.

   La camera è pronta per la sessione di registrazione.

   **Blocco A, scena vuota** (registrato e analizzato il 2026-11-16):
   - 12 spezzoni da 60 s (10 zone di parete, porta e pavimento; A10 a luci spente), 45 084 frame;
   - riprodotti in tempo reale a 62,6 fps con le impostazioni del runtime (`runs/v4_blockA/`);
   - **nessun frame affidabile e nessun aggancio**: nessun falso allarme sarebbe arrivato ai motori;
   - box grezzi del detector: **zero in 11 spezzoni su 12** (537 rilevazioni ciascuno, nessun box nemmeno sopra 0,15);
   - in A10 a luci spente ci sono 100 box deboli su due piccole macchie chiare della parete, sempre negli stessi punti. Score mediano 0,18, massimo 0,33 contro gli 0,5 dell'aggancio; dimensione 38–57 × 19–27 px, molto sotto i 130–300 px attesi per il drone a 21 m. Con le luci accese la stessa zona (A8) dà zero box.
   - Il limite noto della Fase 3 (falso positivo persistente agganciato) non si presenta in questo ambiente.
   - Idea da valutare dopo il blocco B, non applicata: una dimensione minima dei box legata alla distanza di esercizio (oggi `CV_DETECTOR_MIN_BOX_PX` = 4) eliminerebbe anche questi box.

   **Blocco B, primo controllo** (2026-11-16, `20261116_145326_B1_stazionario_a_terra`, drone **fermo a terra** a 21 m, sfondo pavimento e parete grigia):
   - detector sul drone in 290 rilevazioni su 290, score 0,70–0,75;
   - aggancio dopo 0,27 s, poi tracking al 100 %;
   - centro del tracker a 7,5 px da quello del detector (mediana), 26 px al massimo, con una reinizializzazione dalla verifica;
   - box 229 × 76 px (lato medio 132 px, come stimato).

   Lo spezzone breve `20261116_145249_B1_...` (un oggetto lasciato nell'inquadratura) va escluso dalle analisi.

   **Blocco B, B1 in volo stazionario** (2026-11-16, ore 16:29 e 16:31, `20261116_162916_B1_stazionario_in_volo` e `..._163148_..._r2`):
   - registrati nel tardo pomeriggio, nuvoloso e quasi buio, con le luci interne accese: livello medio 30–35 (51 alle 14:53, 82 a mezzogiorno), p99 51–54. La luce delle finestre pesa molto, quindi di sera la scena è più scura;
   - 62,61 fps, 0 perdite;
   - entrambi agganciati dopo 0,26 s, poi tracking al 100 % per 60 s, confidenza del tracker ≥ 0,88;
   - detector: score 0,70–0,76 sul drone in tutte le 290 rilevazioni, come alla luce del giorno. **Il buio di questa sera non pesa sul detector, guadagno non necessario.** Nessun box lontano dal drone; i pochi box in più (score 0,15–0,24) sono doppioni sovrapposti al drone;
   - box del tracker 293 × 91 px (lato medio 162 px); deriva del drone in volo stazionario fino a 179 px (circa 16 cm);
   - spezzone 1: centro del tracker a 3,8 px dal detector (mediana), 14 px al massimo;
   - spezzone r2: il box del tracker **scivola lentamente verso la parte alta del drone** (da 6 a circa 30 px in 35 s; IoU con il detector fino a 0,50). La verifica lo reinizializza al frame 2357 (regola: IoU < `reinit_iou` 0,5 per 2 verifiche di fila), poi la distanza torna a 3 px. Il tracker è rimasto sempre sul drone; 30 px sono circa 1,2 mrad, cioè 2,6 cm a 21 m (controllato sulle immagini).
   - Per la taratura V4, non applicato: il box del drone è largo e basso (circa 3:1), quindi l'IoU è molto sensibile agli spostamenti verticali. `reinit_iou` più alto (ad esempio 0,7) correggerebbe la deriva prima. Va misurato sugli spezzoni con il drone in movimento, dove anche il detector è meno preciso.
   - Terzo spezzone stazionario (`..._163946_B1_stazionario_in_volo_r3`, sfondo con porta): tracking al 100 %, detector 0,58–0,78.

   **Blocco B, B2 movimenti lenti** (2026-11-16, 16:43–16:50, 4 spezzoni `..._B2_lento_in_volo_r1..r4`; fuori quasi buio, luci accese, livello medio 25–35). Gli spezzoni registrati di sera sono la condizione peggiore, non quella d'uso.
   - Sfondo chiaro o alternato (r1, r3): **tracking al 100 %** con il drone che attraversa quasi tutto il frame (x 240–1868, y 71–1398), velocità fino a circa 560 px/s (circa 0,5 m/s a 21 m).
   - Sfondo scuro (r2: fascia grigia scura in basso; r4: porta scura): **tracking all'81 % (r2) e al 40 % (r4)**, con 2 e 6 perdite.
   - Causa delle perdite, controllata sulle immagini: il drone è scuro, quindi davanti a uno sfondo scuro il contrasto cade. Il detector lo vede ancora, ma con score 0,2–0,45, a volte nessun box.
     - Due verifiche di fila senza box portano alla perdita (comportamento voluto).
     - Il riaggancio richiede 3 rilevazioni ≥ 0,5, quindi arriva solo dopo 3–11 s.
   - Il tracker invece segue il drone: confidenza 0,5–0,9 al momento delle perdite.
   - **Nessun falso bersaglio**: tutti i box del detector in queste fasi sono sul drone.
   - Prova con il solo detector sui frame delle perdite, schiariti digitalmente (simulano il guadagno):

     | Frame | Score ≥ 0,5 originale | ×2 (+6 dB) | ×3 (+9,5 dB) |
     |---|---|---|---|
     | r2, fascia scura | 13–17 % | 100 % | 100 % |
     | r4, porta | 0–8 % | 66–76 % | 79–94 % |
     | r3, controllo | 100 % | 100 % | 100 % |

     - I box sono sul drone (8 px di distanza mediana dai box deboli originali).
     - Sulla scena vuota del blocco A, schiarita: nessun box ≥ 0,5 tranne 1 su 60 frame in A10 a luci spente con ×2 (53 px, molto sotto la dimensione del drone); nessuno con ×3.
     - **Il detector soffre le immagini scure, più che il contrasto in sé.**
   - Rimedi possibili, nessuno applicato:
     - guadagno della camera più alto quando la scena è scura;
     - normalizzazione della luminosità prima del detector (cambiamento del codice);
     - frame IDS etichettati nel training del detector (Fase 2, punto 4);
     - in taratura, `acquire_min_score` più basso insieme a una dimensione minima dei box.
   - B3–B7 rimandati alla mattina del 2026-11-17, con la luce del giorno.

   **Blocco B, B3–B7** (2026-11-17, 09:26–10:36, cielo nuvoloso, luci della sala al massimo):
   - La luce è quasi quella di ieri sera: livello medio 24–44 a 0 dB. È una **condizione d'uso reale**, non un caso limite.
   - Tutte le registrazioni sono valide: 62,61 fps, 0 perdite. Risultati in `runs/v4_blockB/`.
   - L'utente ha incluso apposta gli sfondi difficili (porta, cambi di colore) anche in B3–B5.

   | Spezzone | Situazione | Tracking dopo il 1° aggancio |
   |---|---|---|
   | B3 r1 / r2 / r3 | rapidi (fino a circa 2 m/s), r2 con porta | 89 / 57 / 94 % |
   | B4 r1 / r2 | uscite e rientri, fuori qualche secondo | 56 / 50 % (perdite quasi tutte ai bordi, come previsto) |
   | B5 r1 / r2 | rotazioni | 81 / 92 % (perdite brevi, riaggancio in 0,3–1,5 s) |
   | B6 r1 | sfondo alternato grigio e bianco | 98 % |
   | B6 r2 | porta | 34 % (riagganci fino a 17 s) |
   | B6 r3 | alternato con simbolo dell'estintore | 69 % |
   | B6 r4 | angolo molto scuro | 41 % (riagganci fino a 21 s) |
   | B7 18,9 / 21,2 / 23,4 m | stazionario | 100 / 100 / 100 % |

   - **Guadagno +6 dB** (decisione dell'utente: 3 spezzoni B6 sugli stessi sfondi, poi guadagno riportato a 0 per B7):
     - livello medio 44–55, nessuna saturazione, fattore riletto 1,995;
     - stesso sfondo, 0 dB → +6 dB: angolo scuro 41 → **97 %**; alternato con estintore 69 → **93 %**; porta 34 → **87 %**;
     - i voli non sono identici, ma l'effetto è grande e coerente con la prova digitale di ieri.
     - **Il guadagno +6 dB risolve gran parte delle perdite su sfondo scuro con questa luce.**
   - Controllo visivo dei casi con box del detector lontano dal tracker (60–135 px) in B3, B4, B5, B6: **entrambi i box sono sempre sul drone, nessun falso bersaglio.** Cause:
     - il tracker in ritardo su un drone veloce e mosso;
     - un artefatto della misura: il box del detector arriva qualche frame dopo il frame su cui è calcolato;
     - il box allungato sul braccio durante le rotazioni;
     - davanti alla porta, a 0 dB, il box del tracker si allarga fino a 592 × 173 px restando centrato sul drone. Poi la verifica lo scarta per dimensione e il tracker va in perdita.
   - B7: box del detector 330, 280 e 251 px a 18,9, 21,2 e 23,4 m (larghezza × distanza costante entro il 6 %). Score 0,70–0,80 a 18,9 m, 0,50–0,73 a 23,4 m (93 su 98 ≥ 0,5): cala con la distanza. I 25 m non sono stati registrati.
     - In `meta.json` dello spezzone `20261117_103356_B7_distanze_23_m` la distanza è 21,2 per errore: quella vera è **23,4 m** (nelle note).

   **Scena vuota a +6 dB** (2026-11-17, A20–A23, 4 spezzoni da 60 s: parete alternata, porta, estintore, zona scura). Servono a verificare i falsi allarmi con il nuovo guadagno.
   - Livello medio 41–68, guadagno riletto 1,995, 0 perdite.
   - **Nessun aggancio e nessun frame affidabile**, come a 0 dB.
   - Box grezzi del detector: zero in 3 spezzoni su 4. In A20 ci sono 11 box deboli (score mediano 0,18, massimo 0,26) sempre nello stesso punto, al bordo inferiore del frame (circa 885, 1490). Sono piccoli, 38–61 × 29–35 px, come le macchie di A10. Risultati in `runs/v4_blockA/`.

   **Blocco C, marker ArUco** (2026-11-17, 11:33–11:38, +6 dB, parete bianca, livello medio circa 85). 4 spezzoni da 60 s:
   - `..._113301_C1_stazionario_aruco_r1` e `..._113427_..._r2`;
   - `..._113735_C1_lento_aruco_r1` e `..._113623_C1_lento_aruco_r2`. Sono i **C2** (movimenti lenti): il nome riporta C1 per errore; non vanno rinominati.

   Risultati:
   - **Tracking al 100 % in tutti e 4**. Detector ≥ 0,5 in tutte le 290 rilevazioni di ogni spezzone (0,56–0,76).
   - I marker ID 1 e 2 sono montati su un'asta orizzontale ai due lati del drone, quindi tracker e detector inquadrano drone, asta e marker insieme: box di circa 500 × 112 px invece di circa 290 × 90. L'aspetto del bersaglio è diverso da quello del blocco B.
   - ArUco con i parametri di C-UASP (un frame ogni 4):
     - stazionario: marker letti nel 100 % dei frame;
     - movimenti lenti: ID 1 nel 96–97 %, ID 2 nell'82–91 %;
     - lato del marker 51–52 px; distanza tra i centri dei due marker 390–405 px;
     - nessun ID inatteso.
   - **Anteprima di V7**: centro dei due marker (quando sono letti entrambi) meno centro del box SiamFC.
     - Stazionario: mediana +0,6/+4 px in x, +3,5/+6,5 px in y (0,03–0,27 mrad), dispersione p5–p95 entro ±10–23 px.
     - Movimenti lenti: mediana fino a 13–14 px (0,6 mrad, circa 1,2 cm a 21 m).
     - Se si legge un solo marker, il suo centro dista circa 200 px dal centro del bersaglio. V7 deve usare il punto di riferimento corretto: il punto medio dei marker o gli offset per marker di C-UASP.
   - Blocco D (camera in movimento) rimandato: per ora non è possibile. La sessione di registrazione si è chiusa il 2026-11-17.
   - Da considerare in V7: il lato misurato dei marker (51–52 px) è circa il 10 % sotto i 57 px attesi per 5 cm a 21,0 m con la calibrazione in uso. Può dipendere dalla distanza reale (in `--distance-m` è rimasto 21,0 in tutti e 4 gli spezzoni) o dall'inclinazione dei marker. Va tenuto presente nel confronto degli angoli.
5. V3 (OTB), poi V4b (ricerca iperparametri), poi V4 (dati drone e IDS) e taratura delle soglie.
6. V7, poi V8 al banco.
