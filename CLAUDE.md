# CLAUDE.md — Siamese-FC / cuasp_siamfc

Contesto per Claude Code su questa repository, in particolare sul PC di training ed esercizio: portatile con **RTX 5060 Laptop (8 GB)**,
Ryzen 7 260, 31 GB di RAM. Ha sostituito il PC con RTX 5090 previsto in origine (decisione del 2026-11-06).
Il piano dettagliato, con lo stato di ogni fase, è in [PLAN.md](PLAN.md). La procedura operativa per questo PC è in
[cuasp_siamfc/README.md](cuasp_siamfc/README.md) (sezione "Using this repository on the training PC").

## Regole di lavoro (decise dall'utente, vincolanti)

- **Rispondere all'utente in italiano.** Codice, commenti e README della cartella `cuasp_siamfc` sono in inglese.
- **Nessuna operazione git che modifichi lo stato** (commit, branch, checkout, push, reset, ...) **senza un permesso esplicito dato in quel momento.** Anche se un piano dice "commit", chiedere. Comandi git in sola lettura (`status`, `diff`, `log`) sono liberi. I commit e i push li fa l'utente.
- **La repository C-UASP non si modifica mai.** Se presente sulla macchina, si legge soltanto. Il lavoro avviene sulle copie in `cuasp_siamfc/` (provenienza e hash in `cuasp_siamfc/PROVENANCE.md`).
- **Sicurezza** (sistema di puntamento fisico, regole di C-UASP): non indebolire limiti, clamp, controlli di freschezza del campione o stop su perdita; nessuna conversione di unità implicita. Le prove dal vivo partono con `CV_CUAS_ENABLE_MOTOR_COMMANDS = 0`.
- **Motori disabilitati di default** (`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0` in `settings.py`, decisione del 2026-11-10). **Quando si arriva alla prova con motori abilitati (V8), ricordarlo all'utente prima di iniziare:** vanno abilitati esplicitamente e solo per quella prova.
- Prima di comandi lunghi o costosi (training, download di GB, ricerche di iperparametri) indicare durata e spazio previsti. I processi lunghi vanno in background, con un controllo che si fermi anche in caso di errore.
- Le proposte P1–P6 in `PLAN.md` **non sono approvate**: chiedere prima di applicarle.

## Cos'è questo progetto

Sostituire, nel sistema di puntamento C-UASP, il tracking basato su marker ArUco con il tracker visivo **SiamFC** sul drone,
più un detector **YOLO** per l'inizializzazione e il riaggancio. A valle non cambia nulla: `CameraCorrection` in rad → PBVS → dispatcher motori PCMM/KAS.

- La radice della repo è un fork del codice MATLAB originale di SiamFC (`tracking/`, `training/`, `util/`). Serve **solo come riferimento**, insieme ai sorgenti CFNet in `reference/cfnet/`.
- Tutto il codice Python è in `cuasp_siamfc/`.
- Camera: IDS **U3-33DxXCP-M** (Sony IMX900, global shutter, sensore 2064x1552, pixel 2,25 µm), Mono8, teleobiettivo. Dal 2026-11-16: **2040x1536, esposizione 15 ms, banda 300 MB/s → 62,6 fps**, verificato con la camera.
- Pesi SiamFC: **CFNet Baseline-conv5, variante in grigio** (`baseline-conv5_gray_e100`). L'originale del 2016 non è più scaricabile.
- Detector: Ultralytics YOLO11s. Licenza AGPL-3.0, accettata perché per ora è solo una validazione.
- Distanza: Z fissa configurabile per la parallasse camera→laser.
- Bersaglio: **DJI Mavic Mini o simile a 20–22 m** (circa 130–340 px di lato nel frame IDS; Z fissa proposta 21 m).
- Ambiente operativo: **al coperto e illuminato** (deciso il 2026-11-09). Il caso notturno all'aperto non è prioritario. I dataset di training sono all'aperto, quindi i frame IDS dell'ambiente reale sono la priorità per la validazione.

## Struttura

```
PLAN.md                         piano e stato delle fasi (leggere per primo)
RECORDING_PROTOCOL.md           protocollo delle riprese IDS per V4 (per chi registra)
cuasp_siamfc/
  README.md                     setup del PC di training, comandi, validazione, licenze
  PROVENANCE.md                 file copiati da C-UASP + SHA256
  control/cuasp.py              supervisore C-UASP (copia); SiamFC di default (CV_TARGET_TRACKER)
  control/suncubes/             camera.py, motors.py, settings.py (copie C-UASP, modificate per SiamFC)
  control/suncubes/siamfc_source.py  sorgente camera ids_siamfc: macchina a stati, geometria Z fissa, adapter
  control/suncubes/siamfc/      port SiamFC (convert_matconvnet, net, tracker, imresize) + detector.py (YOLO)
  tools/                        eval_sequence, bench_siamfc, prepare_yolo_dataset, train_detector,
                                eval_detector, replay_siamfc_source, record_ids,
                                prelabel_ids + review_labels + ids_labels (etichette V4)
  tests/                        test C-UASP copiati + test SiamFC, detector, adapter, e2e con cuasp.main()
  .venv/                        ambiente Python (gitignored)
pretrained/                     pesi CFNet .mat e .pth convertiti (gitignored)
datasets/                       dataset + MANIFEST.md (link, hash, ricostruzione; dati gitignored)
runs/                           output dei training e delle analisi (gitignored); runs/v4_labels: cache delle rilevazioni e backup delle etichette
labels/ids/                     etichette V4 delle registrazioni IDS (CSV + JSON, versionate: contengono il lavoro di revisione dell'utente)
reference/cfnet/                sorgenti MATLAB di CFNet usati come riferimento
demo-sequences/vot15_bag/       sequenza demo per i controlli di regressione
```

## Comandi frequenti (PowerShell, da `cuasp_siamfc/`)

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python tools\bench_siamfc.py --weights ..\pretrained\networks\baseline-conv5_gray_e100.pth
.\.venv\Scripts\python tools\eval_sequence.py ..\demo-sequences\vot15_bag --weights ..\pretrained\networks\baseline-conv5_gray_e100.pth --gray
.\.venv\Scripts\python tools\train_detector.py --data ..\datasets\yolo_uav_gray\data.yaml --imgsz 1280 --epochs 40 --workers 2 --name yolo11s_uav_gray_1280_e40
.\.venv\Scripts\python tools\train_detector.py --name yolo11s_uav_gray_1280_e40 --request-stop       # stop pulito a fine epoca
.\.venv\Scripts\python tools\train_detector.py --data ..\datasets\yolo_uav_gray\data.yaml --name yolo11s_uav_gray_1280_e40 --resume --workers 2
.\.venv\Scripts\python tools\eval_detector.py --weights ..\runs\detector\yolo11s_uav_gray_1280_e40\weights\best.pt --data ..\datasets\yolo_uav_gray\data.yaml
# sorgente ids_siamfc completa su una sequenza registrata (DUT o IDS); --empty-scene per i falsi allarmi
.\.venv\Scripts\python tools\replay_siamfc_source.py ..\datasets\DUT-Anti-UAV\tracking\Anti-UAV-Tracking-V0\video01 --gt ..\datasets\DUT-Anti-UAV\tracking\Anti-UAV-Tracking-V0GT\video01_gt.txt
```

Stop e ripresa del training: `--request-stop` crea `runs/detector/<name>/STOP`. Il training finisce l'epoca in corso, salva `last.pt` ed esce con codice 3. Non usare `trainer.stop` né la fine normale di Ultralytics per fermare un run da riprendere: `final_eval()` chiama `strip_optimizer` su `last.pt` e la ripresa diventa impossibile.

Riferimenti di regressione:
- pytest: tutto verde tranne `test_multi_marker_fusion.py::test_settings_default_target_offsets_match_restored_rig_values`, che fallisce già in C-UASP;
- `vot15_bag` con la rete grigia: IoU medio 0.796, errore medio del centro 5.9 px. Il tempo varia con la macchina, la precisione no.

## Insidie già incontrate

- **PyTorch su GPU Blackwell** (RTX 5060 Laptop di questo PC, come qualsiasi RTX 50xx): serve l'indice **cu130** (`sm_120`). La build cu126 si ferma a `sm_90` e qui non parte.
- **Python ≥ 3.13 su Windows**: con 3.12 `time.monotonic()` avanza a scatti di 16 ms, più di un frame a 70 Hz. C-UASP lo usa per età e freschezza dei campioni, e `test_read_candidate_reuses_cached_result_for_unchanged_sequence` fallisce a intermittenza. La venv va creata con `py -3.13 -m venv .venv`.
- **Script con worker su Windows**: qualsiasi script che lancia un training Ultralytics con `workers > 0` deve avere `if __name__ == "__main__":`, altrimenti i processi figli rieseguono lo script e il training si ferma con `RuntimeError` (`freeze_support`). `train_detector.py` è già a posto.
- **torchvision solo CPU**: installare torch e torchvision dallo stesso indice CUDA **prima** di `ultralytics`. Altrimenti pip prende torchvision da PyPI e NMS fallisce su CUDA.
- **OpenCV**: `ultralytics` installa anche `opencv-python` accanto a `opencv-contrib-python`. Controllare che `cv2.aruco` esista.
- **Training su Windows**: i worker del dataloader impegnano circa 1.7 GB di memoria virtuale ciascuno, e la validazione ne usa il doppio. Con 4 worker il portatile è andato in `WinError 1455` e la validazione si è bloccata senza errori. Usare **`--workers 2`**; la ripresa da `last.pt` si fa con `train_detector.py --resume`.
- **Google Drive**: i file grandi possono dare "Quota exceeded" (pagina HTML di circa 2 KB al posto dello zip). Controllare dimensione e SHA256 di ogni download (`datasets/MANIFEST.md`).
- **Fedeltà al riferimento MATLAB** (vedi README, "Known reference quirks"): crop centrati 1 px prima, `imresize` bicubica di MATLAB, ricerca del picco column-major. Sono riprodotti apposta: non "correggerli" senza misurarne l'effetto.
- **Impostazioni della camera rifiutate in silenzio**: `camera.py` (C-UASP) usa `_try_set_node`, che ignora gli errori. `settings.py` è solo una richiesta: verificare sempre in `camera.readback` del `meta.json` di `record_ids.py` (v2) o con una lettura in sola lettura dei nodi.
- **Risoluzione e focale** (2026-11-16). La camera parte da `UserSet1` con ROI 2040x1536 a offset (8, 8); il 2592x1944 ereditato veniva rifiutato, ora `settings.py` chiede 2040x1536. È la risoluzione delle calibrazioni: il runtime riscala K con fattore 1 e l'ambiguità del 27 % non si presenta.
  - **Calibrazione scelta dall'utente** (2026-11-16): resta l'override `calibration_legacy_20260921_x_test.yaml` (fx 24 119, cx 1140) e non il più recente `calibration.yaml` (fx 21 062, cx 697), con le incoerenze accettate: fx da 47–54 mm contro il "16mm" scritto nei YAML. Non riproporre il cambio.
  - La calibrazione vale solo con il ROI di `UserSet1` (il runtime non imposta gli offset): non salvare UserSet né cambiare il ROI con IDS Cockpit.
- **Frame rate**: il limite era il `DeviceLinkThroughputLimit` di 162 MB/s salvato nella camera (43,2 fps a qualsiasi esposizione). Ora il runtime lo porta a 300 MB/s (`CV_IDS_LINK_THROUGHPUT_LIMIT_BPS`). Massimi con 300 MB/s: 47,7 fps a 20 ms, 62,6 a 15 ms (scelto dall'utente), 67,9 a 12 ms o meno; **70 Hz non raggiungibile** con questo ROI. Il ciclo di `cuasp.py` gira a `CV_CUAS_LOOP_HZ` = 70, ma invia comandi motore solo con un frame nuovo (`_is_new_visual_sample`).
- **Unità del guadagno**: corretto il 2026-11-16 (`_set_gain_db()` in `camera.py`). Da dB a fattore lineare (1,0–15,85 su questa camera); fuori range l'avvio della camera fallisce. A 15 ms le immagini sono il 25 % più scure che a 20 ms: +2,5 dB le riportano al livello precedente.
- **Binding IDS** (2026-11-11): l'SDK IDS peak 26.06.1 non include wheel Python; i binding vengono da PyPI (`ids_peak==1.17.0`, `ids_peak_ipl==1.18.0`, extra `ids` in `pyproject.toml`), già installati nella venv. In `camera.py` (C-UASP) `_buffer_to_mono8_numpy()` chiama `ConvertTo("Mono8")` con una stringa, che `ids_peak_ipl` 1.18 rifiuta; l'errore viene ignorato e l'immagine usata così com'è. Va bene perché la camera consegna davvero Mono8 (`PixelFormat` riletto, immagini verificate il 2026-11-16).
- **Memoria virtuale esaurita da Esplora risorse** (2026-11-11): `explorer.exe`, aperto dal 7 novembre, aveva 101 GB di memoria privata. La memoria virtuale del sistema era esaurita (0,3 GB liberi su 127) e ogni nuovo processo Python falliva all'avvio con `OpenBLAS error: Memory allocation still failed`. Non è un errore del codice. Rimedio: Gestione attività → Esplora risorse → Riavvia (lo fa l'utente; non terminare processi di sistema). Probabile causa: miniature di cartelle con moltissime immagini (dataset, registrazioni).

## ▶ PUNTO DI RIPRESA (aggiornato al 2026-11-16)

**Fasi 0–4 completate.** Il codice è pronto per girare dal vivo con la camera IDS: `cuasp.py` in modalità SiamFC, motori disabilitati di default.

**Prima prova con la camera fatta (2026-11-16)**, con camera collegata a questo PC:
- immagini in `recordings/20261116_11*_prova_camera` e `..._prova_readback`;
- Mono8 corretto, 0 perdite, 43,2 fps, frame 2040x1536, 0,3–0,5 MB per frame;
- scena di prova (una parete) scura: livello medio 26–84 su 255, p99 35–113;
- dettagli su camera, focale, frame rate e guadagno nelle insidie qui sopra.

`record_ids.py` è passato alla v2: dimensione reale dei frame, livelli di grigio, rilettura dei nodi della camera dopo la registrazione, `checks` in `meta.json`. Test: 13 in `test_record_ids.py`.

**Decisioni dell'utente del 2026-11-16, applicate e verificate con la camera** (`recordings/20261116_113901_prova_15ms`: 62,61 fps, 0 perdite, rilettura 300 MB/s, 15 ms, guadagno 1,0, 2040x1536 a (8, 8)):
1. esposizione 15 ms e banda 300 MB/s: nuova impostazione `CV_IDS_LINK_THROUGHPUT_LIMIT_BPS`, applicata da `camera.py`;
2. `CV_IDS_WIDTH`/`HEIGHT` = 2040x1536;
3. guadagno: conversione esplicita da dB a fattore in `camera.py`;
4. calibrazione: resta l'override `_x_test`; in `cuasp.py` è cambiato solo un commento.

Sono documentate in `PROVENANCE.md` (cambiamenti di comportamento 3–5) e nel README. Test nuovi in `tests/test_ids_camera_settings.py` (17). Suite: 281 passati + 1 fallimento noto.

**Prova in sala fatta dall'utente (2026-11-16, 15 ms, guadagno 0 dB)** su `recordings/20261116_115630_prova_sala` e `..._115714_prova_sala`:
- luci spente: media 55, p1–p99 43–74;
- luci accese: media 82, p1–p99 60–117;
- in entrambe 62,6 fps, 0 perdite, nessun avviso. **Guadagno non necessario**; la condizione di esercizio è a luci accese.
- L'inquadratura è la stessa parete delle prove del mattino: chiesto all'utente se è lo sfondo reale di volo.

**Sessioni di registrazione avviate dall'utente il 2026-11-16.**
- **Blocco A (scena vuota) fatto e analizzato**: 12 spezzoni, nessun aggancio né frame affidabile. Box grezzi del detector a zero in 11 spezzoni; in A10 a luci spente box deboli (score ≤ 0,33, contro 0,5 per agganciare) su due macchie chiare della parete. Dettagli in PLAN, punto 4b; risultati in `runs/v4_blockA/`.
- Lo spezzone con oggetti in movimento non è stato fatto, perché l'utente dice che nel caso d'uso non ci saranno. È consigliato per sicurezza (una persona nell'inquadratura), facoltativo.
- **Non analizzare mentre l'utente registra**: sullo stesso portatile si rischiano perdite di frame.
- **Blocco B in corso.** Primo spezzone, B1 con il drone **fermo a terra**: detector 290/290 (score 0,70–0,75), aggancio in 0,27 s, tracking al 100 %. Dettagli in PLAN, punto 4b; risultati in `runs/v4_blockB/`.
  - **Escludere `recordings/20261116_145249_B1_stazionario_a_terra`** (spezzone breve con un oggetto nell'inquadratura, indicazione dell'utente). Non cancellarlo.
  - **B1 in volo stazionario fatto** (2 spezzoni, ore 16:29 e 16:31, tardo pomeriggio quasi buio con luci accese, livello medio 30–35): aggancio in 0,26 s, tracking al 100 %, detector 0,70–0,76 in tutte le rilevazioni. Il buio non pesa e il guadagno non serve.
  - In r2 il box del tracker scivola verso l'alto del drone fino a circa 30 px, prima che la verifica lo corregga (`reinit_iou` 0,5). Da valutare in V4 un `reinit_iou` più alto; non applicato.
  - **B2 fatto di sera** (fuori buio, condizione peggiore e non d'uso), 4 spezzoni:
    - con sfondo chiaro tracking al 100 %;
    - davanti alla fascia scura (r2) e alla porta scura (r4) 81 % e 40 %. Il detector dà score bassi sul drone scuro su sfondo scuro, le verifiche falliscono e il riaggancio tarda 3–11 s;
    - nessun falso bersaglio;
    - schiarendo i frame ×2 il detector torna sopra 0,5 nel 66–100 % dei casi. Dettagli e rimedi (non applicati) in PLAN, punto 4b.
  - **B3–B7 fatti il 2026-11-17 mattina.** Cielo nuvoloso e luci al massimo, livello medio 24–44: la scena scura è una condizione d'uso reale.
    - Sfondi chiari: tracking 89–100 %. Porta o angolo scuro a 0 dB: 34–41 %.
    - **3 spezzoni B6 a +6 dB** sugli stessi sfondi (`CV_IDS_GAIN_DB` portato a 6 con permesso, poi **riportato a 0.0**, file uguale al commit): 87–97 %. Il guadagno risolve gran parte del problema.
    - Nessun falso bersaglio, verificato sulle immagini.
    - B7 al 100 % a 18,9, 21,2 e 23,4 m. Il `meta.json` dello spezzone a 23 m riporta 21,2 per errore (la distanza vera è nelle note). Dettagli in PLAN, punto 4b.
    - **Decisione dell'utente (2026-11-17): `CV_IDS_GAIN_DB = 6.0`** per i blocchi successivi e come impostazione attuale (luce d'uso prevista simile al 16–17 novembre). Aggiornati test, `PROVENANCE.md` (cambiamento 5) e README.
    - La normalizzazione della luminosità prima del detector è salvata come **proposta P5** in PLAN, da valutare in seguito; non è approvata.
    - **Scena vuota a +6 dB (A20–A23) fatta:** nessun aggancio. Box grezzi solo in A20: 11 deboli (≤ 0,26, 38–61 px) al bordo inferiore.
  - **Blocco C fatto** (4 spezzoni, +6 dB; i due "C1_lento" sono in realtà C2):
    - tracking al 100 %; marker letti nell'82–100 % dei frame;
    - centro dei marker contro centro del box SiamFC: 0,03–0,6 mrad di mediana;
    - i marker su asta laterale allargano il bersaglio a circa 500 px. Dettagli in PLAN, punto 4b.
  - Blocco D (facoltativo) rimandato: per ora l'utente non può farlo. **Sessione di registrazione chiusa il 2026-11-17.**
  - **V4 anticipato** rispetto a V3 e V4b (decisione dell'utente del 2026-11-17).
    - **Fase 1 (etichette) completata il 2026-11-18**: 29 registrazioni in `labels/ids/`, rivedute dall'utente. Non rigenerarle mai (`--force`): contengono la revisione. Backup in `runs/v4_labels/backups/`.
    - Convenzione per il drone in parte fuori dal frame in PLAN, "Ordine di esecuzione" punto 5.
    - `review_labels.py` ignora il Bloc Maiusc dal 2026-11-18: prima, con il Bloc Maiusc attivo, funzionava solo lo spazio.
    - **Fase 2 (misura di partenza) fatta il 2026-11-18**: `runs/v4_baseline/`, riepilogo con `tools/v4_report.py ../runs/v4_baseline --empty ../runs/v4_blockA`.
      - Copertura 85,3 %, errore 0,40 mrad di mediana, 0 falsi allarmi sulla scena vuota.
      - Frame persi: quasi tutti per score del detector sotto 0,5 prima dell'aggancio.
      - Episodi pericolosi: tre riprese spontanee da "lost" con confidenza appena sopra 0,25, proposta P6.
    - **Fase 3 avviata il 2026-11-18 con P6** (approvata dall'utente per implementazione e misura).
      - Implementata con valori che non cambiano il comportamento: `CV_SIAMFC_RECOVER_CONF = 0.25`, `CV_SIAMFC_RECOVER_EDGE_PX = None`. Suite: 320 passati + 1 fallimento noto.
      - **Prossimo passo: misurare le 4 varianti** (comandi e piano in PLAN, "Proposta P6"; circa 2–2,5 ore in background, niente registrazioni nel frattempo). Poi far scegliere i valori all'utente.
      - Dopo P6: la soglia di aggancio (`CV_SIAMFC_ACQUIRE_MIN_SCORE`, `CV_DETECTOR_CONFIRM_FRAMES`, da riprovare anche sulla scena vuota); facoltativo, nuovo training del detector con i frame IDS etichettati.
    - Il guadagno si cambia solo da `CV_IDS_GAIN_DB` in `settings.py`: `record_ids.py` non ha un'opzione per il guadagno.
  - Analisi del blocco B: `tools/replay_siamfc_source.py <rec>/frames --calibration calibration\calibration_legacy_20260921_x_test.yaml --fps 62.6 --out ..\runs\v4_blockB\<nome>.csv` (da `cuasp_siamfc/`). Gli script di controllo (accordo tracker–detector, dettaglio delle perdite, provini con i box, prova del guadagno simulato) erano nella cartella temporanea della sessione: vanno riscritti se servono.

**Prossimo passo: registrare frame IDS nell'ambiente reale e analizzarli (V4)** secondo `RECORDING_PROTOCOL.md`: blocco A scena vuota (priorità massima), B drone a 20–22 m, C con marker ArUco, D camera in movimento. Circa 35–40 minuti di riprese, 45–75 GB.

Quando l'utente ha le registrazioni (cartelle `recordings/<data_ora>_<nome>/`, gitignored):
1. analizzare con `tools/replay_siamfc_source.py` su `recordings/.../frames`: `--calibration` con la calibrazione vera, `--empty-scene` per il blocco A, `--gt` se esistono etichette (da pre-generare con detector e tracker e far correggere all'utente);
2. misurare il divario di dominio (dataset all'aperto contro uso al coperto) e i falsi positivi persistenti della scena, il limite noto della Fase 3;
3. tarare `CV_SIAMFC_ACQUIRE_MIN_SCORE`, le soglie di perdita e la confidenza; se servono, aggiungere i frame etichettati al dataset del detector (PLAN Fase 2, punto 4).

**Stato git:** tutto committato fino a `c975782` "Fix: fps" (impostazioni della camera, registratore v2). **Non ancora committati** (V5, 2026-11-16), da far committare all'utente: `tools/bench_siamfc.py` (`--frame-size`), `tools/replay_siamfc_source.py` (`--resize`, `--preload`, box grezzi del detector in `*_detections.csv`), `tests/test_replay_tool.py`, `cuasp_siamfc/README.md`, `PLAN.md` e questo file.

**Ultima modifica fatta (2026-11-11), opzione A scelta dall'utente:** in `MotorCommandDispatcher` lo stop su perdita viene inviato dal worker `_run()` invece che da un task separato, quindi arriva sempre dopo un `abs_rotate` già in volo. Un comando la cui lettura encoder era in corso al momento della perdita viene scartato. È documentato come cambiamento di comportamento rispetto a C-UASP in `cuasp_siamfc/PROVENANCE.md` e nel README ("Behaviour changes with respect to C-UASP"). In C-UASP la corsa resta, perché quel repository non si modifica. Test end-to-end 30/30, suite 251 passati + 1 fallimento noto.

**Poi, in ordine** (dettagli in `PLAN.md`, "Ordine di esecuzione"):
1. ✅/🟡 V5 fatto il 2026-11-16 (risultati in `runs/v5_2040x1536/`, README, PLAN). A 2040x1536 e 62,6 fps:
   - `read_candidate` ha mediana 6,3 ms e p99 17,6 ms contro un budget di 16 ms;
   - il 2,3 % dei frame sfora, mai due di fila, a circa 5 Hz: è la contesa con la verifica del detector;
   - effetto pratico nullo (comandi motore ≤ 50 Hz). Il rimedio (YOLO senza `predict`) non è applicato: è una proposta.

   Notato nella stessa prova: riprodurre i video DUT più velocemente peggiora il tracking su video15 (copertura 77 % a 50 fps contro 60 % a 62,6 fps, alla risoluzione nativa). È un artefatto: equivale a un drone più veloce rispetto alla logica a tempo del detector. Da verificare sulle registrazioni IDS vere. Per la taratura V4: `CV_SIAMFC_LOST_FRAMES` e `CV_SIAMFC_REF_FRAMES` sono in frame, quindi a 62,6 fps valgono 80 ms invece di 100.
2. V3 (OTB) → V4b (ricerca iperparametri, circa 28 ore su questo PC) → V4 → taratura delle soglie.
3. V7 confronterà gli angoli con il riferimento ArUco, usando la calibrazione scelta (l'override `_x_test`).
4. V7 (banco, motori spenti) → V8 (banco, motori accesi: **ricordare all'utente che i motori sono disabilitati di default** e vanno abilitati esplicitamente; verificare sul banco che dopo lo stop su perdita non arrivino altri comandi).

## Storico dello stato

Sul PC di training (dettagli in `PLAN.md`):
1. ✅ Setup: venv cu130 con Python 3.13, pesi, test, benchmark; latenza V5 di SiamFC nel README (2026-11-06).
2. ✅ Dataset ricostruiti con `datasets/MANIFEST.md` (2026-11-06; hash e conteggi verificati, zip conservati).
3. ✅ Training di YOLO11s, run `yolo11s_uav_gray_1280_e40` (2026-11-06 → 09, 40 epoche, circa 10 minuti l'una):
   - pesi in `runs/detector/yolo11s_uav_gray_1280_e40/weights/best.pt` (epoca 36, SHA256 `3a7c173244899df1ac5185916cbb8196bd0959b3d5626acc9f9cc3976d8040d5`); log in `runs/yolo11s_uav_gray_1280_e40.log`;
   - test complessivo: mAP50 0,949, mAP50-95 0,593, P 0,962, R 0,869;
   - punto debole su Anti-UAV300 sotto i 64 px, dovuto a scene notturne con luci urbane (confidenza mediana 0,11). Non prioritario, perché l'ambiente operativo è al coperto (PLAN, Fase 2, punto 2).
4. ✅ Fase 3 (2026-11-10):
   - `siamfc/detector.py`: 29 ms per frame da 2592x1944, filtri sulla dimensione dei box, conferma temporale;
   - `siamfc_source.py`: sorgente `ids_siamfc` con macchina a stati, verifica sul frame intero, perdita non recuperabile dal tracker se causata dal detector o dal bersaglio uscito dal frame;
   - routing in `camera.py` e `tools/replay_siamfc_source.py`.

   Riproduzione su DUT tracking: drone giusto nel 73–94 % dei frame con drone presente. Limite noto: un falso positivo persistente del detector è indistinguibile da un drone in volo stazionario.
5. ✅ Fase 4 (2026-11-10 / 11):
   - `settings.py`: `CV_TARGET_TRACKER = "siamfc"`, parametri `CV_SIAMFC_*`/`CV_DETECTOR_*`, motori a 0;
   - `cuasp.py`: configurazione, validazione, banner, debug; stop su perdita sequenziato dal worker del dispatcher (corsa ereditata da C-UASP corretta, documentata in `PROVENANCE.md`);
   - test: suite 251 passati + 1 fallimento noto; e2e `cuasp.main()` → server motori simulato 30/30.
