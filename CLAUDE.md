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
- Prima di comandi lunghi o costosi (training, download di GB, ricerche di iperparametri) indicare durata e spazio previsti. I processi lunghi vanno in background, con un controllo che si fermi anche in caso di errore.
- Le proposte P1–P4 in `PLAN.md` **non sono approvate**: chiedere prima di applicarle.

## Cos'è questo progetto

Sostituire, nel sistema di puntamento C-UASP, il tracking basato su marker ArUco con il tracker visivo **SiamFC** sul drone,
più un detector **YOLO** per l'inizializzazione e il riaggancio. A valle non cambia nulla: `CameraCorrection` in rad → PBVS → dispatcher motori PCMM/KAS.

- La radice della repo è un fork del codice MATLAB originale di SiamFC (`tracking/`, `training/`, `util/`). Serve **solo come riferimento**, insieme ai sorgenti CFNet in `reference/cfnet/`.
- Tutto il codice Python è in `cuasp_siamfc/`.
- Camera: IDS Mono8, 2592x1944 in `settings.py`, teleobiettivo.
- Pesi SiamFC: **CFNet Baseline-conv5, variante in grigio** (`baseline-conv5_gray_e100`). L'originale del 2016 non è più scaricabile.
- Detector: Ultralytics YOLO11s. Licenza AGPL-3.0, accettata perché per ora è solo una validazione.
- Distanza: Z fissa configurabile per la parallasse camera→laser.

## Struttura

```
PLAN.md                         piano e stato delle fasi (leggere per primo)
cuasp_siamfc/
  README.md                     setup del PC di training, comandi, validazione, licenze
  PROVENANCE.md                 file copiati da C-UASP + SHA256
  control/cuasp.py              supervisore C-UASP (copia; ancora pipeline ArUco)
  control/suncubes/             camera.py, motors.py, settings.py (copie C-UASP)
  control/suncubes/siamfc/      port SiamFC: convert_matconvnet, net, tracker, imresize
  tools/                        eval_sequence, bench_siamfc, prepare_yolo_dataset, train_detector
  tests/                        test C-UASP copiati + test SiamFC
  .venv/                        ambiente Python (gitignored)
pretrained/                     pesi CFNet .mat e .pth convertiti (gitignored)
datasets/                       dataset + MANIFEST.md (link, hash, ricostruzione; dati gitignored)
runs/                           output dei training (gitignored)
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
- **Focale ambigua** (punto aperto ereditato da C-UASP): calibrazione a 2040x1536 contro acquisizione a 2592x1944, con fx ≈ 21 060 oppure ≈ 26 760 px. Va chiarito prima della validazione angolare.
- **Esposizione di 20 ms contro target di 70 Hz**: la camera probabilmente non supera circa 50 fps. Verificare sul banco.

## Stato al 2026-11-06 e prossimi passi

Fatto: copia della baseline C-UASP; port SiamFC validato offline (con copia dei crop in uint8: 6.9–8.9 ms per frame sulla 4060 e 5.9–8.1 ms sulla 5060 Laptop, fino a target di 400 px); dataset del detector pronti sul portatile; script di training pronto.

Sul PC di training, in ordine (dettagli in `PLAN.md`, sezione "Ordine di esecuzione"):
1. ✅ Setup secondo il README (venv cu130 con Python 3.13, pesi, test, benchmark) e latenza V5 registrata nel README (2026-11-06).
2. ✅ Dataset: ricostruiti con `datasets/MANIFEST.md` il 2026-11-06 (hash e conteggi verificati; zip conservati).
3. 🟡 Training di YOLO11s ✅: run `yolo11s_uav_gray_1280_e40` completato il 2026-11-09 (40 epoche, circa 10 minuti l'una). Pesi in `runs/detector/yolo11s_uav_gray_1280_e40/weights/best.pt` (epoca 36, SHA256 `3a7c173244899df1ac5185916cbb8196bd0959b3d5626acc9f9cc3976d8040d5`); log cumulativo in `runs/yolo11s_uav_gray_1280_e40.log`. Test complessivo: mAP50 0,949, mAP50-95 0,593, P 0,962, R 0,869. La suddivisione per sorgente e per dimensione (`tools/eval_detector.py`, risultati in PLAN Fase 2 punto 2) mostra il punto debole su Anti-UAV300 sotto i 64 px.
4. Fasi 3–4: `detector.py`, adapter `ids_siamfc`, settings e `cuasp.py`, con i relativi test.
5. V3 → V4b (ricerca iperparametri) → V4 → taratura delle soglie → V7/V8 al banco.
