# Protocollo di registrazione IDS — validazione SiamFC al coperto (V4)

Per chi fa le riprese: l'operatore al portatile e il pilota del drone. Non serve conoscere il codice.

## A cosa servono

Il detector e il tracker sono stati addestrati e provati su video **all'aperto**. Prima di
usarli per puntare il laser servono riprese **dell'ambiente reale**, per misurare quattro cose:

1. **Falsi allarmi dell'ambiente**: oggetti fissi (lampade, strutture, riflessi) che il detector scambia per un drone. È il rischio principale e si misura solo con riprese **senza drone**.
2. **Prestazioni al coperto**: il drone viene trovato e seguito su pareti, lampade e strutture?
3. **Taratura delle soglie** con dati veri invece che con i dataset pubblici.
4. **Riferimento angolare**: con i marker ArUco montati sul drone, il vecchio sistema fornisce angoli di confronto sugli stessi frame.

Durata delle riprese: circa **35–40 minuti**. La sessione completa, compresa la preparazione, richiede circa 2 ore.

## Vincolo da tenere presente: l'inquadratura è stretta

Con il teleobiettivo, **a 21 m la camera vede solo circa 2,0–2,6 m di larghezza** (circa
1,5–2 m di altezza). Durante le riprese la camera è ferma, perché i motori non vengono
comandati, quindi il drone deve muoversi dentro questa finestra.

- Prima di iniziare, segnare in sala i limiti dell'inquadratura a 21 m (nastro, coni, riferimenti sulle pareti), guardando l'anteprima sul portatile.
- Durante gli spezzoni, l'operatore guida il pilota a voce guardando l'anteprima.
- A 21 m il drone appare grande: circa 130 px di lato visto di fianco e fino a circa 340 px visto da sotto o di fronte, su 2592 px di larghezza.

## Prima della sessione

**Portatile** (il PC di training ed esercizio, con la camera IDS collegata):
- [ ] SDK IDS peak installato, con i binding Python (`ids_peak` 1.17.0, `ids_peak_ipl` 1.18.0) nella venv di `cuasp_siamfc`. Sul portatile di training sono già installati (2026-11-11); su un altro PC vedere il README, sezione 8.
- [ ] Spazio libero su disco: **almeno 200 GB**. Le riprese occupano circa 100–180 GB.
- [ ] Portatile collegato alla corrente, in modalità prestazioni, sospensione disattivata, altri programmi chiusi.
- [ ] **Non avviare `cuasp.py`.** Il registratore non comanda i motori. In `settings.py` i comandi motore restano disabilitati (`CV_CUAS_ENABLE_MOTOR_COMMANDS = 0`).

**Camera** (le stesse impostazioni del funzionamento reale):
- [ ] Impostazioni lette da `cuasp_siamfc/control/suncubes/settings.py`, da non cambiare per le riprese: 2592x1944, Mono8, esposizione 20 ms, guadagno e frequenza come configurati. Il registratore le riporta in ogni `meta.json`.
- [ ] **Messa a fuoco e zoom dell'obiettivo fissi**, come per l'uso reale. Non toccarli: la calibrazione non sarebbe più valida.

**Materiale:**
- [ ] Metro laser o rotella, per misurare la distanza del drone dalla camera in ogni spezzone.
- [ ] Marker ArUco montati sul drone, solo per il blocco C (ID 1 e 2, dizionario 4x4, 5 cm).
- [ ] Questo protocollo stampato, o il registro di sessione in fondo.

**Prova del registratore** (5 minuti, da `cuasp_siamfc/`):
```powershell
# 1) senza camera: verifica disco e programma
.\.venv\Scripts\python tools\record_ids.py --name prova_fake --fake --duration 20 --no-preview
# 2) con la camera: 10 secondi della scena
.\.venv\Scripts\python tools\record_ids.py --name prova_camera --duration 10
```
Nella prova con la camera, controllare anche che l'**anteprima sia un'immagine normale**
in scala di grigi: niente strisce, immagine ripetuta o rumore uniforme. Se non lo è, il
formato pixel della camera non è Mono8: fermarsi e segnalarlo.

Alla fine di ogni registrazione il programma stampa due righe da controllare:
- `saved N frames ... measured XX fps`: annotare gli fps misurati. Con un'esposizione di 20 ms è normale che siano circa 50 invece di 70. È un dato che ci serve, non un errore.
- `lost: capture 0, writer queue 0, write errors 0`: deve essere tutto **0**. Altrimenti vedere "Problemi" in fondo.

## Come registrare uno spezzone

Ogni spezzone è un comando, lanciato da `cuasp_siamfc/`. Esempio:
```powershell
.\.venv\Scripts\python tools\record_ids.py --name B1_stazionario_r1 --block B --distance-m 21.0 --lighting "tutte accese" --drone "Mavic Mini" --notes "stazionario al centro, nessuno in sala"
```
- Si apre l'anteprima: **`q` ferma la registrazione**. In alternativa `--duration 60` ferma da sola dopo 60 secondi.
- `--name`: il codice dello spezzone (tabelle sotto) più `_r1`, `_r2`… per le ripetizioni. Solo lettere, numeri, `-` e `_`.
- `--block`: A, B, C o D.
- `--distance-m`: la distanza **misurata**, se disponibile.
- `--notes`: cosa succede, in breve (persone in sala, luci, imprevisti).
- I file finiscono in `recordings\<data_ora>_<nome>\`. Il programma non sovrascrive mai una cartella esistente.

Dopo ogni spezzone controllare le due righe finali (frame salvati, perse = 0) e annotarle nel registro di sessione.

## Blocco A — Scena vuota (priorità massima)

**Nessun drone in volo né visibile.** La camera va puntata, una zona alla volta, su ogni
parte di sfondo in cui il drone volerà o potrebbe trovarsi, con un margine attorno. Per
spostare la camera tra una zona e l'altra si usano gli strumenti del rig, a discrezione
dell'operatore; mai `cuasp.py`.

| Codice | Cosa | Durata |
|---|---|---|
| A1–A8 | Una zona di sfondo per codice (da 5 a 8 zone), condizioni normali di esercizio | 60 s ciascuna |
| A9 | La zona più "difficile" (lampade, strutture, riflessi), con **persone o oggetti in movimento** sullo sfondo | 60 s |
| A10 | Come A9, con un'altra configurazione di luci, se in esercizio se ne usano diverse | 60 s |

Annotare nelle note cosa contiene ogni zona (per esempio "parete est, 2 lampade, finestra").
Totale: circa **10 minuti**.

## Blocco B — Drone a 20–22 m

Spezzoni da 30–60 secondi, ciascuno ripetuto **2–3 volte** (`_r1`, `_r2`, `_r3`).
Misurare la distanza all'inizio di ogni spezzone.

| Codice | Cosa deve fare il drone | Durata |
|---|---|---|
| B1 | **Stazionario** al centro dell'inquadratura | 30–60 s |
| B2 | **Movimenti lenti** orizzontali e verticali, fino ai bordi dell'inquadratura | 45–60 s |
| B3 | **Movimenti rapidi**, alla massima velocità realistica per l'uso previsto, restando nell'inquadratura | 30–45 s |
| B4 | **Esce dall'inquadratura e rientra**, 3–4 volte, da lati diversi, restando fuori qualche secondo | 45–60 s |
| B5 | **Rotazioni**: il drone ruota su se stesso, così da essere visto di lato, di fronte e, se possibile, da sotto | 30–45 s |
| B6 | **Sfondi difficili**: passa davanti a lampade, finestre, strutture, zone molto chiare o molto scure | 45–60 s |
| B7 | **Distanze diverse**: stazionario a circa 18, 20, 22, 25 m (un sotto-spezzone per distanza, con la distanza nel nome, per esempio `B7_18m_r1`) | 20 s per distanza |
| B8 | Facoltativo: passa dietro un pilastro o una struttura; oppure una persona cammina sullo sfondo mentre il drone è in stazionario | 30–45 s |

Totale: circa **15–20 minuti**.

## Blocco C — Drone con marker ArUco

Con i **marker ArUco montati sul drone**, ripetere:

| Codice | Cosa | Durata |
|---|---|---|
| C1 | Come B1 (stazionario al centro) | 60 s, 2 volte |
| C2 | Come B2 (movimenti lenti) | 60 s, 2 volte |

Totale: circa **5 minuti**. In `--drone` scrivere "Mavic Mini + ArUco 1,2".

## Blocco D — Camera in movimento (facoltativo)

Solo se l'operatore può muovere il gimbal **in sicurezza con gli strumenti del rig**, non
con `cuasp.py`. Drone in stazionario a 21 m; la camera si sposta lentamente, poi più
velocemente, fino a perdere e ritrovare il drone.

| Codice | Cosa | Durata |
|---|---|---|
| D1 | Spostamenti lenti della camera, drone sempre inquadrato | 30–45 s |
| D2 | Spostamenti ampi: il drone esce dall'inquadratura e la camera torna a inquadrarlo | 30–45 s |

## Dopo la sessione

- Lasciare la cartella `recordings\` sul portatile così com'è: non rinominare, non cancellare, non convertire. Ogni spezzone contiene `frames\`, `timestamps.csv` e `meta.json`.
- Consegnare il registro di sessione, anche come foto. Le note a mano completano i `meta.json`.
- Le etichette (dove si trova il drone nei frame) **non** si fanno durante la sessione: si preparano dopo, in parte automaticamente.

## Problemi

| Sintomo | Cosa fare |
|---|---|
| `lost: capture N` diverso da 0 | Il programma non ha preso tutti i frame. Chiudere gli altri programmi; rilanciare con `--no-preview`; ripetere lo spezzone. |
| `writer queue N` diverso da 0 | Disco o CPU non tengono il ritmo. Chiudere gli altri programmi, lasciare il portatile alla corrente, provare `--writers 6`; ripetere lo spezzone. |
| `timestamp gaps` diverso da 0 | Frame persi prima di arrivare al PC (cavo, USB, carico). Annotarlo; se ricorre, controllare il collegamento della camera. |
| fps misurati circa 50 invece di 70 | Atteso con un'esposizione di 20 ms. Annotarlo e proseguire. |
| `No IDS camera detected` | Camera non collegata o non riconosciuta: ricollegarla e verificare con IDS peak Cockpit, poi **chiudere il Cockpit** prima di registrare (la camera si apre in modo esclusivo). |
| `OpenBLAS error: Memory allocation ... failed`, o errori di memoria all'avvio | La memoria virtuale di Windows è esaurita, spesso per Esplora risorse rimasto aperto per giorni. Gestione attività → "Esplora risorse" → **Riavvia**, oppure riavviare il PC. |
| Spazio su disco quasi esaurito | Il programma rifiuta di partire sotto i 20 GB liberi. Liberare spazio prima di continuare. |

## Registro di sessione

Data: ______  Luogo: ______  Operatore: ______  Pilota: ______  Drone: ______

Configurazione luci: ______  fps misurati nella prova camera: ______

| Spezzone (`--name`) | Ora | Distanza (m) | Frame salvati | Perse (capture / writer / gaps) | Note |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
