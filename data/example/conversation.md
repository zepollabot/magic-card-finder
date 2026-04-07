sto provando a scrivere un software che, data una serie di immagini contenenti delle carte di magic, sia in grado di dire quante carte ci sono nell'immagine ed il loro nome..
Sto provando con OpenCV ma per ora ho scarsi risultati.
Ho pensato di utilizzare un approccio del tipo object detection con Ultralytics YOLO al quale vorrei poi insegnare a riconoscere la posizione del nome delle carte e fare in modo che venga estratta in modo corretto.
Secondo te è una buona soluzione? Come potrei fare ad implementarlo in un progetto docker compose?

Mar 17
Analisi dell'approccio
L'idea è buona ma ti suggerisco un approccio a due stadi che è più robusto:

Stadio 1 — Card Detection (YOLO)
Usare YOLO per rilevare i bounding box di ogni carta nell'immagine. Qui YOLO eccelle perché:

Gestisce bene oggetti multipli sovrapposti/ruotati
È veloce e funziona bene anche senza GPU dedicata
Ha bisogno di relativamente pochi dati di training (qualche centinaio di immagini di mazzi sono sufficienti)
Stadio 2 — Name Recognition (OCR/Hash)
Una volta estratto il crop di ogni carta hai due opzioni:

Approccio Pro Contro
OCR sul nome (es. EasyOCR + crop della fascia titolo) Funziona senza dataset, generalizza a carte nuove Sensibile a rotazione, luce, qualità immagine
Image hashing / embedding (es. pHash o CLIP) Molto preciso, robusto alla qualità Richiede un database di immagini di riferimento
API Scryfall (dopo OCR parziale) Database completo e aggiornato gratis Richiede connessione, OCR deve essere abbastanza buono
Il combo vincente è: YOLO → crop → EasyOCR sul titolo → lookup su Scryfall API per normalizzare il nome.

Architettura Docker Compose
project/
├── docker-compose.yml
├── detector/ # Servizio YOLO (FastAPI)
│ ├── Dockerfile
│ ├── main.py
│ └── models/ # pesi .pt
├── recognizer/ # Servizio OCR + Scryfall (FastAPI)
│ ├── Dockerfile
│ └── main.py
├── api_gateway/ # Nginx o FastAPI aggregatore
│ └── ...
└── frontend/ # Opzionale, UI upload immagini
yaml

# docker-compose.yml

services:
detector:
build: ./detector
ports: - "8001:8001"
volumes: - ./detector/models:/app/models
deploy:
resources:
reservations:
devices: - driver: nvidia
count: all
capabilities: [gpu] # rimuovi se no GPU

recognizer:
build: ./recognizer
ports: - "8002:8002"
environment: - SCRYFALL_API=https://api.scryfall.com

gateway:
build: ./api_gateway
ports: - "8000:8000"
depends_on: - detector - recognizer

redis:
image: redis:alpine # cache risultati Scryfall
ports: - "6379:6379"
Codice base dei servizi
detector/main.py

python
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import cv2, numpy as np

app = FastAPI()
model = YOLO("models/card_detector.pt") # o yolov8n.pt per iniziare

@app.post("/detect")
async def detect(file: UploadFile):
img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD*COLOR)
results = model(img)
cards = []
for box in results[0].boxes:
x1,y1,x2,y2 = map(int, box.xyxy[0])
crop = img[y1:y2, x1:x2] # estrai solo la fascia del titolo (top ~10% del crop)
title_strip = crop[:int(crop.shape[0]\*0.12), :]
*, buf = cv2.imencode(".jpg", title_strip)
cards.append({
"bbox": [x1,y1,x2,y2],
"title_crop_b64": buf.tobytes().hex()
})
return {"count": len(cards), "cards": cards}
recognizer/main.py

python
from fastapi import FastAPI
import easyocr, httpx, redis, json

app = FastAPI()
reader = easyocr.Reader(['en'])
cache = redis.Redis(host='redis', port=6379)

@app.post("/recognize")
async def recognize(payload: dict): # payload: { "title_crop_b64": "..." }
img_bytes = bytes.fromhex(payload["title_crop_b64"])
text_results = reader.readtext(img_bytes)
raw_name = " ".join([t[1] for t in text_results])

    # lookup su Scryfall con fuzzy search
    cached = cache.get(raw_name)
    if cached:
        return json.loads(cached)

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.scryfall.com/cards/named",
            params={"fuzzy": raw_name}
        )
    result = {"raw_ocr": raw_name, "card": r.json() if r.status_code==200 else None}
    cache.setex(raw_name, 3600, json.dumps(result))
    return result

Roadmap pratica
Prima fase — bootstrap senza training: usa yolov8n.pt pretrained e testa solo il detection su foto di mazzi. YOLO riconosce già oggetti rettangolari, potresti ottenere risultati discreti con zero training.
Dataset training: cerca su Roboflow Universe, esistono già dataset annotati di carte Magic. Fine-tuning con 200-500 immagini è sufficiente.
Calibrazione OCR: la fascia del titolo è sempre nella stessa posizione relativa nella carta — questo è il tuo vantaggio enorme. Puoi hardcodare il crop top 10-12% del bounding box.
Fallback: se l'OCR ha confidence bassa, puoi fare un secondo tentativo con image embedding (CLIP) confrontando il crop con miniature scaricate da Scryfall.
Vuoi che ti scriva il Dockerfile per uno dei servizi o che approfondisca la parte di training YOLO?

scusa ma perché devo usare YOLOV8 se siamo alla V26?

Mar 17
Hai ragione a fare la domanda, ma la situazione è un po' più sfumata!

Cercato sul web
Cercato sul web
Hai completamente ragione! Mi scuso per l'errore — stavo usando informazioni datate. Ecco la situazione aggiornata:

YOLO26 è uscito il 14 gennaio 2026 ed è la versione attuale, disponibile come yolo26n.pt, yolo26s.pt, ecc.
Ultralytics

Le novità rilevanti per il tuo progetto sono:

NMS-free end-to-end inference: elimina il post-processing Non-Maximum Suppression, rendendo il deployment più veloce e semplice
Ultralytics
43% più veloce su CPU: ottimo se non hai una GPU dedicata
Ultralytics
ProgLoss + STAL: migliora l'accuratezza specialmente su oggetti piccoli — utile per carte sovrapposte o parzialmente visibili
Ultralytics
MuSGD optimizer: training più stabile e veloce da convergere
Ultralytics
Quindi il codice del detector che ti ho mostrato prima va semplicemente aggiornato così:

python

# PRIMA (sbagliato)

model = YOLO("yolov8n.pt")

# ADESSO (corretto)

model = YOLO("yolo26n.pt")
Per workload in produzione stabili, Ultralytics raccomanda sia YOLO26 che YOLO11 come scelte consigliate
Ultralytics
, ma per un progetto nuovo conviene partire direttamente da YOLO26.

La buona notizia è che l'API di training rimane identica
Ultralytics
, quindi tutto il resto dell'architettura che ti ho descritto resta valido — cambia solo il nome del modello. Vuoi che aggiorni il docker-compose completo con YOLO26?

più che altro mi interessa il dockerfile del progetto detector e come fare il training usando per esempio questo preset https://universe.roboflow.com/proyectgraduacion/mtg-cards-label

Mar 17

File visualizzato, cercato sul web
File visualizzato, cercato sul web
Ho analizzato il dataset. Noto subito una cosa importante da considerare.

Il dataset ha 71 immagini e 5 classi (card, name, text, type, power-toughness) — ottimo perché annota già esattamente la posizione del nome sulla carta, che è proprio quello che ti serve. Tuttavia 71 immagini sono poche, quindi faremo data augmentation aggressiva durante il training.

Nota anche che il dataset originale era per YOLOv5, ma il formato delle annotazioni (YOLO txt) è compatibile con YOLO26 senza modifiche.

Ecco il tutto:

detector/Dockerfile
dockerfile
FROM python:3.11-slim

# Dipendenze di sistema per OpenCV e build tools

RUN apt-get update && apt-get install -y \
 libgl1 \
 libglib2.0-0 \
 libsm6 \
 libxrender1 \
 libxext6 \
 curl \
 git \
 && rm -rf /var/lib/apt/lists/\*

WORKDIR /app

# Installa PyTorch CPU (sostituisci con versione CUDA se hai GPU)

RUN pip install --no-cache-dir \
 torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Ultralytics + API + Roboflow client

RUN pip install --no-cache-dir \
 ultralytics \
 roboflow \
 fastapi \
 uvicorn[standard] \
 python-multipart \
 easyocr \
 httpx

# Copia il codice

COPY . .

# Directory per modelli e dataset

RUN mkdir -p /app/models /app/datasets

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
Se hai GPU, sostituisci la riga torch con:

dockerfile
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
detector/train.py — Script di training completo
python
from roboflow import Roboflow
from ultralytics import YOLO
import yaml, os

# ─── 1. Scarica il dataset da Roboflow ───────────────────────────────────────

# Ottieni la tua API key su https://app.roboflow.com/settings/api

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

project = rf.workspace("proyectgraduacion").project("mtg-cards-label")
version = project.version(5) # versione 5, la più recente nel dataset
dataset = version.download("yolov8") # formato compatibile anche con YOLO26

# Il dataset viene scaricato in ./mtg-cards-label-5/

# con struttura:

# train/images, train/labels

# valid/images, valid/labels

# test/images, test/labels

# data.yaml

# ─── 2. Patch del data.yaml (aggiusta i path assoluti se necessario) ─────────

dataset_yaml = f"{dataset.location}/data.yaml"

with open(dataset_yaml) as f:
cfg = yaml.safe_load(f)

# Forza i path assoluti (evita problemi se lanci da directory diverse)

cfg["path"] = dataset.location
cfg["train"] = "train/images"
cfg["val"] = "valid/images"
cfg["test"] = "test/images"

with open(dataset_yaml, "w") as f:
yaml.dump(cfg, f)

print(f"Classi trovate: {cfg['names']}")

# Atteso: {0: 'card', 1: 'name', 2: 'power-toughness', 3: 'text', 4: 'type'}

# ─── 3. Carica YOLO26 e fai fine-tuning ──────────────────────────────────────

model = YOLO("yolo26n.pt") # nano = più veloce; usa yolo26s.pt per più accuratezza

results = model.train(
data = dataset_yaml,
epochs = 100,
imgsz = 640,
batch = 16, # riduci a 8 se hai poca RAM
patience= 20, # early stopping se non migliora per 20 epoche
device = "cpu", # "0" se hai GPU NVIDIA

    # ── Data augmentation aggressiva (dataset piccolo = serve!) ──
    hsv_h   = 0.015,       # variazione tonalità
    hsv_s   = 0.5,         # saturazione
    hsv_v   = 0.4,         # luminosità
    degrees = 10,          # rotazione ±10°
    translate=0.1,
    scale   = 0.5,
    shear   = 2.0,
    flipud  = 0.1,
    fliplr  = 0.5,
    mosaic  = 1.0,         # combina 4 immagini — molto utile con pochi dati
    mixup   = 0.1,

    # ── Output ───────────────────────────────────────────────────
    project = "/app/models",
    name    = "mtg_detector",
    save    = True,
    plots   = True,        # genera grafici loss/mAP in models/mtg_detector/

)

print(f"\n✅ Training completato!")
print(f" Best weights: /app/models/mtg_detector/weights/best.pt")
print(f" mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
detector/main.py — API di inferenza
python
from fastapi import FastAPI, UploadFile, HTTPException
from ultralytics import YOLO
import cv2, numpy as np, base64, os

app = FastAPI(title="MTG Card Detector")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/mtg_detector/weights/best.pt")

# Fallback al modello base se il training non è ancora stato fatto

if not os.path.exists(MODEL_PATH):
print(f"⚠️ {MODEL_PATH} non trovato, uso yolo26n.pt base")
MODEL_PATH = "yolo26n.pt"

model = YOLO(MODEL_PATH)

CLASS_NAMES = {0: "card", 1: "name", 2: "power-toughness", 3: "text", 4: "type"}

@app.post("/detect")
async def detect(file: UploadFile, conf: float = 0.4):
if not file.content_type.startswith("image/"):
raise HTTPException(400, "Il file deve essere un'immagine")

    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Impossibile decodificare l'immagine")

    results = model(img, conf=conf)[0]

    cards = []
    name_crops = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, "unknown")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        crop = img[y1:y2, x1:x2]
        _, buf = cv2.imencode(".jpg", crop)
        crop_b64 = base64.b64encode(buf).decode()

        detection = {
            "class": cls_name,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2],
            "crop_b64": crop_b64,
        }

        if cls_name == "card":
            cards.append(detection)
        elif cls_name == "name":
            name_crops.append(detection)

    return {
        "total_cards": len(cards),
        "cards": cards,
        "name_regions": name_crops,   # pronti per OCR nel servizio recognizer
    }

@app.get("/health")
def health():
return {"status": "ok", "model": MODEL_PATH}
docker-compose.yml aggiornato (solo servizio detector)
yaml
services:
detector:
build: ./detector
ports: - "8001:8001"
volumes: - ./detector/models:/app/models # persiste i pesi dopo il training - ./detector/datasets:/app/datasets # persiste il dataset scaricato
environment: - ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY} - MODEL_PATH=/app/models/mtg_detector/weights/best.pt # Per GPU: decommenta questo blocco # deploy: # resources: # reservations: # devices: # - driver: nvidia # count: 1 # capabilities: [gpu]
Workflow completo
bash

# 1. Crea il file .env con la tua API key Roboflow

echo "ROBOFLOW_API_KEY=la_tua_chiave" > .env

# 2. Build e avvia

docker compose up -d detector

# 3. Lancia il training (una tantum, ci vorrà ~30min su CPU)

docker compose exec detector python train.py

# 4. L'API è subito disponibile su http://localhost:8001

# testa con:

curl -X POST http://localhost:8001/detect \
 -F "file=@foto_mazzo.jpg" | python -m json.tool
Un paio di cose da tenere a mente: con solo 71 immagini il modello sarà buono a localizzare le carte ma potrebbe faticare in condizioni di luce difficile — ti conviene integrare il dataset con altre 50-100 tue foto annotate tramite Roboflow (gratuito fino a 1000 immagini). Inoltre la classe name ti dà direttamente il crop della fascia del titolo, che è il vantaggio principale di questo dataset rispetto a uno generico.

spiegami come fare per integrare il dataset con mie foto annotate tramite Roboflow

Mar 17
Ecco la guida passo passo!

1. Crea un account e un nuovo progetto
   Vai su app.roboflow.com, registrati gratuitamente e crea un nuovo progetto:

Project Type: Object Detection
Project Name: es. mtg-cards-custom
Annotation Group: cards 2. Importa il dataset esistente come base
Invece di partire da zero, cloni il dataset di proyectgraduacion così erediti già le 71 immagini annotate:

Cercato sul web
Cercato sul web
Perfetto, ho tutto quello che mi serve. Ecco la guida completa:

Step 1 — Fork del dataset esistente
Vai sulla pagina del dataset su Universe, clicca "Use this Dataset" e poi "Fork Project": il dataset verrà copiato nel tuo workspace con tutte le annotazioni già incluse.
Ultralytics

Avrai così le 71 immagini originali già annotate come punto di partenza.

Step 2 — Carica le tue foto
Nell'app Roboflow, dentro il tuo progetto forkato:

Clicca "Upload Data"
Trascina le tue foto (JPG/PNG, anche in batch)
Assegnale al set "Training" (per ora, le sposterai dopo l'annotazione)
Consigli su che foto fare:

Variare l'angolazione (frontale, leggermente inclinato)
Variare la luce (luce naturale, artificiale, ombre)
Foto con più carte sovrapposte o a ventaglio
Foto con carte parzialmente visibili ai bordi
Sfondo diverso (tavolo legno, tappetino da gioco verde, mano che regge le carte)
Più varietà fai, più robusto sarà il modello.

Step 3 — Annota le immagini
Roboflow ha un tool di annotazione integrato. Le classi da usare sono esattamente quelle del dataset originale — fondamentale per non creare conflitti:

Classe Cosa annotare
card L'intera carta
name Solo la fascia del titolo (riga in alto)
type La riga del tipo (es. "Creature — Dragon")
text Il riquadro del testo delle abilità
power-toughness Il numerino in basso a destra (es. "3/3")
Trucco importante: per il tuo caso d'uso (riconoscere il nome) la classe più critica è name. Se hai poco tempo, annota almeno card e name su tutte le foto — le altre classi puoi aggiungerle progressivamente.

Usa l'Auto-Label per velocizzare
Roboflow offre un tool di Model-Assisted Labeling: dopo aver generato una prima versione del modello (anche con le sole 71 immagini originali), puoi farlo girare sulle nuove foto e lui pre-annota automaticamente. Tu poi correggi solo gli errori — riduce il lavoro di annotazione dell'80%.

Step 4 — Genera una nuova versione del dataset
Una volta annotate le tue foto, clicca "Generate New Version". Roboflow ti chiederà le impostazioni di preprocessing e augmentation:

Preprocessing consigliato:

Auto-Orient: ✅ ON (corregge le foto scattate in verticale)
Resize: 640×640 (corrisponde a imgsz=640 nel training)
Augmentation — dato che il dataset è piccolo, attiva tutto:

Flip orizzontale: ✅
Rotazione: ±15°
Luminosità: ±25%
Rumore: fino al 2%
Mosaic: ✅ (combina 4 immagini, molto efficace)
Roboflow moltiplicherà le immagini automaticamente (es. 150 foto → ~450 dopo augmentation).

Step 5 — Aggiorna il train.py
Cambia solo la versione del dataset nello script di training:

python

# Prima puntavi alla versione 5 del dataset originale

version = project.version(5)

# Adesso punti al TUO progetto forkato con la versione più recente

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("tuo-workspace").project("mtg-cards-custom")
version = project.version(1) # o il numero della versione che hai generato
dataset = version.download("yolov8")

```

Il resto del `train.py` rimane identico — YOLO26 farà il fine-tuning sul dataset combinato (originale + tue foto).

---

## Ciclo iterativo consigliato
```

Foto tue (20-30) → Annota → Genera versione → Lancia training
↓
Testa il modello su foto nuove
↓
Dove sbaglia? → Fai più foto di quel tipo → Annota → Ri-train
Con 3-4 cicli di questo tipo e ~100-150 foto totali dovresti arrivare a un mAP50 > 0.85, che per un caso d'uso reale è più che sufficiente.

qual è la logica da seguire per taggare le immagini come training, valid e test?

Mar 17
Ottima domanda, è una delle cose più importanti da capire bene per non avere risultati fuorvianti.

La logica dei tre set
I tre set hanno scopi completamente diversi:

Set % tipica Scopo
Training 70-80% Il modello impara da questi dati
Validation 10-15% Guida il training, decide quando fermarsi (early stopping)
Test 10-15% Valutazione finale onesta — il modello non li ha mai visti
La regola fondamentale è che test e validation non devono mai "influenzare" il modello — se il modello li vede durante il training, stai barando con te stesso e le metriche finali (mAP) saranno ottimistiche rispetto alla realtà.

La regola d'oro: distribuisci per variabilità, non a caso
L'errore più comune è distribuire le immagini casualmente. Il problema è che se hai fatto 10 foto della stessa carta con la stessa luce, e 7 vanno in training e 3 in test, il modello ha già "visto" quella carta in condizioni simili — il test non è onesto.

La distribuzione corretta si basa su scenari, non su immagini:

Sessione A — tavolo legno, luce naturale, 30 foto → 80% train, 20% val
Sessione B — tappetino verde, luce artificiale, 20 foto → tutto TEST
Sessione C — mano che regge le carte, 15 foto → tutto TEST o VAL
In pratica: le sessioni fotografiche diverse vanno in set diversi.

Regole pratiche per il tuo caso MTG
Metti in Training:

La maggior parte delle foto "normali"
Varie carte, varie condizioni, ma tutte sessioni che usi per insegnare
Metti in Validation:

Una sessione fotografica separata, condizioni simili al training
Roboflow la usa in tempo reale per mostrarti i grafici loss/mAP durante il training
Se il validation loss smette di scendere → early stopping
Metti in Test:

Una sessione fotografica completamente separata, idealmente fatta dopo aver fatto training
Condizioni diverse: luce diversa, sfondo diverso, carte che non compaiono nel training
Non guardarle mai fino alla valutazione finale
Con dataset piccolo (il tuo caso)
Con ~150 foto totali, 15-20 immagini di test sono sufficienti. L'importante è che siano qualitativamente diverse, non solo numericamente separate. Una regola pratica:

100 foto sessione A → Training (80) + Validation (20)
30 foto sessione B → Test
20 foto sessione C → Test
Meglio 20 foto di test davvero diverse che 40 foto simili a quelle di training.

In Roboflow
Quando carichi le immagini puoi assegnarle manualmente al set corretto trascinandole, oppure usare lo split automatico 70/20/10 — ma in quel caso raggruppale prima per sessione e fai lo split sessione per sessione, non sull'intero batch. Questo evita che foto simili finiscano in set diversi.

ho intenzione di dare in pasto al modello altre 2-300 immagini taggate da me a mano per il training

Mar 17
Ottima scelta, con 300 immagini ben fatte il modello diventerà molto più robusto. Prima di iniziare a taggare però ti conviene pianificare bene per non sprecare tempo.

Quante immagini per set
Con 300 tue + 71 originali = ~370 totali:

Training: ~260 immagini (70%) ← le tue sessioni principali
Validation: ~55 immagini (15%) ← una sessione separata
Test: ~55 immagini (15%) ← sessione fatta per ultima, condizioni diverse
Cosa fotografare per massimizzare l'utilità
Il punto chiave è coprire i casi difficili, non aggiungere foto simili a quelle già esistenti nel dataset originale. Guardando le 71 immagini originali sono tutte foto abbastanza "pulite" e frontali, quindi le lacune da colmare sono:

Condizioni di luce:

Luce radente laterale (crea ombre sul testo del nome)
Controluce parziale
Luce artificiale gialla (lampada da tavolo)
Riflessi sulla carta (le carte lucide riflettono molto)
Geometria:

Carte tenute in mano a ventaglio
Carte leggermente curve (i mazzi usati si curvano)
Angolazione fino a ~30° rispetto alla camera
Carte parzialmente sovrapposte (il caso più comune in realtà)
Quantità per foto:

1 carta sola — facile, ma utile per il crop del nome
3-5 carte affiancate
7-10 carte a ventaglio come quando le tieni in mano
Pile di carte con solo le prime visibili
Varietà di carte:

Mix di carte con nomi corti (es. "Opt") e lunghi (es. "Dragonlord Ojutai")
Carte con cornice dorata, nera, colorata (il colore del bordo cambia)
Carte vecchie (cornice pre-2003, ha il layout diverso) se vuoi supportarle
Consigli pratici per velocizzare l'annotazione
Con 300 immagini l'annotazione manuale può diventare pesante. Alcune strategie:

1. Usa l'Auto-Label di Roboflow subito Dopo aver annotato le prime 50-60 immagini a mano, genera una versione preliminare del modello e attiva il Model-Assisted Labeling. Roboflow pre-annoterà le immagini successive e tu correggi solo gli errori — tipicamente dimezza i tempi.

2. Priorità alle classi Se il tempo è limitato, annota in questo ordine di priorità:

3. card ← sempre, su tutte le immagini
4. name ← sempre, è il tuo obiettivo principale
5. type ← utile ma secondario
6. text ← opzionale per ora
7. power-toughness ← opzionale, presente solo sulle creature
8. Annota in sessioni da 30-40 minuti L'attenzione cala e la qualità delle annotazioni peggiora. Meglio fare sessioni brevi e mantenere la precisione — un bounding box sloppy sul nome della carta è peggio di nessuna annotazione.

Un dettaglio importante sul bbox del nome
Dato che il nome estratto andrà poi in OCR, la precisione del bounding box della classe name è critica. Tienilo stretto: includi solo il testo del nome, senza il simbolo di mana a destra e senza margini eccessivi sopra/sotto.

❌ bbox largo che include anche il simbolo di mana
✅ bbox stretto attorno solo al testo del nome
Questo farà una differenza enorme sulla qualità dell'OCR a valle.

ma io non ho mica capito cosa devo fare onestamente.
In un pezzo di discorso mi dici di fare il training col file train.py, ma allora io cosa sto facendo su roboflow? non è un training di un modello?

Mar 17
Hai ragione, mi sono spiegato male. Facciamo chiarezza.

Roboflow NON fa il training del tuo modello
Roboflow è solo uno strumento di gestione dati. Quello che ci fai è:

Roboflow
├── Carichi le foto
├── Annoti i bounding box a mano (o con assistenza)
├── Organizzi in train/val/test
├── Applichi augmentation (ruota, schiarisci, ecc.)
└── Esporti il dataset in formato YOLO ← questo è l'output
Il risultato finale è una cartella con foto + file .txt di annotazioni, che scarichi sul tuo PC/server.

Il training vero e proprio è train.py
train.py (gira nel tuo container Docker)
├── Scarica il dataset da Roboflow (quello che hai preparato sopra)
├── Prende YOLO26 preaddestrato su milioni di immagini generiche
├── Lo "specializza" sulle tue carte Magic ← questo è il fine-tuning
└── Produce best.pt ← il tuo modello finale
Il flusso completo quindi è
[Tu con fotocamera]
↓
300 foto
↓
[Roboflow - solo gestione dati]
annoti i bbox
organizzi train/val/test
esporti in formato YOLO
↓
[Docker - train.py]
YOLO26 impara dalle tue annotazioni
↓
[Docker - main.py]
API che usa il modello addestrato
↓
/detect → "ci sono 5 carte, i nomi sono..."
Roboflow offre anche un suo training interno, ma è limitato, meno configurabile e a pagamento oltre un certo punto — per questo ti ho suggerito di usarlo solo come tool di annotazione ed esportare il dataset per addestrare tu stesso con YOLO26.

Ha più senso adesso?

con quale logica devo taggare una immagine come valid o test?
E che percentuale del numero totale di immagini devo avere taggate come valid e test?

Mar 19
Questa domanda l'hai già fatta qualche messaggio fa 😄 — ti avevo risposto in dettaglio. Ti riassumo i punti chiave:

Logica di assegnazione
La regola fondamentale è distribuire per sessione fotografica, non a caso:

Training → le tue sessioni fotografiche principali (condizioni normali)
Validation → una sessione separata, condizioni simili al training
Test → una sessione fatta per ultima, condizioni diverse (luce diversa, sfondo diverso)
Il motivo è che se hai fatto 10 foto della stessa carta nella stessa sessione e le distribuisci a caso, il modello in test "ha già visto" quella carta in condizioni simili — le metriche risultano falsamente ottimistiche.

Percentuali
Con ~370 immagini totali (300 tue + 71 originali):

Training: ~260 (70%)
Validation: ~55 (15%)
Test: ~55 (15%)
Con dataset piccoli come il tuo, l'importante è che i 55 di test siano qualitativamente diversi dal training — meglio 30 foto davvero diverse che 55 simili a quelle di training.

Vuoi che approfondiamo qualche aspetto specifico?

si ma tu presupponi che ci siano delle sessioni fotografiche di carte, cosa che hai deciso da solo. All'inizio della nostra conversazione ti avevo accennato che le immagini provengono da annunci internet, quindi non c'è nessuna "sessione fotografica" da cui partire. Ci sono solo delle foto brutte prese da internet, spesso fatte bene o male, dritte o storte, con cattiva luce o con buona luce, con tante carte o con una carta sola. Quindi ti rifaccio domanda, visto che è più sensata di quello che credi di sapere: in mezzo a questa miriade di immagini completamente casuali (dove non ci sono più foto della stessa carta in condizioni di luce variabile), quali sono i parametri per cui dovrei decidere "questa è una foto da training" o "questa è una foto da test" o "questa è una foto da validation"

Mar 19
Hai ragione, mi scuso — ho fatto un'assunzione sbagliata e poi ho continuato a ripeterla. Ricominciamo da capo con il contesto corretto.

Con immagini casuali da internet la logica cambia
Non avendo sessioni fotografiche, l'unico criterio sensato è distribuire la variabilità già presente in modo uniforme tra i tre set. In pratica:

Raggruppa visivamente le immagini per "tipo di difficoltà" e poi distribuisci ogni gruppo proporzionalmente:

Gruppo A — foto pulite, carta dritta, buona luce
Gruppo B — foto con più carte, parzialmente sovrapposte  
Gruppo C — foto storte, luce cattiva, sfondo caotico
Gruppo D — foto con una carta sola ravvicinata
Da ogni gruppo prendi il 70% per training, 15% per val, 15% per test — non pescare tutti i casi difficili nel training e lasciare solo quelli facili al test, altrimenti le metriche finali non riflettono la realtà.

Il criterio pratico concreto
Se non vuoi raggruppare manualmente, l'unica regola è:

Distribuisci a caso, ma verifica che i tre set abbiano una distribuzione visiva simile — se nel test finiscono solo le foto facili il modello sembrerà più bravo di quello che è, se finiscono solo le difficili sembrerà più scarso.

Roboflow ha uno split automatico casuale che per immagini eterogenee come le tue funziona ragionevolmente bene — con 300+ immagini la casualità tende già a distribuire bene la variabilità.

Le percentuali restano le stesse
Training: 70%
Validation: 15%
Test: 15%
La validation serve a YOLO durante il training per capire quando smettere — non deve essere "speciale", basta che sia rappresentativa. Il test invece tienilo davvero da parte e non guardarlo fino alla fine.

Claude è un'AI e può commettere errori. Verifica le risposte.
