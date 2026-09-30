# Kjentmann

**GPS-fri posisjonering med AI: finn ut hvor et bilde er tatt ved å sammenligne det med satellittbilder.**

GPS kan jammes og forfalskes. I Øst-Finnmark er det et daglig problem, og FFI anbefaler reserveløsninger for posisjonering. Et kamera som ser ned kan ikke jammes. Kjentmann kjenner igjen terrenget og gir en posisjon uten satellittsignaler.

> Status: **v0.2**. Grovsøk med DINOv2 og FAISS, målt mot testbilder fra en annen dato. Presis posisjon kommer i v0.3.

## Slik fungerer det

1. **Kart (én gang per område):** Et skyfritt Sentinel-2-bilde lastes ned og deles i overlappende ruter med kjent posisjon.
2. **Fingeravtrykk (v0.2):** Hver rute gjøres om til en vektor med DINOv2 og legges i en FAISS-indeks.
3. **Grovsøk (v0.2):** Et nytt bilde får sitt eget fingeravtrykk, og de 5 mest like rutene hentes.
4. **Finmatching (v0.3):** LightGlue matcher punkter mot de beste rutene og gir posisjon i meter, med en sikkerhetsscore.

## Kom i gang

Krever Python 3.10 eller nyere. Et NVIDIA-skjermkort gjør DINOv2 raskere, men er ikke nødvendig.

```bash
git clone https://github.com/Jorgenfje/kjentmann.git
cd kjentmann
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch med GPU-støtte (NVIDIA). Uten GPU: dropp denne linjen.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"

kjentmann all      # v0.1: kart, ruter og et kart du kan åpne
kjentmann v02      # v0.2: testbilder, søk og evaluering
pytest             # testene (uten nett)
```

Resultat:

- `data/askim_map.html`: satellittbildet og rutenettet over Kartverkets kart
- `data/results/askim/results.md`: tabell med treffsikkerhet
- `data/results/askim/dinov2_map.html`: hvert testbilde på kartet, grønt for funnet og rødt for bom

## Evaluering

Testbildene er 200 utsnitt fra et Sentinel-2-bilde tatt på en **annen dato** enn kartet, med annet lys, andre skygger og annen vegetasjon. De er plassert tilfeldig, uavhengig av rutenettet. Et søk er et treff når en returnert rute faktisk inneholder testbildets sentrum.

Tre tall sammenlignes:

- **pixel:** en naiv metode som sammenligner forminskede piksler direkte
- **dinov2:** fingeravtrykk fra DINOv2 (ViT-S/14)
- **tilfeldig:** hva ren gjetting ville gitt, regnet ut eksakt

Resultatene for Askim kommer her etter første kjøring.

Antagelse: bildene er nordvendte. I et ekte system kommer retningen fra kompasset.

Område, datoer, antall testbilder og modell endres i `config.yaml`.

## Data

Sentinel-2 L2A fra Copernicus-programmet, hentet via den åpne [Earth Search](https://earth-search.aws.element84.com/v1)-katalogen. Ingen konto eller API-nøkkel trengs. Bare pikslene innenfor området lastes ned.

## Veikart

- [x] v0.1 Kartet finnes
- [x] v0.2 Første treff (DINOv2 + FAISS)
- [ ] v0.3 Presis posisjon (LightGlue) og evaluering
- [ ] v0.4 Norsk vinter: treffsikkerhet per årstid
- [ ] v0.5 På nett (Docker, Azure)
- [ ] v1.0 Demo og lansering

## Lisens

MIT. Sentinel-2-data: Copernicus Sentinel data, se [vilkår](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
