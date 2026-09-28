# Kjentmann

**GPS-fri posisjonering med AI: finn ut hvor et bilde er tatt ved å sammenligne det med satellittbilder.**

GPS kan jammes og forfalskes. I Øst-Finnmark er det et daglig problem, og FFI anbefaler reserveløsninger for posisjonering. Et kamera som ser ned kan ikke jammes. Kjentmann kjenner igjen terrenget og gir en posisjon uten satellittsignaler.

> Status: **v0.1**. Kartet over testområdet lastes ned og deles i ruter. Gjenkjenning kommer i v0.2.

## Slik fungerer det

1. **Kart (én gang per område):** Et skyfritt Sentinel-2-bilde lastes ned og deles i overlappende ruter med kjent posisjon.
2. **Fingeravtrykk (v0.2):** Hver rute gjøres om til en vektor med DINOv2 og legges i en FAISS-indeks.
3. **Grovsøk (v0.2):** Et nytt bilde får sitt eget fingeravtrykk, og de 5 mest like rutene hentes.
4. **Finmatching (v0.3):** LightGlue matcher punkter mot de beste rutene og gir posisjon i meter, med en sikkerhetsscore.

## Kom i gang

Krever Python 3.10 eller nyere.

```bash
git clone https://github.com/Jorgenfje/kjentmann.git
cd kjentmann
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

kjentmann all      # laster ned kart, lager ruter og et kart du kan åpne
pytest             # kjører testene (uten nett)
```

Resultat:

- `data/map/askim.tif`: satellittbildet av området
- `data/tiles/askim/`: rutene som PNG, og `tiles.csv` med posisjonen til hver rute
- `data/askim_map.html`: interaktivt kart for å sjekke at alt ligger riktig

Området, datoer og rutestørrelse endres i `config.yaml`.

## Data

Sentinel-2 L2A fra Copernicus-programmet, hentet via den åpne [Earth Search](https://earth-search.aws.element84.com/v1)-katalogen. Ingen konto eller API-nøkkel trengs. Bare pikslene innenfor området lastes ned.

## Veikart

- [x] v0.1 Kartet finnes
- [ ] v0.2 Første treff (DINOv2 + FAISS)
- [ ] v0.3 Presis posisjon (LightGlue) og evaluering
- [ ] v0.4 Norsk vinter: treffsikkerhet per årstid
- [ ] v0.5 På nett (Docker, Azure)
- [ ] v1.0 Demo og lansering

## Lisens

MIT. Sentinel-2-data: Copernicus Sentinel data, se [vilkår](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
