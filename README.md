# 🧠 Eskalon v4.0 – Henkilökohtainen RAG-assistentti
Eskalon v4.0 on paikallinen, yksityinen ja erittäin suorituskykyinen tekoälyassistentti, joka on suunniteltu Acer Nitro -ympäristöön (RTX 4060). Se yhdistää koodauksen ammattilaistason tuen ja kodin älykkään "Second Brain" -muistin yhdeksi saumattomaksi kokonaisuudeksi.

## Yksityisyys
Kaikki Eskalon-projektiini syöttämäni data pysyy omalla Acer Nitro -koneellani. Järjestelmä ei lähetä koodiani tai muita tietoja kolmansille osapuolille, tarjoten turvallisen vaihtoehdon julkisille pilvipalveluille.


## Keskeiset Ominaisuudet
- Dual-Memory Arkkitehtuuri: Erilliset muistipankit ammattimaiseen koodaukseen (kielet) ja yksityiseen muistin hallintaan (Koti).

- Hybrid Search Engine: Hyödyntää sekä semanttista vektorihakua (GTR-T5) että perinteistä tekstihakua optimaalisen tarkkuuden saavuttamiseksi.

## Älykkyyspäivitykset:

- Query Expansion: Laajentaa kysymyksiö synonyymeillä löytääkseen parempaa kontekstia.

- Confidence Scoring: Laskee jokaiselle vastaukselle varmuusprosentin muistin osumien perusteella.

- Dynaaminen Chunking: Säätää haettavan tiedon määrää kysymyksen monimutkaisuuden mukaan.

- True Black UI: Streamlit-pohjainen käyttöliittymä, jossa on kustomoitu gradient-teema ja suuret, luettavat fontit.

## Tekniikka

- Vektorikanta: ChromaDB (paikallinen pysyvyys).

- Embedding: sentence-transformers/gtr-t5-base.

- LLM-moottori: Ollama (valittavissa 3 eri mallia: qwen2.5-coder:14b, qwen2.5 coder:7b ja llama3.1 ).

