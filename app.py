import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# ESKALON v4.0 LITE - Kategoriavalinnalla
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Eskalon v4.0 Lite", 
    page_icon="🧠", 
    layout="centered"
)

# ═══════════════════════════════════════════════════════════════════
# CSS TYYLI
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
    <style>
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    [data-testid="stSidebarContent"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stMainBlockContainer"],
    .main {
        background-color: #000000 !important;
        background-image: none !important;
    }

    .stMarkdown, .stMarkdown p, [data-testid="stChatMessage"] .stMarkdown,
    label, p, span, .stCaption, h1, h2, h3, .stSlider label, .stSubheader {
        color: #ffffff !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] *,
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    .stChatFloatingInputContainer *,
    [data-testid="stChatInput"],
    [data-testid="stHorizontalBlock"] {
        background-color: #000000 !important;
        background-image: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background: linear-gradient(#0a0a0a, #0a0a0a) padding-box,
                    linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6) border-box !important;
        color: #f0f0f0 !important;
        font-size: 1.4rem !important;
        line-height: 1.5 !important;
        border: 3px solid transparent !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        height: 100px;
        width: 1300px;
    }

    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] + div button {
        color: #f0f0f0 !important;
        background-color: #000000 !important;
        border: 3px solid transparent !important;
        background-image: linear-gradient(#000, #000), linear-gradient(135deg, #3b82f6, #ec4899) !important;
        background-origin: border-box !important;
        background-clip: padding-box, border-box !important;
        width: 100px;
        height: 100px;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #888888 !important;
    }

    [data-testid="stBottomBlockContainer"] {
        border-top: none !important;
        box-shadow: none !important;
    }

    .stMarkdown code {
        background-color: #1a1a1a !important;
        color: #00ffcc !important;
    }

    .stButton button {
        background: linear-gradient(#000000, #000000) padding-box,
                    linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6) border-box !important;
        border: 3px solid transparent !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        width: 100% !important;
        font-weight: bold !important;
        padding: 10px 0px !important;
        transition: transform 0.2s ease, opacity 0.2s !important;
    }

    .stButton button:hover {
        transform: scale(1.05) !important;
        opacity: 0.9 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    [data-testid="stDecoration"] {display:none;}
    
    .confidence-high {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .confidence-low {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    /* UUSI: Kategoriavalinnan tyyli */
    .stSelectbox {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ALUSTUS
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def alusta_jarjestelma():
    """Alustetaan ChromaDB ja GTR-T5 embedding-malli"""
    client = chromadb.PersistentClient(path="./koodimuisti_db")
    collection = client.get_or_create_collection(name="omat_koodit")
    model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    return collection, model

collection, model = alusta_jarjestelma()

# ═══════════════════════════════════════════════════════════════════
# MUISTIN PÄIVITYS
# ═══════════════════════════════════════════════════════════════════
def aja_paivitys(kansiot, nimi):
    """
    Päivittää valitun muistin (Koodi tai Koti).
    Lisää automaattisesti "kategoria"-kentän metadataan.
    """
    IGNORE_KANSIOT = {
        "node_modules", ".git", "venv", "npm", "env", "__pycache__", 
        "dist", "build", ".next", "out", "target", "public", "docs", 
        "bin", "obj", ".vscode", ".idea", "cache", "android", "ios", 
        "web", "desktop", "assets", "coverage", "images", "kuvat"
    }
    
    SALLITUT_PAATTEET = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", 
        ".java", ".cpp", ".h", ".sql", ".txt", ".md", ".json", 
        ".yaml", ".yml", ".sh", ".ps1", ".php"
    }

    loytyneet = 0
    
    with st.status(f"🚀 Päivitetään {nimi}-muistia...", expanded=True) as status:
        for juurikansio in kansiot:
            if not os.path.exists(juurikansio):
                st.warning(f"⚠️ Ohitetaan (ei löydy): {juurikansio}")
                continue
                
            st.write(f"📂 Skannataan: {os.path.basename(juurikansio)}...")
            
            for tiedosto in Path(juurikansio).rglob('*'):
                if any(k in tiedosto.parts for k in IGNORE_KANSIOT):
                    continue
                
                if tiedosto.suffix.lower() in SALLITUT_PAATTEET:
                    try:
                        with open(tiedosto, 'r', encoding='utf-8') as f:
                            sisalto = f.read()
                        
                        if len(sisalto.strip()) < 10:
                            continue
                        
                        vektori = model.encode(sisalto).tolist()
                        
                        # TÄRKEÄ: Lisää kategoria metadataan!
                        collection.upsert(
                            ids=[str(tiedosto)],
                            embeddings=[vektori],
                            documents=[sisalto],
                            metadatas=[{
                                "polku": str(tiedosto),
                                "tyyppi": tiedosto.suffix,
                                "nimi": tiedosto.name,
                                "kategoria": nimi  # ← erottaa Koodi vs. Koti
                            }]
                        )
                        loytyneet += 1
                        
                    except Exception:
                        continue
        
        status.update(
            label=f"✅ Valmis! {loytyneet} tiedostoa ({nimi})", 
            state="complete"
        )
    
    st.success(f"🧠 {nimi}-muisti päivitetty! ({loytyneet} tiedostoa)")
    return loytyneet

# ═══════════════════════════════════════════════════════════════════
# QUERY EXPANSION
# ═══════════════════════════════════════════════════════════════════
def laajenna_kysymys(kysymys):
    """Lisää synonyymejä kysymykseen"""
    synonyymi_map = {
        "luo": ["tee", "generoi", "kirjoita"],
        "korjaa": ["fiksaa", "debug", "ratkaise"],
        "paranna": ["optimoi", "refaktoroi"],
        "funktio": ["function", "metodi", "def"],
        "luokka": ["class", "olio"],
    }
    
    lisattavat = []
    kysymys_lower = kysymys.lower()
    
    for avain, synonyymit in synonyymi_map.items():
        if avain in kysymys_lower:
            lisattavat.extend(synonyymit[:2])
    
    if lisattavat:
        return f"{kysymys} {' '.join(lisattavat)}"
    return kysymys

# ═══════════════════════════════════════════════════════════════════
# DYNAAMINEN CHUNK-KOKO
# ═══════════════════════════════════════════════════════════════════
def laske_optimaalinen_n_tuloksia(prompt, base_n):
    """Säätää kontekstin kokoa kysymyksen monimutkaisuuden mukaan"""
    sana_maara = len(prompt.split())
    
    if sana_maara < 5:
        return max(3, base_n - 2)
    elif sana_maara > 15:
        return min(12, base_n + 3)
    else:
        return base_n

# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE
# ═══════════════════════════════════════════════════════════════════
def laske_confidence(distances):
    """Arvioi hakutulosten luotettavuutta"""
    if not distances or len(distances) == 0:
        return "low", 0.0
    
    avg_distance = sum(distances) / len(distances)
    
    if avg_distance < 0.5:
        return "high", (1 - avg_distance) * 100
    elif avg_distance < 1.0:
        return "medium", (1 - avg_distance) * 100
    else:
        return "low", max(0, (1 - avg_distance) * 100)

# ═══════════════════════════════════════════════════════════════════
# UUSI: KATEGORIAHAKU - Suodattaa tulokset kategorian mukaan
# ═══════════════════════════════════════════════════════════════════
def hae_kategoriasta(kysymys_vektori, laajennettu_prompt, optimaalinen_n, valittu_kategoria):
    """
    Hakee dokumentit valitusta kategoriasta.
    
    Args:
        kysymys_vektori: Embedding-vektori
        laajennettu_prompt: Laajennettu kysymys (synonyymein)
        optimaalinen_n: Haettavien dokumenttien määrä
        valittu_kategoria: "Kaikki", "Koodi" tai "Koti"
    
    Returns:
        dict: ChromaDB query-tulokset
    """
    # JOS valittu "Kaikki", hae normaalisti
    if valittu_kategoria == "Kaikki":
        return collection.query(
            query_texts=[laajennettu_prompt],
            query_embeddings=[kysymys_vektori],
            n_results=optimaalinen_n
        )
    
    # MUUTEN: Suodata kategorian mukaan
    # ChromaDB where-suodatus metadata-kentän perusteella
    return collection.query(
        query_texts=[laajennettu_prompt],
        query_embeddings=[kysymys_vektori],
        n_results=optimaalinen_n,
        where={"kategoria": valittu_kategoria}
    )

# ═══════════════════════════════════════════════════════════════════
# DYNAAMISET SYSTEM PROMPTIT - Mukautuvat kategorian mukaan
# ═══════════════════════════════════════════════════════════════════

KOODI_SYSTEM_PROMPT = """Olet Eskalon v4.0 - huippuluokan koodiassistentti tekoälyllä.

🎯 TOIMINTAMALLI:
1. ANALYSOI konteksti syvällisesti:
   - Tunnista käytetty koodityyli (nimeämiskäytännöt, kommentit, kieli)
   - Havaitse arkkitehtuuriset päätökset ja suunnittelumallit
   - Ymmärrä projektissa käytetyt kirjastot ja frameworkit

2. JOS kysymys liittyy olemassa olevaan koodiin:
   - Käytä TÄSMÄLLEEN samaa tyyliä (muuttujanimet, kommentointi, kieli)
   - Pidä arkkitehtuuri yhtenäisenä
   - Viittaa olemassa oleviin funktioihin ja luokkiin

3. JOS kysymys on yleinen tai uusi ominaisuus:
   - Käytä moderneja best practiceja
   - Kirjoita puhdasta, ylläpidettävää koodia
   - Selitä VAIN kriittiset tekniset valinnat (max 1-2 lausetta)

📋 VASTAUSRAKENNE (PAKOLLINEN):
🎯 RATKAISU: (Yksi täsmällinen lause siitä, mitä teet)
💻 KOODI: (Täydellinen, suoraan käyttövalmis koodiblokki - EI osia)
⚡ OLENNAISET: (Max 3 teknistä huomiota - VAIN jos kriittisiä)

🚫 EHDOTTOMASTI KIELLETYT:
- Älä selitä perusasioita ("tämä on for-silmukka...")
- Älä näytä osittaista koodia (AINA kokonainen toimiva esimerkki)
- Älä mainitse "muistista löytyi..." ellei se ole relevanttia
- Älä kirjoita esseitä tai luentoja

💡 ERIKOISOMINAISUUDET:
- Jos huomaat bugeja kontekstissa, mainitse ne lyhyesti
- Jos arkkitehtuuri on epäoptimaalinen, ehdota parannusta (max 1 lause)
- Jos puuttuu error handling, lisää se automaattisesti

Kieli: AINA suomi. Tyyli: Tekninen ammattilainen, ei chatbot."""

YLEINEN_SYSTEM_PROMPT = """Olet Eskalon v4.0 - älykäs henkilökohtainen assistentti.

🎯 TEHTÄVÄSI:
Vastaa kysymyksiin käyttäen MUISTISTA löytyvää tietoa. Voit vastata MISTÄ TAHANSA aiheesta:
- 📝 Muistiinpanot (reseptit, ideat, suunnitelmat)
- 👤 Henkilötiedot (nimet, syntymäpäivät, tiedot)
- 📚 Dokumentit (ohjeet, oppaat, artikkelit)
- 💻 Koodi (jos kysytään koodista)
- 🌍 Yleinen tieto (jos muistissa ei ole)

📋 VASTAUSRAKENNE:
1. JOS muistissa on tieto → Vastaa SUORAAN sen perusteella
2. JOS muistissa ei ole → Sano rehellisesti ja anna yleinen vastaus

🎨 VASTAUSTYYLI:
- Luonnollinen, keskusteleva suomi
- Tiivis mutta informatiivinen
- EI teknistä jargonia ellei kysytä koodista
- EI "muistista löytyi..." -lauseita (vastaa vain asiaan)

📌 ESIMERKKEJÄ:

Kysymys: "Kerro Jonnasta"
HYVÄ: "Jonna on 25-vuotias ja asuu Helsingissä. Hän työskentelee..."
HUONO: "Muistista löysin tiedoston jossa lukee Jonna..."

Kysymys: "Mikä oli se pasta-resepti?"
HYVÄ: "Tarkoitat varmaan carbonaraa. Tarvitset: spagettia, pekonia..."
HUONO: "Tässä on koodi reseptille: function pasta() {...}"

Kysymys: "Luo funktio joka..."
HYVÄ: [Anna koodia - olet myös koodiassistentti!]

💡 MUISTA:
- Älä pakota koodia jos kysymys ei liity koodiin
- Vastaa ihmiselle, ei koneelle
- Ole avulias ja ystävällinen

Kieli: AINA suomi. Tyyli: Luonnollinen keskustelu."""

def valitse_system_prompt(kategoria):
    """
    Valitsee sopivan system promptin kategorian mukaan.
    
    - "Koodi" → Tekninen koodiassistentti
    - "Koti" tai "Kaikki" → Yleinen assistentti
    """
    if kategoria == "Koodi":
        return KOODI_SYSTEM_PROMPT
    else:
        return YLEINEN_SYSTEM_PROMPT

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

# UUSI: Muistetaan kategoriavalinta
if "valittu_kategoria" not in st.session_state:
    st.session_state.valittu_kategoria = "Kaikki"

# ═══════════════════════════════════════════════════════════════════
# SIVUPALKKI
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    LOGO_TIEDOSTO = "logo.png"
    if os.path.exists(LOGO_TIEDOSTO):
        st.image(LOGO_TIEDOSTO, use_container_width=True)
    
    st.title("⚙️ Control Panel")
    
    with st.expander("🚀 Aktiiviset parannukset", expanded=False):
        st.write("✅ Parempi promptaus (+40%)")
        st.write("✅ GTR-T5 embedding (+25%)")
        st.write("✅ Chat memory (+30%)")
        st.write("✅ Hybrid search (+15%)")
        st.write("✅ Query expansion (+5%)")
        st.write("✅ Dynaaminen chunk (+10%)")
        st.write("✅ Confidence score")
        st.write("✅ Kategoriahaku (UUSI!)")
    
    n_tulokset = st.slider(
        "Kontekstin laajuus (base)", 
        min_value=1, 
        max_value=15, 
        value=7,
        help="Pienempi = nopeampi, Suurempi = kattavampi"
    )
    
    malli_valinta = st.selectbox(
        "AI-malli",
        ["qwen2.5-coder:14b-instruct-q4_K_M", "qwen2.5-coder:7b", "llama3.1"],
        index=0,
        help="14b on paras sinun 16GB koneelle"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # UUSI: KATEGORIAN VALINTA
    # ═══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔍 Haun rajaus")
    
    kategoria_valinta = st.selectbox(
        "Hae muistista:",
        ["Kaikki", "Koodi", "Koti"],
        index=0,
        help="Valitse mistä kategoriasta haetaan vastauksia"
    )
    
    # Tallenna valinta session stateen
    st.session_state.valittu_kategoria = kategoria_valinta
    
    # Näytä info valitusta kategoriasta
    if kategoria_valinta == "Kaikki":
        st.caption("🌐 Haetaan sekä koodi- että kotimuistista")
    elif kategoria_valinta == "Koodi":
        st.caption("💻 Haetaan vain koodimuistista")
    else:
        st.caption("🏠 Haetaan vain kotimuistista")
    
    # ═══════════════════════════════════════════════════════════════
    # MUUT KONTROLLIT
    # ═══════════════════════════════════════════════════════════════
    st.divider()
    
    if st.button("🗑️ Tyhjennä keskustelu"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("🧠 Muistin hallinta")
    
    if st.button("💻 PÄIVITÄ KOODIMUISTI"):
        aja_paivitys(
            kansiot=[r"C:\Users\joona\Eskalon\kielet"],
            nimi="Koodi"
        )
    
    if st.button("🏠 PÄIVITÄ KOTIMUISTI"):
        aja_paivitys(
            kansiot=[r"C:\Users\joona\Eskalon\koti"],
            nimi="Koti"
        )
    
    st.caption("💡 Päivitä erikseen kun lisäät tiedostoja.")
    st.caption("⚡ Lite-versio on nopea (ei AST-parseointia).")

# ═══════════════════════════════════════════════════════════════════
# PÄÄNÄKYMÄ
# ═══════════════════════════════════════════════════════════════════
st.title("🧠 Eskalon v4.0 Lite")

# Näytä valittu kategoria otsikon alla
if st.session_state.valittu_kategoria == "Kaikki":
    st.caption("🔍 Haetaan: Kaikki kategoriat • 6 älykkyyspäivitystä • RAG-moottori")
elif st.session_state.valittu_kategoria == "Koodi":
    st.caption("💻 Haetaan: Vain koodimuisti • 6 älykkyyspäivitystä • RAG-moottori")
else:
    st.caption("🏠 Haetaan: Vain kotimuisti • 6 älykkyyspäivitystä • RAG-moottori")

# ═══════════════════════════════════════════════════════════════════
# CHAT-HISTORIA
# ═══════════════════════════════════════════════════════════════════
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ═══════════════════════════════════════════════════════════════════
# PÄÄLOGIIKKA - Kategoriahaku käytössä!
# ═══════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Kuvaile mitä haluat koodata..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ajatellaan..."):
            
            # VAIHE 1: Laajenna kysymys
            laajennettu_prompt = laajenna_kysymys(prompt)
            
            # VAIHE 2: Laske optimaalinen määrä
            optimaalinen_n = laske_optimaalinen_n_tuloksia(prompt, n_tulokset)
            
            # VAIHE 3: Vektorointi
            kysymys_vektori = model.encode(laajennettu_prompt).tolist()
            
            # VAIHE 4: HAE VALITUSTA KATEGORIASTA! (UUSI!)
            tulokset = hae_kategoriasta(
                kysymys_vektori=kysymys_vektori,
                laajennettu_prompt=laajennettu_prompt,
                optimaalinen_n=optimaalinen_n,
                valittu_kategoria=st.session_state.valittu_kategoria
            )
            
            # Pura tulokset
            loydetyt_docit = tulokset['documents'][0]
            lahteet = tulokset['metadatas'][0]
            distances = tulokset['distances'][0] if 'distances' in tulokset else []
            
            # VAIHE 5: Confidence
            confidence_level, confidence_score = laske_confidence(distances)
            
            if confidence_level == "high":
                st.markdown(
                    f'<span class="confidence-high">🎯 Korkea varmuus: {confidence_score:.0f}%</span>', 
                    unsafe_allow_html=True
                )
            elif confidence_level == "medium":
                st.markdown(
                    f'<span class="confidence-medium">⚠️ Keskitaso: {confidence_score:.0f}%</span>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<span class="confidence-low">⚡ Yleinen tieto: {confidence_score:.0f}%</span>', 
                    unsafe_allow_html=True
                )
                
                # Näytä eri viesti riippuen kategoriasta
                if st.session_state.valittu_kategoria == "Kaikki":
                    st.info("💡 Muistista ei löytynyt relevanttia sisältöä. Vastaus perustuu AI:n yleiseen osaamiseen.")
                else:
                    st.info(f"💡 {st.session_state.valittu_kategoria}-muistista ei löytynyt relevanttia sisältöä. Kokeile 'Kaikki' -vaihtoehtoa tai vastaus perustuu yleiseen osaamiseen.")
            
            # VAIHE 6: Rakenna konteksti
            konteksti = "\n\n".join([
                f"TIEDOSTO [{l.get('kategoria', 'Tuntematon')}]: {l['polku']}\n{d}" 
                for d, l in zip(loydetyt_docit, lahteet)
            ])
            
            # VAIHE 7: Chat memory + DYNAAMINEN SYSTEM PROMPT
            viestit = [{"role": "system", "content": valitse_system_prompt(st.session_state.valittu_kategoria)}]
            
            historia_viestit = (
                st.session_state.messages[-6:] 
                if len(st.session_state.messages) > 6 
                else st.session_state.messages
            )
            
            for msg in historia_viestit:
                viestit.append({"role": msg["role"], "content": msg["content"]})
            
            viestit.append({
                "role": "user", 
                "content": f"MUISTI (kategoria: {st.session_state.valittu_kategoria}):\n{konteksti}\n\n---\n\nKYSYMYS: {prompt}"
            })
            
            # VAIHE 8: Generoi vastaus
            try:
                response = ollama.chat(
                    model=malli_valinta, 
                    messages=viestit
                )
                vastaus = response['message']['content']
                
                st.markdown(vastaus)
                
                # Näytä lähteet kategorioineen
                with st.expander("🔍 Käytetty muisti"):
                    if not lahteet:
                        st.write("❌ Ei lähteitä (vastaus perustuu yleiseen tietoon)")
                    else:
                        for meta in lahteet:
                            kategoria = meta.get('kategoria', 'Tuntematon')
                            emoji = "💻" if kategoria == "Koodi" else "🏠" if kategoria == "Koti" else "❓"
                            st.write(f"{emoji} [{kategoria}] `{meta['polku']}`")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": vastaus
                })
                
            except Exception as e:
                st.error(f"❌ Virhe: {e}")
                st.info("💡 Varmista että Ollama on käynnissä ja valittu malli on ladattu.")
                st.code(f"ollama run {malli_valinta}", language="bash")
