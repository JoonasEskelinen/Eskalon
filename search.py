import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ═══════════════════════════════════════════════════════════════════
# ESKALON v4.0
# ═══════════════════════════════════════════════════════════════════

print("📦 Ladataan koodimuistia...")
client = chromadb.PersistentClient(path="./koodimuisti_db")
collection = client.get_collection(name="omat_koodit")

print("🧠 Ladataan GTR-T5 embedding-mallia...")
model = SentenceTransformer('sentence-transformers/gtr-t5-base')

# ═══════════════════════════════════════════════════════════════════
# ULTRA SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════
ULTRA_SYSTEM_PROMPT = """Olet Eskalon v4.0 - huippuluokan koodiassistentti tekoälyllä.

🎯 TOIMINTAMALLI:
1. ANALYSOI konteksti syvällisesti
2. Käytä olemassa olevan koodin tyyliä
3. Anna täydellinen, suoraan käyttövalmis koodi

📋 VASTAUSRAKENNE:
🎯 RATKAISU: (1 lause)
💻 KOODI: (Kokonainen toimiva esimerkki)
⚡ OLENNAISET: (Max 3 huomiota)

Kieli: Suomi. Tyyli: Tekninen ammattilainen."""

# ═══════════════════════════════════════════════════════════════════
# QUERY EXPANSION
# ═══════════════════════════════════════════════════════════════════
def laajenna_kysymys(kysymys):
    """Laajentaa kysymystä synonyymein."""
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
# CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════
def laske_confidence(distances):
    """Laskee varmuusasteen hakutuloksista."""
    if not distances or len(distances) == 0:
        return "❓ Tuntematon", 0.0
    
    avg_distance = sum(distances) / len(distances)
    
    if avg_distance < 0.5:
        return "🎯 KORKEA", (1 - avg_distance) * 100
    elif avg_distance < 1.0:
        return "⚠️ KESKITASO", (1 - avg_distance) * 100
    else:
        return "⚡ MATALA", max(0, (1 - avg_distance) * 100)

# ═══════════════════════════════════════════════════════════════════
# PÄÄOHJELMA
# ═══════════════════════════════════════════════════════════════════
def kysy_tekoalylta():
    print("\n" + "="*60)
    print("🧠 ESKALON v4.0 LITE - COMMAND LINE")
    print("="*60)
    print("Parannukset: GTR-T5 • Hybrid Search • Query Expansion • Confidence")
    print("="*60 + "\n")
    
    historia = []
    
    while True:
        kysymys = input("💬 Kysy koodistasi (tai 'q' poistuaksesi): ")
        if kysymys.lower() == 'q':
            print("\n👋 Näkemiin!")
            break
        
        print("\n🔍 Etsitään vastaavuuksia muistista...")
        
        # Query expansion
        laajennettu = laajenna_kysymys(kysymys)
        if laajennettu != kysymys:
            print(f"   └─ Laajennettu haku: '{laajennettu[:50]}...'")
        
        # Vektorointi
        kysymys_vektori = model.encode(laajennettu).tolist()
        
        # HYBRID SEARCH
        tulokset = collection.query(
            query_texts=[laajennettu],
            query_embeddings=[kysymys_vektori],
            n_results=5
        )
        
        loydetyt_dokumentit = tulokset['documents'][0]
        lahteet = tulokset['metadatas'][0]
        distances = tulokset['distances'][0] if 'distances' in tulokset else []
        
        # Confidence
        confidence_text, confidence_score = laske_confidence(distances)
        print(f"\n{confidence_text} varmuus: {confidence_score:.0f}%")
        
        # Näytä lähteet
        print("\n📚 LÖYDETYT LÄHTEET:")
        for i, meta in enumerate(lahteet, 1):
            print(f"{i}. {meta.get('nimi', 'Tuntematon')}")
            print(f"   └─ {meta.get('polku', 'N/A')}")
        
        # Rakennetaan konteksti
        konteksti = "\n\n".join([f"TIEDOSTO: {l['polku']}\nSISÄLTÖ:\n{d}" for d, l in zip(loydetyt_dokumentit, lahteet)])
        
        # CHAT MEMORY
        viestit = [{"role": "system", "content": ULTRA_SYSTEM_PROMPT}]
        
        for h_msg in historia[-4:]:
            viestit.append(h_msg)
        
        viestit.append({
            "role": "user",
            "content": f"KOODIMUISTI:\n{konteksti}\n\n---\n\nKYSYMYS: {kysymys}"
        })
        
        print("\n🧠 Generoidaan vastausta (Qwen 2.5 Coder)...")
        
        try:
            vastaus = ollama.chat(model='qwen2.5-coder:7b', messages=viestit)
            
            print("\n" + "─"*60)
            print("VASTAUS:")
            print("─"*60)
            print(vastaus['message']['content'])
            print("─"*60 + "\n")
            
            # Päivitä historia
            historia.append({"role": "user", "content": kysymys})
            historia.append({"role": "assistant", "content": vastaus['message']['content']})
            
        except Exception as e:
            print(f"\n❌ Virhe: {e}")
            print("💡 Varmista että Ollama on käynnissä.")

if __name__ == "__main__":
    kysy_tekoalylta()
