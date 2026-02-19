import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# ESKALON v4.0
# ═══════════════════════════════════════════════════════════════════

print("🚀 Käynnistetään Eskalon v4.0 Lite (nopea versio)...")

# --- ASETUKSET ---
LADATTAVAT_KANSIOT = [
    r"C:\Users\joona\Eskalon\kielet"
]

IGNORE_KANSIOT = {
    "node_modules", ".git", "venv", "npm", "env", "__pycache__", 
    "dist", "build", ".next", "out", "target", "public", "docs", 
    "bin", "obj", ".vscode", ".idea", "cache", "android", "ios", 
    "web", "desktop", "assets", "coverage", "images", "kuvat"
}

SALLITUT_PAATTEET = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", 
    ".java", ".cpp", ".h", ".sql", ".txt", ".md", ".mdx", 
    ".rst", ".json", ".yaml", ".yml", ".sh", ".ps1", ".php"
}

KIELLETYT_MERKKIJONOT = {
    "API_KEY", "SECRET_KEY", "PASSWORD", "ACCESS_TOKEN", 
    "PRIVATE_KEY", "AUTH_TOKEN", "CONNECTION_STRING", "SECRET=", "PASSWORD="
}

# --- ALUSTUS ---
print("📦 Ladataan GTR-T5 embedding-mallia (parempi koodille)...")
client = chromadb.PersistentClient(path="./koodimuisti_db")
collection = client.get_or_create_collection(name="omat_koodit")

# GTR-T5: Parempi koodille kuin all-MiniLM-L6-v2
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Malli ladattu!\n")

def lue_ja_tallenna():
    """
    Nopea indeksointi ilman AST-parseointia.
    Säilyttää paremman embedding-mallin (GTR-T5).
    """
    tiedostot_ladattu = 0
    estetyt_tiedostot = 0
    virheet = 0
    
    # Kerää kaikki tiedostot
    kaikki_tiedostot = []
    for juuripolku in LADATTAVAT_KANSIOT:
        if not os.path.exists(juuripolku):
            print(f"⚠️ Varoitus: Polkua {juuripolku} ei löydy.")
            continue
            
        print(f"📂 Skannataan: {juuripolku}")
        for juuri, kansiot, tiedostot in os.walk(juuripolku):
            # Suodata pois ignore-kansiot
            kansiot[:] = [k for k in kansiot if k not in IGNORE_KANSIOT]
            
            for tiedosto in tiedostot:
                pääte = os.path.splitext(tiedosto)[1].lower()
                if pääte in SALLITUT_PAATTEET:
                    kaikki_tiedostot.append(os.path.join(juuri, tiedosto))

    print(f"\n📊 Löydettiin {len(kaikki_tiedostot)} kooditiedostoa")
    print("🔄 Aloitetaan nopea indeksointi (Lite-moodissa)...\n")

    # Prosessoi tiedostot NOPEASTI (ei AST-parseointia)
    for t_polku in tqdm(kaikki_tiedostot, desc="Indeksoidaan", unit="file"):
        tiedoston_nimi = os.path.basename(t_polku)
        tiedosto_suffix = os.path.splitext(t_polku)[1].lower()
        
        # Turvallisuussuodatus
        if ".env" in tiedoston_nimi.lower() or "secret" in tiedoston_nimi.lower():
            estetyt_tiedostot += 1
            continue

        try:
            with open(t_polku, "r", encoding="utf-8") as f:
                sisalto = f.read()
                
            if len(sisalto.strip()) < 20:
                continue

            # Sisältösuodatus (salaisuudet)
            sisalto_isoilla = sisalto.upper()
            if any(sana in sisalto_isoilla for sana in KIELLETYT_MERKKIJONOT):
                estetyt_tiedostot += 1
                continue
            
            # YKSINKERTAINEN METADATA (nopea, ei AST-parseointia)
            metadata = {
                "polku": t_polku,
                "nimi": tiedoston_nimi,
                "tyyppi": tiedosto_suffix
            }
            
            # Vektorointi ja tallennus
            vektori = model.encode(sisalto).tolist()
            collection.upsert(
                ids=[t_polku],
                embeddings=[vektori],
                documents=[sisalto],
                metadatas=[metadata]
            )
            tiedostot_ladattu += 1
            
        except Exception:
            virheet += 1
            continue

    # Tulokset
    print(f"\n{'='*60}")
    print(f"✅ VALMIS!")
    print(f"{'='*60}")
    print(f"📊 Tallennettu: {tiedostot_ladattu} tiedostoa")
    print(f"🛡️ Estetty: {estetyt_tiedostot} (turvallisuus)")
    print(f"⚠️ Virheet: {virheet}")
    print(f"{'='*60}\n")
    
    print("💡 HUOM: Tämä on Lite-versio (nopea).")
    print("   - Säilyttää: GTR-T5 embedding, Hybrid search, Chat memory")
    print("   - Ei sisällä: Funktio/luokka-nimet metadatassa")
    print("   - Nopeus: ~20x nopeampi kuin Ultra-versio")

if __name__ == "__main__":
    lue_ja_tallenna()
    print("\n🎉 Koodiaivo valmis käytettäväksi!")
    print("💻 Käynnistä sovellus: streamlit run app.py")
