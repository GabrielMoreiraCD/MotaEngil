# audit_teoria.py — roda em 2 minutos
import os
import re
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))

# Amostra 200 chunks aleatórios
results = client.scroll(
    collection_name="petrobras_rag_teoria",
    limit=6618,
    with_payload=True,
    with_vectors=False,
)[0]

# Métricas de qualidade
ocr_noise_pattern = re.compile(r'[a-zA-Z]{1}[^a-zA-ZáéíóúàãõâêôçÁÉÍÓÚÀÃÕÂÊÔÇ\s\d]{3,}')
total = len(results)
noisy = 0
english_only = 0
short_chunks = 0

for r in results:
    texto = r.payload.get("texto", "")
    
    # Detecta ruído de OCR: sequências sem vogais, caracteres misturados
    noise_hits = len(ocr_noise_pattern.findall(texto))
    if noise_hits > 3:
        noisy += 1
    
    # Detecta chunks predominantemente em inglês (livros importados)
    pt_words = len(re.findall(r'\b(de|da|do|para|com|que|em|tubulação|norma|pressão)\b', texto, re.I))
    en_words = len(re.findall(r'\b(the|of|and|for|with|pressure|vessel|design)\b', texto, re.I))
    if en_words > pt_words and en_words > 2:
        english_only += 1
    
    if len(texto) < 100:
        short_chunks += 1

print(f"Total amostrado    : {total}")
print(f"Com ruído OCR      : {noisy} ({100*noisy/total:.1f}%)")
print(f"Predomin. inglês   : {english_only} ({100*english_only/total:.1f}%)")
print(f"Chunks < 100 chars : {short_chunks} ({100*short_chunks/total:.1f}%)")
print(f"\nDecisão recomendada:")
if noisy/total > 0.3:
    print("  → REINGERIR com OCR pipeline melhorado")
elif english_only/total > 0.5:
    print("  → SEPARAR em coleção petrobras_rag_en e petrobras_rag_pt")
else:
    print("  → Apenas limpeza de chunks curtos por scroll+delete")