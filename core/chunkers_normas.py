# core/chunkers_normas.py
from core.extractors_normas import NormaChunk

CHUNK_MAX_TOKENS = 400   # ~300 palavras — cabe no contexto sem diluir
CHUNK_OVERLAP    = 80    # tokens de sobreposição entre chunks adjacentes

def _estimate_tokens(text: str) -> int:
    # Aproximação: 1 token ≈ 4 chars para PT-BR técnico
    return len(text) // 4

def build_contextual_text(chunk: NormaChunk) -> str:
    """
    Injeta contexto hierárquico no início do texto para embedding.
    Isso aumenta precision@k sem alterar o texto original armazenado.
    
    Fórmula: context_prefix + separador + chunk_text
    O vetor resultante = embedding(context + conteúdo) captura
    tanto a localização estrutural quanto o conteúdo semântico.
    """
    prefix = (
        f"[Norma: {chunk.norma_id}] "
        f"[Seção: {chunk.secao} — {chunk.titulo_secao}] "
        f"[Tipo: {chunk.tipo}]\n"
    )
    return prefix + chunk.texto

def chunk_norma_chunks(raw_chunks: list[NormaChunk]) -> list[dict]:
    """
    Converte NormaChunks em dicts prontos para upsert no Qdrant.
    Aplica sliding window apenas em chunks de texto longo.
    Tabelas nunca são quebradas — são atômicas.
    """
    output = []

    for rc in raw_chunks:
        token_count = _estimate_tokens(rc.texto)

        if rc.tipo in ("tabela", "figura", "nota") or token_count <= CHUNK_MAX_TOKENS:
            # Atômico: nunca quebrar tabelas/figuras
            output.append({
                "texto": rc.texto,
                "texto_embedding": build_contextual_text(rc),  # usado só para embed
                "norma_id": rc.norma_id,
                "secao": rc.secao,
                "titulo_secao": rc.titulo_secao,
                "tipo": rc.tipo,
                "pagina": rc.pagina,
                "tags": rc.tags,
            })
        else:
            # Sliding window para textos longos
            words = rc.texto.split()
            step = CHUNK_MAX_TOKENS - CHUNK_OVERLAP
            i = 0
            part = 0
            while i < len(words):
                window = words[i : i + CHUNK_MAX_TOKENS]
                chunk_text = " ".join(window)
                rc_part = NormaChunk(
                    norma_id=rc.norma_id,
                    secao=rc.secao,
                    titulo_secao=rc.titulo_secao,
                    tipo=rc.tipo,
                    texto=chunk_text,
                    pagina=rc.pagina,
                    tags=rc.tags,
                )
                output.append({
                    "texto": chunk_text,
                    "texto_embedding": build_contextual_text(rc_part),
                    "norma_id": rc.norma_id,
                    "secao": rc.secao,
                    "titulo_secao": rc.titulo_secao,
                    "tipo": rc.tipo,
                    "pagina": rc.pagina,
                    "tags": rc.tags,
                    "part": part,
                })
                i += step
                part += 1

    return output