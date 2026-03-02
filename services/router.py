'''
router.py - Roteador de Consultas para Classificação de Intenções
============================================================================================================
Este módulo implementa a função `route_query`, que classifica a intenção de uma pergunta técnica para determinar a 
estratégia de filtragem vetorial a ser aplicada no sistema de recuperação de documentos.
A função utiliza um modelo de linguagem para analisar a pergunta e categorizar se a resposta exigirá principalmente informações extraídas de imagens/diagramas, 
texto descritivo ou ambos.
'''
import logging

logger = logging.getLogger(__name__)

def route_query(query: str, llm) -> str:
    """
    Classifica a intenção da busca para definir a estratégia de filtragem vetorial no Qdrant.
    """
    routing_prompt = f"""Analise a seguinte pergunta e determine se a resposta exigirá primariamente consultar:
    1 - Uma figura, diagrama, imagem ou tabela gráfica.
    2 - O texto descritivo e normas escritas.
    3 - Ambos.
    
    Responda EXCLUSIVAMENTE com uma destas três palavras: IMAGEM, TEXTO ou AMBOS.
    Pergunta: {query}
    Classificação:"""
    
    response = llm.invoke(routing_prompt).strip().upper()
    logger.debug(f"Resposta bruta do roteador: {response}")
    
    if "IMAGEM" in response or "FIGURA" in response:
        return "imagem_descrita"
    elif "TEXTO" in response or "TABELA" in response:
        return "texto_extraido"
    else:
        return "ambos"