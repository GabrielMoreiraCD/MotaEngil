"""
main.py - Interface de Linha de Comando para Consulta ao Sistema RAG Multimodal
============================================================================================================
Este módulo implementa uma interface de linha de comando (CLI) para interagir com o Motor de Recuperação Augmentada (RAG) desenvolvido. A CLI permite que os usuários façam perguntas técnicas relacionadas a normas e documentos, e recebem respostas geradas com base em informações extraídas de texto e imagens. 
O sistema é projetado para ser robusto, com tratamento de erros e logging detalhado para facilitar a depuração e monitoramento.
"""
import sys
import logging
from services.rag_engine import RAGEngine

# Configuração formal de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAG_CLI")

def main():
    logger.info("Iniciando aplicação...")
    
    try:
        engine = RAGEngine()
    except Exception as e:
        logger.critical(f"Falha na inicialização do sistema: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("SISTEMA RAG MULTIMODAL PRONTO PARA CONSULTA")
    print("Digite 'sair' a qualquer momento para encerrar.")
    print("="*60 + "\n")

    while True:
        try:
            query = input("\n[Usuário] Digite sua pergunta: ")
            
            if query.strip().lower() in ['sair', 'exit', 'quit', 'encerrar']:
                logger.info("Encerrando o sistema...")
                break
                
            if not query.strip():
                continue

            result = engine.process_query(query)
            
            print("\n--- CHUNKS RECUPERADOS ---")
            for i, doc in enumerate(result["docs"]):
                tipo = doc.metadata.get('tipo', 'desconhecido')
                pagina = doc.metadata.get('page', 'N/A')
                print(f"Chunk {i+1} [Tipo: {tipo}] - Página {pagina}")
            print("--------------------------\n")
            
            print(f"[Assistente Técnica]:\n{result['answer']}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            logger.info("Interrupção manual detectada. Encerrando.")
            break
        except Exception as e:
            logger.error(f"Erro Crítico durante o processamento: {e}", exc_info=True)

if __name__ == "__main__":
    main()