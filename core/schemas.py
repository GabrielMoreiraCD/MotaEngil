'''
schemas.py - Define os modelos de dados para a triagem de serviços, utilizando Pydantic para validação e documentação.
====================================================================================================================
Este módulo contém a definição da classe `EscopoTriagem`, que representa a estrutura de dados necessária para a triagem de serviços. 
Cada campo é descrito com detalhes para garantir clareza e consistência na coleta de informações no documento inicial.
'''
from pydantic import BaseModel, Field
from typing import List

#aqui vai ficar mockado por enquanto, depois a gente pode pensar em colocar um banco de dados ou algo do tipo para armazenar os dados de forma mais estruturada e persistente.
#o modelo nao esta identificando acentos mudar isso - docs em ptbr
from pydantic import BaseModel, Field
from typing import List

class SubEscopo(BaseModel):
    disciplina_ou_sistema: str = Field(description="Nome genérico da disciplina ou subsistema extraído do MD (ex: Gás de Exportação, Gás Lift, Estrutura Metálica, Instrumentação)")
    tarefas: List[str] = Field(description="Lista de ações físicas a serem executadas descritas no Memorial Descritivo para este sub-escopo")
    documentos_referencia: List[str] = Field(description="Códigos exatos de documentos técnicos (ex: Isométricos IS-..., Desenhos DE-..., Listas LI-...) citados no MD para este sub-escopo")
    tags_relacionados: List[str] = Field(description="TAGs de linhas ou TIE-INs (ex: TIE-IN-003, 3/4\"-P-G2-782) citados neste sub-escopo do MD")

class EscopoTriagemUnificado(BaseModel):
    # --- MACRO: Dados do Formulário CSV (Estrutura Validada) ---
    id_servico: str = Field(description="Código principal do serviço (ex: LC-029)")
    titulo_servico: str = Field(description="Título completo da atividade (item 1.6)")
    plataforma: str = Field(description="Nome da plataforma (item 1.1)")
    ativo: str = Field(description="Nome do ativo (item 1.2)")
    ordem_manutencao_om: str = Field(description="Número da OM (item 1.5)")
    servico_critico: str = Field(description="Extrair [RESPOSTA: SIM] ou [RESPOSTA: NÃO]")
    sistema: str = Field(description="Sistema listado no item 2.1")
    local_aplat: str = Field(description="Localização física na planta listada no item 2.2")
    tag_equipamento_principal: str = Field(description="TAG exato do equipamento (item 2.4)")
    tag_linha_principal: str = Field(description="TAG exato da linha principal (item 2.5)")
    documentos_referencia: List[str] = Field(description="Lista de códigos dos documentos referenciados no item 2.8 e 4.1 do formulário")
    tarefas_execucao: List[str] = Field(description="Lista de ações a serem executadas descritas no item 4.1 do formulário")
    notas_e_restricoes: List[str] = Field(description="Condições, itens ausentes ou escopos parciais listados em NOTAS no item 4.1 do formulário")
    numero_ze: str = Field(description="Número do projeto ZE listado no item 4.3")
    materiais_criticos: List[str] = Field(description="Lista de materiais críticos identificados no item 4.8")
    servicos_simultaneos: List[str] = Field(description="Lista de serviços com interdependência (item 4.10)")
    estimativa_mao_de_obra: str = Field(description="Quantidade de mão de obra estimada no item 4.11")
    estimativa_prazo_horas: str = Field(description="Prazo de execução estimado em horas no item 4.12")

    # --- MICRO: Dados do Memorial Descritivo (MD) ---
    normas_petrobras_aplicaveis: List[str] = Field(description="Lista de normas Petrobras citadas no texto do Memorial Descritivo (ex: N-279, N-115, N-858, N-1374)")
    detalhamento_por_disciplina: List[SubEscopo] = Field(description="Mapeamento detalhado extraído EXCLUSIVAMENTE do Memorial Descritivo, agrupando as tarefas e os documentos (Isométricos) por sistema ou disciplina.")