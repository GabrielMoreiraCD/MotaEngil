'''
schemas.py - Define os modelos de dados para a triagem de serviços, utilizando Pydantic para validação e documentação.
====================================================================================================================
Este módulo contém a definição da classe `EscopoTriagem`, que representa a estrutura de dados necessária para a triagem de serviços. 
Cada campo é descrito com detalhes para garantir clareza e consistência na coleta de informações no documento inicial.
'''
from pydantic import BaseModel, Field
from typing import List, Optional


# =============================================================================
# SUB-MODELOS: Especificações de Soldagem (IEIS) e Spool List (EBP)
# =============================================================================

class IEISEspecificacoes(BaseModel):
    """Especificações técnicas extraídas do IEIS (Instrução de Execução e Inspeção de Solda)."""
    material_base_tubo: Optional[str] = Field(default=None, description="Material do tubo (ex: API 5L GrB, ASTM A106 GrB)")
    material_base_acessorios: Optional[str] = Field(default=None, description="Material de acessórios (ex: ASTM A105)")
    processo_soldagem: Optional[str] = Field(default=None, description="Processo(s) de soldagem (SMAW, GTAW, GMAW, FCAW)")
    metal_adicao: Optional[str] = Field(default=None, description="Eletrodo/arame de adição (ex: E7018, ER70S-6)")
    classificacao_aws: Optional[str] = Field(default=None, description="Classificação AWS completa (ex: AWS A5.1 E7018)")
    ndt_requerido: List[str] = Field(default_factory=list, description="Ensaios não-destrutivos requeridos (LP, VT, UT, RT, MT)")
    normas_aplicaveis: List[str] = Field(default_factory=list, description="Normas citadas no IEIS (N-XXXX)")
    classe_tubo: Optional[str] = Field(default=None, description="Classe do tubo (Classe I, II, III)")
    pre_aquecimento_min_C: Optional[str] = Field(default=None, description="Temperatura mínima de pré-aquecimento em °C")
    pwht_requerido: Optional[str] = Field(default=None, description="Pós-aquecimento (PWHT) requerido? Sim/Não")


class SpoolItem(BaseModel):
    """Um spool individual da lista de spools do EBP."""
    spool_id: Optional[str] = Field(default=None, description="Identificador do spool (ex: LP448-S01)")
    material_tubo: Optional[str] = Field(default=None, description="Material do tubo do spool (ex: API 5L GrB)")
    dn: Optional[str] = Field(default=None, description="Diâmetro nominal em polegadas (ex: '2\"')")
    schedule: Optional[str] = Field(default=None, description="Schedule do tubo (ex: SCH 40, STD)")
    comprimento_m: Optional[float] = Field(default=None, description="Comprimento real do spool em metros (ex: 2.08)")
    flange_quantidade: Optional[int] = Field(default=None, description="Número de flanges no spool")
    flange_tipo: Optional[str] = Field(default=None, description="Tipo de flange (ex: pescoço de solda, encaixe)")
    flange_classe: Optional[str] = Field(default=None, description="Classe de pressão do flange (ex: 150#, 300#)")


class IsometricExtractedSpec(BaseModel):
    """Especificação de material extraída visualmente de um isométrico por modelo de visão (Qwen2.5-VL)."""
    tipo_material: str = Field(description="Tipo: tubo_conducao|flange_pescoço|flange_encaixe|junta_espiralada|parafuso_estojo|cotovelo|tee|valvula|eletrodo_smaw|consumivel_end")
    descricao_tecnica: str = Field(description="Descrição técnica extraída do isométrico")
    quantidade: Optional[float] = Field(default=None, description="Quantidade identificada no isométrico")
    unidade: Optional[str] = Field(default=None, description="Unidade: M, UN, KG")
    diametro_nps: Optional[str] = Field(default=None, description="Diâmetro nominal em polegadas")
    schedule: Optional[str] = Field(default=None, description="Schedule/espessura")
    fonte_imagem: str = Field(description="Nome do arquivo de imagem fonte (ex: Figura 1 atualizada.jpg)")
    confianca: float = Field(default=0.8, description="Confiança da extração (0.0–1.0)")
    notas: Optional[str] = Field(default=None, description="Notas adicionais do modelo de visão")


class SubEscopo(BaseModel):
    disciplina_ou_sistema: str = Field(description="Nome genérico da disciplina ou subsistema extraído do MD (ex: Gás de Exportação, Gás Lift, Estrutura Metálica, Instrumentação)")
    tarefas: List[str] = Field(description="Lista de ações físicas a serem executadas descritas no Memorial Descritivo para este sub-escopo")
    documentos_referencia: List[str] = Field(description="Códigos exatos de documentos técnicos (ex: Isométricos IS-..., Desenhos DE-..., Listas LI-...) citados no MD para este sub-escopo")
    tags_relacionados: List[str] = Field(description="TAGs de linhas ou TIE-INs (ex: TIE-IN-003, 3/4\"-P-G2-782) citados neste sub-escopo do MD")

class EscopoTriagemUnificado(BaseModel):
    # --- MACRO: Dados do Formulário CSV ---
    # Todos os campos são Optional com defaults: a triage pode retornar resultado parcial
    # (ex: Pass 1 falha com Llama3) e o pipeline continua com o que foi extraído.
    id_servico: Optional[str] = Field(default=None, description="Código principal do serviço (ex: SS-93, LC-029)")
    titulo_servico: Optional[str] = Field(default=None, description="Título completo da atividade (item 1.6)")
    plataforma: Optional[str] = Field(default=None, description="Nome da plataforma (item 1.1)")
    ativo: Optional[str] = Field(default=None, description="Nome do ativo (item 1.2)")
    ordem_manutencao_om: Optional[str] = Field(default=None, description="Número da OM (item 1.5)")
    servico_critico: Optional[str] = Field(default=None, description="Extrair [RESPOSTA: SIM] ou [RESPOSTA: NÃO]")
    sistema: Optional[str] = Field(default=None, description="Sistema listado no item 2.1")
    local_aplat: Optional[str] = Field(default=None, description="Localização física na planta listada no item 2.2")
    tag_equipamento_principal: Optional[str] = Field(default=None, description="TAG exato do equipamento (item 2.4)")
    tag_linha_principal: Optional[str] = Field(default=None, description="TAG exato da linha principal (item 2.5)")
    documentos_referencia: List[str] = Field(default_factory=list, description="Lista de códigos dos documentos referenciados no item 2.8 e 4.1 do formulário")
    tarefas_execucao: List[str] = Field(default_factory=list, description="Lista de ações a serem executadas descritas no item 4.1 do formulário")
    notas_e_restricoes: List[str] = Field(default_factory=list, description="Condições, itens ausentes ou escopos parciais listados em NOTAS no item 4.1 do formulário")
    numero_ze: Optional[str] = Field(default=None, description="Número do projeto ZE listado no item 4.3")
    materiais_criticos: List[str] = Field(default_factory=list, description="Lista de materiais críticos identificados no item 4.8")
    servicos_simultaneos: List[str] = Field(default_factory=list, description="Lista de serviços com interdependência (item 4.10)")
    estimativa_mao_de_obra: Optional[str] = Field(default=None, description="Quantidade de mão de obra estimada no item 4.11")
    estimativa_prazo_horas: Optional[str] = Field(default=None, description="Prazo de execução estimado em horas no item 4.12")

    # --- MICRO: Dados do Memorial Descritivo (MD) ---
    normas_petrobras_aplicaveis: List[str] = Field(default_factory=list, description="Lista de normas Petrobras citadas no texto do Memorial Descritivo (ex: N-279, N-115, N-858, N-1374)")
    detalhamento_por_disciplina: List[SubEscopo] = Field(default_factory=list, description="Mapeamento detalhado extraído EXCLUSIVAMENTE do Memorial Descritivo, agrupando as tarefas e os documentos (Isométricos) por sistema ou disciplina.")

    # --- IEIS: Especificações de Soldagem (Pass 3) ---
    especificacoes_soldagem: Optional[IEISEspecificacoes] = Field(default=None, description="Especificações técnicas de soldagem extraídas do IEIS (eletrodo, processo, NDT)")

    # --- EBP: Spool List e Isométricos (Pass 4) ---
    spool_list: List[SpoolItem] = Field(default_factory=list, description="Lista de spools com comprimentos reais, extraídos do EBP/Planejamento Executivo")
    isometricos_referenciados: List[str] = Field(default_factory=list, description="Isométricos referenciados (ex: IS-2-F-B10S-200-001)")
    piping_class_referencia: Optional[str] = Field(default=None, description="Referência da piping class (ex: I-ET-3010.68-1200-200)")

    # --- Isométrico Visual: specs extraídas por modelo de visão (Pass 5) ---
    isometric_specs: List[IsometricExtractedSpec] = Field(default_factory=list, description="Especificações extraídas visualmente de isométricos via Qwen2.5-VL")


# =============================================================================
# ETAPA 3 — Consulta de Normas: especificações de materiais extraídas das normas
# =============================================================================

class EspecificacaoMaterial(BaseModel):
    """Um material ou consumível identificado explicitamente em uma norma técnica."""
    tipo_material: str = Field(description="Categoria do material: 'eletrodo_revestido' | 'arame_solda' | 'tubo_reparo' | 'consumivel_end' | 'epi' | 'outro'")
    descricao_tecnica: str = Field(description="Descrição técnica completa do material (ex: 'Eletrodo revestido baixo hidrogênio para aço carbono')")
    norma_origem: str = Field(description="Código da norma de onde foi extraído (ex: 'N-115')")
    secao_norma: str = Field(description="Seção da norma de onde foi extraído (ex: '4.3.2')")
    especificacao_aws: Optional[str] = Field(default=None, description="Classificação AWS se aplicável (ex: 'AWS A5.1 E7018')")
    especificacao_astm: Optional[str] = Field(default=None, description="Norma ASTM se aplicável (ex: 'ASTM A106 Gr B')")
    especificacao_asme: Optional[str] = Field(default=None, description="Norma ASME se aplicável (ex: 'ASME B31.3')")
    diametro_nps: Optional[str] = Field(default=None, description="Diâmetro nominal em polegadas (ex: '2')")
    schedule: Optional[str] = Field(default=None, description="Schedule/espessura (ex: 'STD', 'XS', '80')")
    pressao_classe: Optional[str] = Field(default=None, description="Classe de pressão (ex: '150#', '300#')")
    unidade: Optional[str] = Field(default=None, description="Unidade de fornecimento: 'kg' | 'm' | 'un' | 'l'")
    observacoes: Optional[str] = Field(default=None, description="Observações adicionais da norma")
    confianca: float = Field(default=1.0, description="Grau de confiança na extração (0.0–1.0)")


class NormasConsultaResult(BaseModel):
    """Resultado da Etapa 3: especificações de materiais extraídas das normas aplicáveis."""
    id_servico: Optional[str] = Field(default=None, description="Código do serviço (ex: 'SS-93')")
    normas_consultadas: List[str] = Field(description="Lista de normas que foram consultadas")
    especificacoes_extraidas: List[EspecificacaoMaterial] = Field(description="Especificações de materiais encontradas nas normas")
    chunks_utilizados: int = Field(default=0, description="Total de chunks de norma utilizados na extração")
    normas_sem_resultado: List[str] = Field(default_factory=list, description="Normas consultadas mas sem chunks encontrados no Qdrant")


# =============================================================================
# ETAPA 4 — Mapeamento de Materiais: specs → itens do catálogo
# =============================================================================

class CatalogItemMatch(BaseModel):
    """Um item do catálogo Qdrant que corresponde a uma especificação de material."""
    requisito_origem: str = Field(description="Descrição técnica do requisito que originou esta busca")
    codigo: Optional[str] = Field(default=None, description="Código do material no catálogo Petrobras")
    descricao_catalogo: str = Field(description="Descrição do item conforme consta no catálogo")
    categoria_catalogo: str = Field(description="Categoria do item (nome do arquivo XLSX de origem)")
    especificacao_catalogo: Optional[str] = Field(default=None, description="Especificação técnica do item no catálogo")
    diametro: Optional[str] = Field(default=None, description="Diâmetro nominal do item")
    material_base: Optional[str] = Field(default=None, description="Material base do item")
    norma_catalogo: Optional[str] = Field(default=None, description="Norma de referência do item no catálogo")
    unidade_fornecimento: str = Field(default="UN", description="Unidade de fornecimento do item")
    source_file: str = Field(description="Nome do arquivo XLSX de onde o item foi extraído")
    score_similaridade: float = Field(description="Score de similaridade coseno com a query (0.0–1.0)")
    mapeamento_status: str = Field(description="Status do mapeamento: 'matched' | 'partial' | 'unmapped'")
    quantidade: float = Field(default=1.0, description="Quantidade estimada necessária")
    quantidade_estimada: bool = Field(default=True, description="True se a quantidade foi estimada (não calculada do isométrico)")
    observacoes: Optional[str] = Field(default=None, description="Observações ou avisos sobre o item")


class MateriaisRequisitados(BaseModel):
    """Resultado da Etapa 4: itens do catálogo mapeados para cada especificação."""
    id_servico: Optional[str] = Field(default=None, description="Código do serviço")
    itens: List[CatalogItemMatch] = Field(description="Lista de itens do catálogo mapeados")
    total_requisitos: int = Field(description="Total de especificações recebidas da Etapa 3")
    total_mapeados: int = Field(description="Itens com mapeamento_status='matched' ou 'partial'")
    total_nao_mapeados: int = Field(description="Itens com mapeamento_status='unmapped'")


# =============================================================================
# ETAPA 5 — BOM: lista de materiais consolidada e exportável
# =============================================================================

class BOMLineItem(BaseModel):
    """Uma linha da lista de materiais."""
    item_numero: int = Field(description="Número sequencial do item na BOM")
    codigo_material: Optional[str] = Field(default=None, description="Código do material no catálogo (null se não mapeado)")
    descricao: str = Field(description="Descrição do material")
    especificacao_tecnica: str = Field(description="Especificação técnica completa (AWS, ASTM, ASME, etc.)")
    diametro_nps: Optional[str] = Field(default=None, description="Diâmetro nominal em polegadas")
    quantidade: float = Field(description="Quantidade necessária")
    unidade: str = Field(description="Unidade de medida (UN, KG, M, L)")
    quantidade_estimada: bool = Field(description="True se quantidade foi estimada")
    norma_origem: str = Field(description="Norma técnica que especifica este material")
    categoria: str = Field(description="Categoria do material")
    source_file_catalogo: str = Field(default="", description="Arquivo XLSX de origem no catálogo")
    observacoes: Optional[str] = Field(default=None, description="Observações ou avisos sobre o item")
    fonte: str = Field(default="rag_normas", description="Origem do item: 'rag_normas' | 'ieis_direto' | 'ebp_spool' | 'catalog_exact' | 'nao_fornecido'")


class ListaMateriais(BaseModel):
    """Lista de Materiais (BOM) gerada automaticamente para um serviço de engenharia."""
    id_servico: Optional[str] = Field(default=None, description="Código do serviço (ex: 'SS-93')")
    titulo_servico: Optional[str] = Field(default=None, description="Título completo do serviço")
    plataforma: Optional[str] = Field(default=None, description="Plataforma onde o serviço será executado")
    tag_equipamento: Optional[str] = Field(default=None, description="TAG do equipamento/linha principal")
    data_geracao: str = Field(description="Data/hora de geração da BOM (ISO 8601)")
    gerado_por: str = Field(default="pipeline_automatico_v1", description="Sistema que gerou a BOM")
    itens: List[BOMLineItem] = Field(description="Itens da lista de materiais")
    total_itens: int = Field(description="Total de itens na BOM")
    itens_sem_codigo: int = Field(description="Quantidade de itens sem código no catálogo (requerem revisão manual)")
    advertencias: List[str] = Field(default_factory=list, description="Avisos e alertas para o engenheiro revisor")
    normas_base: List[str] = Field(description="Normas técnicas consultadas para gerar esta BOM")