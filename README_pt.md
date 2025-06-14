# Protótipo de Previsor de Propriedades de Materiais

## Descrição

Este projeto é uma aplicação GUI Tkinter desenhada como um protótipo para prever propriedades de materiais a partir de Arquivos de Informação Cristalográfica (CIF) e para gerenciar um pequeno conjunto de dados de propriedades de materiais. Ele permite aos usuários:
*   Selecionar um arquivo CIF e prever propriedades usando modelos de aprendizado de máquina pré-treinados.
*   Inserir manualmente dados de materiais e salvá-los em um conjunto de dados CSV local.
*   Gerar um conjunto de dados inicial usando a API do Materials Project (requer uma chave de API).
*   Treinar modelos de aprendizado de máquina baseados no conjunto de dados gerado.

## Componentes Principais e Fluxo de Trabalho

O projeto está estruturado em vários componentes chave:
*   **Scripts de Aquisição de Dados (`fetch_mp_data.py`, `process_raw_data.py`):** Estes scripts lidam com a criação e preparação do conjunto de dados de fontes externas como o Materials Project.
*   **Script de Treinamento de Modelos (`train_model.py`):** Este script é responsável por treinar modelos de aprendizado de máquina usando o conjunto de dados processado.
*   **Script de Preparação de Dataset para GNN (`prepare_gnn_data.py`):** Este script processa dados brutos de materiais (ex: de strings CIF no output JSON de `fetch_mp_data.py`) em representações de grafo adequadas para Redes Neurais de Grafos (GNNs). Ele converte estruturas em objetos `torch_geometric.data.Data`, salva o conjunto de dados de grafos processados completo e o divide em conjuntos de treinamento, validação e teste. Isso é essencial para o desenvolvimento de modelos baseados em GNN.
*   **Aplicação GUI (`material_predictor_gui.py`):** Fornece a interface de usuário principal para interagir com os modelos de predição e gerenciar dados.
*   **Arquivo de Configuração (`config.yml`):** Um arquivo YAML central para gerenciar todas as configurações operacionais importantes, caminhos de arquivos, chaves de API e parâmetros de modelos. Isso melhora a manutenibilidade separando as configurações do código, tornando mais fácil para os usuários adaptar o projeto às suas necessidades ou diferentes ambientes sem alterar os scripts Python.
*   **Utilitários (`utils/`):** Este diretório contém módulos Python compartilhados:
    *   `config_loader.py`: Fornece uma maneira padronizada de carregar o arquivo `config.yml`, garantindo acesso consistente aos parâmetros de configuração em todos os scripts e lidando com erros potenciais como arquivos ausentes ou YAML malformado.
    *   `schema.py`: Centraliza as definições de estruturas de dados, como `DATA_SCHEMA` (usado na busca e processamento de dados para definir campos esperados e suas descrições) e `MANUAL_ENTRY_CSV_HEADERS` (usado na GUI para entrada manual de dados para garantir compatibilidade CSV). Isso evita redundância e garante consistência em como os dados são estruturados e interpretados em todo o projeto.
*   **Testes (`tests/`):** Este diretório abriga todos os arquivos de teste. Testes unitários focam em módulos e funções individuais, usando _mocking_ para isolar componentes (ex: chamadas de API, interações com o sistema de arquivos). Testes de integração verificam se diferentes partes do sistema funcionam juntas como esperado, como o pipeline completo de processamento de dados, desde a busca de dados até o treinamento de modelos.

O fluxo de trabalho geral envolve:

1.  **Configuração (Novo!):**
    *   Modificar `config.yml` para definir sua chave de API do Materials Project, caminhos de arquivos e ajustar parâmetros de modelos. (Mais detalhes na seção "Configuração" abaixo).

2.  **Aquisição de Dados (Opcional, para geração de conjunto de dados):**
    *   Primariamente desenhado para criar um conjunto de dados de compostos baseados em Fe usando a API do Materials Project.
    *   **`fetch_mp_data.py`**: Este script consulta a API do Materials Project.
        *   **Requisito**: Você **deve** fornecer sua chave de API do Materials Project. O método principal é definir `mp_api_key` no arquivo `config.yml`. Se não encontrada lá, o script verificará uma variável de ambiente chamada `MP_API_KEY`. Você pode obter uma chave registrando-se em [materialsproject.org](https://materialsproject.org).
        *   Ele busca dados brutos para materiais (padronizando para baseados em Ferro se não configurado de outra forma) e os salva em um arquivo JSON (padrão: `mp_raw_data.json`, configurável em `config.yml`).
    *   **`process_raw_data.py`**: Este script processa o arquivo de dados JSON bruto.
        *   Ele usa `pymatgen` para analisar strings CIF e extrair características estruturais.
        *   Ele combina estas com dados originados da API e salva o resultado em `Fe_materials_dataset.csv`.
    *   **`prepare_gnn_data.py` (Para Modelos GNN):** Este script pega os dados brutos (ex: `mp_raw_data.json`) e os converte em conjuntos de dados de grafos.
        *   Processa materiais em objetos `torch_geometric.data.Data`.
        *   Salva o conjunto de dados completo e os conjuntos pré-divididos de treino/validação/teste como arquivos `.pt`.
        *   A configuração para este script (caminhos de entrada/saída, proporções de divisão) é gerenciada em `config.yml` na seção `prepare_gnn_data`.
    *   **Conjunto de Dados de Exemplo**: Um `Fe_materials_dataset.csv` de exemplo está incluído no repositório. Isso permite que a GUI e o script de treinamento de modelos rodem para fins de demonstração, mesmo que você não busque imediatamente novos dados da API. Os nomes dos arquivos de entrada e saída são configuráveis via `config.yml`.

3.  **Treinamento de Modelos (`train_model.py`):**
    *   Este script carrega o conjunto de dados processado (padrão: `Fe_materials_dataset.csv`, configurável).
    *   Ele treina vários modelos de aprendizado de máquina conforme definido no script.
    *   Parâmetros de modelos (ex: tamanho do conjunto de teste, número de estimadores) e nomes dos arquivos de saída para modelos e pré-processadores são gerenciados via `config.yml`.
    *   **Uso**: `python train_model.py`

4.  **Aplicação GUI (`material_predictor_gui.py`):**
    *   Uma interface gráfica de usuário baseada em Tkinter com duas abas principais.
    *   **Aba "Prever a partir de CIF":**
        *   Permite aos usuários selecionar um arquivo CIF local.
        *   Extrai características estruturais usando `pymatgen`.
        *   Usa os modelos `.joblib` pré-treinados (carregados na inicialização) para prever propriedades: Band Gap, Energia de Formação, Metalicidade (com pontuação de confiança) e DOS no nível de Fermi (se previsto como metal).
        *   Se algum modelo/pré-processador requerido não for encontrado (ex: se `train_model.py` não tiver sido executado), as predições para essas propriedades específicas aparecerão como "N/A (modelo não carregado)".
    *   **Aba "Entrada Manual de Dados":**
        *   Fornece um formulário para inserir manualmente dados para todas as características definidas no esquema do projeto.
        *   **Botão "Carregar CIF para Extração de Características":** Permite selecionar um arquivo CIF para preencher automaticamente campos derivados do `pymatgen` (ex: fórmula, densidade, parâmetros de rede).
        *   **Botão "Salvar no Conjunto de Dados":** Anexa os dados inseridos como uma nova linha ao `Fe_materials_dataset.csv`. Isso permite aos usuários aumentar o conjunto de dados ou construir um se o acesso à API não estiver disponível.
        *   **Botão "Limpar Campos":** Reseta todos os campos de entrada.

## Configuração e Uso

1.  **Clonar o repositório:**
    ```bash
    git clone <repository_url>
    # Substitua <repository_url> pela URL real do repositório
    cd <repository_directory>
    ```

2.  **Criar um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows use: .venv\Scripts\activate
    ```

3.  **Instalar dependências:**
    Certifique-se de ter o Python 3.x instalado. Os pacotes requeridos estão listados em `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Isso inclui `pymatgen`, `scikit-learn`, `pandas`, `numpy`, `mp-api`, `joblib` e `PyYAML`.

4.  **Configurar `config.yml` (Primeiro Passo Crucial):**
    *   Abra `config.yml` em um editor de texto.
    *   **Defina sua `mp_api_key`**. Isso é essencial para `fetch_mp_data.py`.
    *   Revise outras configurações como caminhos de arquivos e parâmetros de modelos, e ajuste se necessário.

5.  **Executando a Aplicação e Fluxo de Trabalho:**
    *   **Opção A: Usar dados de exemplo e modelos pré-treinados (se fornecidos no repositório e configurados em `config.yml`)**
        1.  Certifique-se de que `config.yml` aponta para arquivos de conjunto de dados e modelos existentes se você não for treiná-los localmente.
        2.  Execute a GUI: `python material_predictor_gui.py`
        3.  Use a aba "Prever a partir de CIF" com seus próprios arquivos CIF, ou explore a aba "Entrada Manual de Dados".
    *   **Opção B: Gerar conjunto de dados e treinar modelos localmente**
        1.  **Certifique-se de que a Chave de API está definida em `config.yml`**. (Fallback para a variável de ambiente `MP_API_KEY` também é possível se `mp_api_key` em `config.yml` for um placeholder ou estiver ausente).
        2.  Execute a busca de dados: `python fetch_mp_data.py` (usa configurações de `config.yml`)
        3.  Processe dados brutos: `python process_raw_data.py` (usa configurações de `config.yml`)
        3b. Preparar dataset GNN (se usando modelos GNN): `python prepare_gnn_data.py` (usa configurações de `config.yml`)
        4.  Treine modelos: `python train_model.py` (usa configurações de `config.yml`)
        5.  Execute a GUI: `python material_predictor_gui.py` (carrega modelos e conjuntos de dados conforme `config.yml`)

## Configuração (`config.yml`)

As configurações do projeto são gerenciadas centralmente no arquivo `config.yml` localizado no diretório raiz. Este arquivo permite que você personalize vários parâmetros sem modificar os scripts diretamente, o que melhora a manutenibilidade separando as configurações do código, tornando mais fácil para os usuários adaptar o projeto às suas necessidades ou diferentes ambientes sem alterar os scripts Python.

**Configurações chave incluem:**
*   `mp_api_key`: **Sua chave de API do Materials Project. Isto é essencial para buscar dados usando `fetch_mp_data.py`.** Garantir que `mp_api_key` esteja corretamente definida neste arquivo é o primeiro e mais crucial passo para habilitar as capacidades de busca de dados.
*   `fetch_data`: Parâmetros para `fetch_mp_data.py`, como `max_total_materials` a serem buscados, `output_filename` para os dados JSON brutos, e `criteria_sets` para definir os critérios de busca no Materials Project (ex: número de elementos, elementos específicos como 'Fe'). Um valor especial de `-5` para `max_total_materials` instruirá o script a tentar buscar todos os materiais que correspondam aos critérios combinados da consulta inicial da API, ignorando os limites individuais de `limit_per_set` e o limite geral de `max_total_materials`.
*   `process_data`: Configurações para `process_raw_data.py`, incluindo `raw_data_filename` (entrada) e `output_filename` para o conjunto de dados CSV processado.
*   `train_model`: Configuração para `train_model.py`, como `dataset_filename` (CSV de entrada), `test_size` para divisão treino-teste, `random_state` para reprodutibilidade, `n_estimators` para modelos Random Forest, e caminhos para salvar `models` treinados e `preprocessors`.
*   `prepare_gnn_data`: Configurações para `prepare_gnn_data.py`.
    *   `raw_data_filename`: Arquivo JSON de entrada contendo dados brutos de materiais (ex: `mp_raw_data.json`).
    *   `processed_graphs_filename`: Caminho de saída para o arquivo contendo a lista completa de objetos `torch_geometric.data.Data` processados (ex: `data/processed/processed_graphs.pt`).
    *   `train_graphs_filename`, `val_graphs_filename`, `test_graphs_filename`: Caminhos de saída para os conjuntos de dados divididos (objetos de grafo de treinamento, validação e teste).
    *   `random_seed`: Semente inteira para divisão reprodutível do conjunto de dados.
    *   `train_ratio`, `val_ratio`, `test_ratio`: Valores de ponto flutuante para as proporções da divisão do conjunto de dados (ex: 0.7, 0.2, 0.1).
*   `gui`: Configurações para `material_predictor_gui.py`, como o `title` da aplicação, `geometry` da janela, caminhos para `models_to_load`, e `manual_entry_csv_filename` para salvar dados inseridos manualmente.

**Importante:** Antes de executar `fetch_mp_data.py` pela primeira vez, você **deve** atualizar o campo `mp_api_key` em `config.yml` com sua chave de API pessoal do Materials Project. Se esta chave não for encontrada ou estiver definida com o placeholder `"YOUR_MP_API_KEY"` em `config.yml`, o sistema então verificará a variável de ambiente `MP_API_KEY` como um fallback.

## Executando Testes

O projeto inclui um conjunto de testes unitários e de integração localizados no diretório `tests/`. Estes testes são construídos usando o framework `pytest`.

Para executar todos os testes, navegue até o diretório raiz do projeto em seu terminal e execute:
```bash
pytest
```
Isso descobrirá e executará todos os arquivos de teste (ex: `test_*.py`).
*   **Testes unitários** verificam a funcionalidade de módulos individuais (ex: carregamento de configuração de `utils/config_loader.py`, definições de esquema em `utils/schema.py`). Eles também testam a lógica central dentro de cada script, como regras de transformação de dados em `process_raw_data.py`, uso correto de parâmetros em `train_model.py` baseado na configuração, e o fluxo de trabalho de busca de dados em `fetch_mp_data.py` (simulando várias respostas de API usando mocks).
*   **Testes de integração** verificam se diferentes partes do sistema funcionam juntas corretamente, especificamente o pipeline de dados principal (`fetch_data` -> `process_data` -> `train_models`) garantindo que estes componentes passem dados corretamente (via arquivos, conforme configurado) de uma etapa para a próxima.
*   **Nota sobre Testes GUI**: A funcionalidade da GUI relacionada ao carregamento de modelos e predições é testada no nível do código (ex: garantindo que os modelos sejam carregados conforme a configuração pela lógica do script da GUI), mas testes de interação automatizados da GUI (ex: simulando cliques de botão) não estão atualmente implementados devido a desafios de ambiente `tkinter` específicos encontrados durante o desenvolvimento.

## Tratamento de Erros & Disponibilidade de Modelos
*   A GUI mostrará avisos se os arquivos de modelo (`.joblib`, caminhos configurados em `config.yml`) não forem encontrados durante a inicialização, e as predições correspondentes serão desabilitadas.
*   O script de busca de dados (`fetch_mp_data.py`) avisará se a chave de API não estiver configurada corretamente (veja a seção Configuração) e pode falhar ou recuperar dados limitados.
*   Mensagens de erro básicas são mostradas para problemas de análise de CIF ou arquivos de conjunto de dados ausentes.

## Critérios de Busca de Dados

A estratégia de aquisição de dados baseia-se em papéis específicos para cada fonte de dados:

*   **SuperCon**: Este conjunto de dados é a fonte primária para a variável alvo, que é a temperatura crítica (Tc) dos materiais supercondutores.
*   **OQMD (Open Quantum Materials Database)**: O OQMD é utilizado para obter propriedades complementares dos materiais (ex.: energia de formação, band gap, estrutura cristalina) para composições identificadas no conjunto de dados SuperCon. Serve também como uma base de dados mais ampla para obter propriedades de materiais e estruturas cristalinas para análise geral e treino de modelos.
*   **Materials Project (MP)**: A API do Materials Project é uma fonte *opcional* para adquirir propriedades complementares de materiais e estruturas cristalinas. Pode ser usada de forma semelhante ao OQMD para enriquecer o conjunto de dados ou como uma fonte alternativa para tais informações.

## Modelo GNN OracleNet

Este projeto inclui o OracleNet, um modelo de Rede Neural de Grafos (GNN) projetado para prever propriedades de materiais. O GNN utiliza estruturas de materiais representadas como grafos (onde os nós são átomos e as arestas são ligações/conexões) e aprende a prever propriedades alvo.

#### Guia Rápido para Execução (Fase II - OracleNet)

Para executar o fluxo de preparação de dados e treinamento do modelo GNN OracleNet (correspondente à Fase II do Roadmap), siga os passos abaixo. Certifique-se de que o arquivo `config.yml` está corretamente configurado, pois ambos os scripts dependem dele para seus parâmetros.

1.  **Preparar os Dados para a GNN:**
    Este script processa os dados brutos (conforme definido em `config.yml`, seção `prepare_gnn_data`) e os converte em representações de grafo que a GNN pode utilizar. Os grafos processados são salvos em arquivos `.pt`.

    ```bash
    python scripts/prepare_gnn_data.py
    ```

2.  **Treinar o Modelo GNN OracleNet:**
    Após a preparação dos dados, este script carrega os grafos de treinamento e validação para treinar o modelo OracleNet GNN. As configurações de treinamento (como taxa de aprendizado, épocas, etc.) e o caminho para salvar o modelo treinado são definidos em `config.yml` (seção `gnn_settings`).

    ```bash
    python scripts/train_gnn_model.py
    ```

Após a execução bem-sucedida desses comandos, você terá um conjunto de dados processado para a GNN e um modelo GNN treinado (`oracle_net_gnn.pth` por padrão, ou conforme especificado em `config.yml`).

#### Arquitetura do Modelo

O `OracleNetGNN` (definido em `models/gnn_oracle_net.py`) é uma Rede Neural Convolucional de Grafos (GCN) construída usando PyTorch Geometric. Sua arquitetura é projetada para processar dados de materiais baseados em grafos e prever uma única propriedade numérica.

Os componentes chave são:

*   **Camada de Entrada**: O modelo espera objetos de dados de grafo de `torch_geometric.data.Data`. Cada objeto deve conter:
    *   `x`: Matriz de características dos nós com forma `[num_nodes, num_node_features]`. Tipicamente, `num_node_features` é 2, representando o número atômico e a eletronegatividade de Pauling.
    *   `edge_index`: Conectividade do grafo em formato COO, com forma `[2, num_edges]`, tipo `torch.long`.
    *   `edge_attr`: Matriz de características das arestas, com forma `[num_edges, num_edge_features]`. Tipicamente, `num_edge_features` é 1, representando a distância interatômica. (Nota: As camadas `GCNConv` atuais usam estas como `edge_weight` se forem unidimensionais; caso contrário, podem ser ignoradas pela `GCNConv` padrão se não tratadas explicitamente).
    *   `batch`: Um vetor que atribui cada nó ao seu respectivo grafo em um lote, com forma `[num_nodes]`, tipo `torch.long`.

*   **Camadas Convolucionais de Grafo**:
    *   O modelo emprega duas camadas `GCNConv` do PyTorch Geometric.
    *   A primeira camada `GCNConv` mapeia as características de entrada dos nós para um espaço de maior dimensão (`hidden_channels`).
    *   A segunda camada `GCNConv` processa adicionalmente esses embeddings.
    *   Cada camada `GCNConv` é seguida por uma função de ativação `ReLU` para introduzir não linearidade.
    *   Se os atributos das arestas (`edge_attr`) forem unidimensionais (e.g., distâncias escalares), eles podem ser passados como `edge_weight` para as camadas `GCNConv`, influenciando a passagem de mensagens.

*   **Pooling Global**:
    *   Após as camadas convolucionais, uma operação `global_mean_pool` é aplicada. Isso agrega todos os embeddings de nós dentro de cada grafo em um lote em um único vetor de embedding em nível de grafo de tamanho `hidden_channels`. Isso permite que o modelo lide com grafos de tamanhos variados.

*   **Camada de Saída**:
    *   Uma camada de dropout (`F.dropout`) é aplicada ao embedding em nível de grafo para regularização durante o treinamento.
    *   Finalmente, uma camada linear (`torch.nn.Linear`) mapeia o embedding do grafo para um único valor de saída numérico, que é a propriedade do material prevista.

O fluxo geral de dados é:
`Lote de Grafos de Entrada -> GCNConv1 -> ReLU -> GCNConv2 -> ReLU -> Pooling Médio Global -> Dropout -> Camada Linear de Saída -> Valor(es) Previsto(s)`

### Preparação de Dados para o GNN

O desempenho eficaz de uma GNN depende de dados de grafo bem estruturados. O processo de preparação envolve a conversão de informações brutas de materiais (tipicamente dados cristalográficos e propriedades alvo) em representações de grafo adequadas para `torch_geometric`.

1.  **Entrada de Dados Brutos**:
    *   O processo começa com dados brutos de materiais, frequentemente obtidos de bancos de dados como o Materials Project ou OQMD. Espera-se que esses dados estejam tipicamente em formato JSON (e.g., `data/mp_raw_data.json` conforme configurado em `config.yml`).
    *   Cada entrada de material no arquivo JSON deve idealmente conter uma string CIF (Crystallographic Information File) e as propriedades alvo a serem previstas (e.g., gap de banda, energia de formação).

2.  **Conversão de Estrutura para Grafo (`utils/graph_utils.py`)**:
    *   O núcleo da conversão de grafo é tratado pela função `structure_to_graph` dentro de `utils/graph_utils.py`.
    *   Esta função recebe como entrada um objeto `pymatgen.core.structure.Structure` (analisado a partir da string CIF).
    *   **Extração de Características dos Nós**: Para cada átomo (sítio) na estrutura, ela extrai:
        *   `atomic_number`: O número atômico do elemento (e.g., Si é 14).
        *   `electronegativity`: A eletronegatividade de Pauling do elemento (e.g., Si é aprox. 1.90).
        Estes são montados em um vetor de características de nó para cada átomo.
    *   **Definição de Arestas e Extração de Características**:
        *   As arestas são tipicamente definidas entre átomos que estão dentro de um certo raio de corte um do outro (e.g., 3.0 Angstroms, conforme definido em `structure_to_graph`).
        *   Para cada par de átomos (aresta potencial), a `distance` (distância) interatômica real é calculada. Essa distância serve como a principal característica da aresta.
        A conectividade do grafo (`edge_index`) e as características das arestas (`edge_attr`) são construídas com base nesses critérios.

3.  **Criação do Conjunto de Dados de Grafos (`scripts/prepare_gnn_data.py`)**:
    *   O script `scripts/prepare_gnn_data.py` orquestra todo o fluxo de trabalho de preparação de dados:
        *   Ele carrega as entradas brutas de materiais do arquivo JSON especificado.
        *   Para cada material, ele analisa a string CIF em um objeto `Structure` do `pymatgen`.
        *   Em seguida, chama `structure_to_graph` para obter as características dos nós, o índice de arestas e as características das arestas.
        *   Esses componentes são usados para construir objetos `torch_geometric.data.Data`. Cada objeto `Data` representa um único grafo de material e armazena:
            *   `x`: Tensor de características dos nós (número atômico, eletronegatividade).
            *   `edge_index`: Tensor que define a conectividade do grafo.
            *   `edge_attr`: Tensor de características das arestas (distâncias).
            *   `y`: Um tensor contendo a(s) propriedade(s) alvo. Por exemplo, se estiver prevendo gap de banda e energia de formação, `y` pode ser `torch.tensor([[valor_band_gap, valor_energia_formacao]])`. O alvo específico usado durante o treinamento é determinado por `gnn_target_index` no `config.yml`.
            *   `material_id`: O identificador original do material para rastreamento.
        *   O script processa todos os materiais, pula aqueles com erros (e.g., CIFs ausentes, incapacidade de analisar) e coleta todos os objetos `Data` válidos.
    *   Finalmente, o script divide o conjunto de dados completo em conjuntos de treinamento, validação e teste com base nas proporções definidas em `config.yml` (e.g., 70% treino, 20% validação, 10% teste).
    *   Esses conjuntos de dados divididos são salvos como arquivos tensores do PyTorch (`.pt`) no diretório `data/` (e.g., `train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`), prontos para serem carregados pelos scripts de treinamento e avaliação.

Esta preparação detalhada garante que o GNN receba representações de grafo consistentes e significativas dos materiais.

### Treinando o Modelo GNN

O OracleNet GNN é treinado usando o script `scripts/train_gnn_model.py`. Este script orquestra o carregamento de dados, a execução das épocas de treinamento, a validação do modelo e o salvamento da versão com melhor desempenho.

**Execução:**
Para iniciar o processo de treinamento, execute:
```bash
python scripts/train_gnn_model.py
```

**Passos Chave no Processo de Treinamento:**

1.  **Configuração e Setup**:
    *   O script começa carregando as configurações específicas do GNN de `config.yml` sob a chave `gnn_settings`. Isso inclui caminhos de arquivo, parâmetros de aprendizado (taxa de aprendizado, tamanho do lote, épocas), detalhes da arquitetura do modelo (canais ocultos) e o `gnn_target_index`.
    *   Ele determina o dispositivo para o treinamento (CUDA se disponível, caso contrário CPU).

2.  **Carregamento de Dados e Batching (Divisão em Lotes)**:
    *   Os conjuntos de dados pré-processados de treinamento (`train_graphs.pt`) e validação (`val_graphs.pt`) são carregados do diretório `data/`. Esses arquivos contêm listas de objetos `torch_geometric.data.Data`.
    *   Instâncias de `torch_geometric.loader.DataLoader` são criadas para os conjuntos de treinamento e validação. O `DataLoader` lida com a divisão dos dados de grafo em lotes, o que é crucial para gerenciar a memória e fornecer estocasticidade ao treinamento. Ele combina múltiplos objetos `Data` em um único objeto `Batch` para processamento eficiente.

3.  **Inicialização do Modelo**:
    *   O modelo `OracleNetGNN` (de `models/gnn_oracle_net.py`) é instanciado. O número de características de nó de entrada para o modelo é determinado dinamicamente a partir do conjunto de dados carregado.
    *   O modelo é então movido para o dispositivo de computação selecionado.

4.  **Otimizador e Função de Perda (Loss Function)**:
    *   Um **otimizador Adam** (`torch.optim.Adam`) é usado para atualizar os pesos do modelo durante o treinamento. A taxa de aprendizado é configurável.
    *   A **perda de Erro Quadrático Médio (MSE)** (`torch.nn.MSELoss`) é empregada como a função de perda, adequada para tarefas de regressão onde o objetivo é prever uma propriedade numérica contínua.

5.  **Seleção da Propriedade Alvo**:
    *   Os objetos `Data` podem armazenar múltiplas propriedades alvo em seu atributo `y` (e.g., `data.y = torch.tensor([[band_gap, formation_energy]])`).
    *   O parâmetro `gnn_target_index` de `config.yml` (e.g., `0` para gap de banda, `1` para energia de formação) é usado para selecionar qual propriedade específica o modelo GNN será treinado para prever. O tensor alvo é fatiado e remodelado conforme necessário.

6.  **Loop de Treinamento**:
    *   O script itera por um número configurado de `epochs` (épocas).
    *   **Fase de Treinamento (por época)**:
        *   O modelo é definido para o modo `train()` (habilitando dropout, etc.).
        *   Ele itera através dos lotes fornecidos pelo `DataLoader` de treinamento.
        *   Para cada lote:
            *   Os gradientes do otimizador são zerados (`optimizer.zero_grad()`).
            *   Uma passagem direta (forward pass) é realizada: o lote de dados de grafo é alimentado através do modelo `OracleNetGNN` para obter predições.
            *   A perda MSE é calculada entre as predições do modelo e os valores alvo verdadeiros para o lote.
            *   Uma passagem reversa (backward pass) é realizada (`loss.backward()`), computando os gradientes da perda em relação aos parâmetros do modelo.
            *   O otimizador atualiza os parâmetros do modelo (`optimizer.step()`).
        *   A perda média de treinamento para a época é calculada e registrada.
    *   **Fase de Validação (por época)**:
        *   O modelo é definido para o modo `eval()` (desabilitando dropout, etc.).
        *   Com os cálculos de gradiente desabilitados (`torch.no_grad()`), ele itera através dos lotes do `DataLoader` de validação.
        *   Para cada lote de validação, predições são feitas e a perda é calculada.
        *   A perda média de validação para a época é calculada e registrada.

7.  **Salvamento do Modelo**:
    *   O script mantém o registro da melhor perda média de validação observada até o momento.
    *   Se a perda de validação da época atual for menor que a melhor anterior, o estado atual do modelo (`model.state_dict()`) é salvo no caminho especificado por `gnn_model_save_path` em `config.yml` (e.g., `data/oracle_net_gnn.pth`).
    *   Isso garante que o modelo salvo seja aquele que teve o melhor desempenho nos dados de validação não vistos.

Ao concluir, o script terá salvo os pesos do modelo GNN que alcançou a menor perda no conjunto de validação, pronto para avaliação.

### Avaliando o Modelo GNN

Para avaliar o desempenho do modelo GNN treinado no conjunto de teste:

```bash
python scripts/evaluate_gnn_model.py
```

- Este script carrega o modelo treinado de `data/oracle_net_gnn.pth` e os dados de teste de `data/test_graphs.pt`.
- Ele calcula e reporta métricas como Erro Absoluto Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE).
- O desempenho do GNN também é comparado com um preditor de linha de base aleatório.
- Uma análise de erro básica é realizada para mostrar as N predições com os maiores erros, ajudando a identificar áreas onde o modelo tem dificuldades.
- A configuração para avaliação (e.g., caminhos, `gnn_target_index`) também é gerenciada via `config.yml` em `gnn_settings`.

### Configuração do GNN

Todas as configurações relacionadas ao modelo OracleNet GNN, incluindo seu treinamento, avaliação e geração de dados fictícios (usados pelos scripts caso os arquivos de dados reais não sejam encontrados), são centralizadas em `config.yml` sob a chave `gnn_settings:`.

Os principais parâmetros configuráveis incluem:

*   **Caminhos dos Arquivos**:
    *   `train_graphs_path`: Caminho para o arquivo de dados do grafo de treinamento (e.g., `"data/train_graphs.pt"`).
    *   `val_graphs_path`: Caminho para o arquivo de dados do grafo de validação (e.g., `"data/val_graphs.pt"`).
    *   `test_graphs_path`: Caminho para o arquivo de dados do grafo de teste (e.g., `"data/test_graphs.pt"`).
    *   `model_save_path`: Caminho onde os pesos do modelo GNN treinado serão salvos (e.g., `"data/oracle_net_gnn.pth"`).

*   **Hiperparâmetros de Treinamento**:
    *   `learning_rate`: Taxa de aprendizado para o otimizador Adam (e.g., `0.001`).
    *   `batch_size`: Tamanho do lote para treinamento e avaliação do GNN (e.g., `32`).
    *   `epochs`: Número de épocas de treinamento para o GNN (e.g., `100`).
    *   `hidden_channels`: Número de canais ocultos nas camadas do GNN (e.g., `64`).
    *   `target_index`: Índice da variável alvo em `data.y` a ser prevista (e.g., `0` se `data.y` for `[[alvo1, alvo2]]` e `alvo1` for o desejado).

*   **Configurações de Avaliação**:
    *   `num_top_errors_to_show`: Número de principais predições com erro a serem exibidas durante a avaliação por `scripts/evaluate_gnn_model.py` (e.g., `5`).

*   **Configurações de Geração de Dados Fictícios**:
    *   Essas configurações são usadas por `scripts/train_gnn_model.py` e `scripts/evaluate_gnn_model.py` se os arquivos de dados de grafo especificados não forem encontrados, permitindo que os scripts sejam executados com dados de placeholder.
    *   `num_node_features_for_dummy_data`: Número de características de nó nos dados de grafo fictícios. Isso deve corresponder à entrada esperada do modelo GNN se estiver carregando um modelo pré-treinado (e.g., `2` para número atômico e eletronegatividade).
    *   `num_targets_in_file_for_dummy_data`: Número de propriedades alvo armazenadas no atributo `y` dos objetos `Data` de grafo fictícios. Isso deve ser consistente com `gnn_target_index` (i.e., `gnn_target_index < num_targets_in_file_for_dummy_data`). (e.g., `2` se o `y` fictício for `[[val1, val2]]`).

Revise e ajuste cuidadosamente esses parâmetros em `config.yml` conforme necessário para seu conjunto de dados e requisitos de treinamento específicos.

## Pilha Tecnológica (Stack)

Este projeto utiliza as seguintes tecnologias principais:

*   **Linguagem**: Python 3.10+
*   **Aprendizado de Máquina (Machine Learning)**:
    *   PyTorch (`torch`): Framework principal de deep learning.
    *   PyTorch Geometric (PyG) (`torch_geometric`): Biblioteca para deep learning em grafos e outras estruturas irregulares.
*   **Química/Ciência dos Materiais**:
    *   RDKit (`rdkit-pypi`): Toolkit para quimioinformática.
    *   Pymatgen (`pymatgen`): Biblioteca Python para genômica de materiais, utilizada para análise de materiais, incluindo manipulação de CIFs e estruturas.
*   **Gerenciamento de Experimentos (Planejado)**:
    *   Weights & Biases (W&B) ou MLflow: Para rastrear experimentos, modelos e conjuntos de dados. (Ainda não integrado)
*   **Controle de Versão de Dados (Planejado)**:
    *   DVC (Data Version Control): Para gerenciar arquivos de dados grandes e modelos de ML em conjunto com o Git. (Ainda não integrado)
