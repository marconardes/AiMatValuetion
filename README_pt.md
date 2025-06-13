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

### Preparação de Dados para o GNN

Antes de treinar o GNN, os dados do grafo precisam ser preparados:
- Dados brutos de materiais (e.g., do Materials Project ou OQMD contendo strings CIF e propriedades alvo) são processados em estruturas de grafo.
- O script `scripts/prepare_gnn_data.py` lida com essa conversão. Ele recebe dados brutos (e.g., `mp_raw_data.json`), converte estruturas em grafos usando `pymatgen` e `torch_geometric`, e salva os conjuntos de dados (`train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`) no diretório `data/`.
  - As características dos nós (node features) tipicamente incluem número atômico e eletronegatividade.
  - As características das arestas (edge features) podem incluir distâncias interatômicas.
  - As propriedades alvo (e.g., gap de banda, energia de formação) são armazenadas com cada grafo.

### Treinando o Modelo GNN

O modelo OracleNet GNN pode ser treinado usando o seguinte script:

```bash
python scripts/train_gnn_model.py
```

- Este script carrega os dados de grafo pré-processados de `data/train_graphs.pt` e `data/val_graphs.pt`.
- Ele instancia o modelo `OracleNetGNN` (uma arquitetura baseada em GCN definida em `models/gnn_oracle_net.py`).
- O treinamento envolve a otimização do modelo para prever uma propriedade alvo (e.g., gap de banda, especificada por `gnn_target_index` na configuração).
- O script usa um otimizador Adam e a perda de Erro Quadrático Médio (MSE).
- O modelo com a menor perda de validação é salvo em `data/oracle_net_gnn.pth` (ou conforme configurado).
- Hiperparâmetros chave de treinamento (taxa de aprendizado, tamanho do lote, épocas, canais ocultos, índice do alvo) podem ser configurados em `config.yml` na seção `gnn_settings`.

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

Todas as configurações relacionadas ao modelo GNN, treinamento, avaliação e geração de dados fictícios (usados se os arquivos de dados reais estiverem ausentes) estão localizadas em `config.yml` sob a chave `gnn_settings:`. Isso inclui caminhos de arquivo, parâmetros de aprendizado, detalhes da arquitetura do modelo (como canais ocultos) e seleção da propriedade alvo.

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
