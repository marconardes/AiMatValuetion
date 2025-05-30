# Protótipo de Previsor de Propriedades de Materiais

## Descrição

Este projeto é uma aplicação GUI Tkinter desenhada como um protótipo para prever propriedades de materiais a partir de Arquivos de Informação Cristalográfica (CIF) e para gerenciar um pequeno conjunto de dados de propriedades de materiais. Ele permite aos usuários:
*   Selecionar um arquivo CIF e prever propriedades usando modelos de machine learning pré-treinados.
*   Inserir manualmente dados de materiais e salvá-los em um conjunto de dados CSV local.
*   Gerar um conjunto de dados inicial usando a API do Materials Project (requer uma chave de API).
*   Treinar modelos de machine learning com base no conjunto de dados gerado.

## Componentes Principais e Fluxo de Trabalho

1.  **Aquisição de Dados (Opcional, para geração do conjunto de dados):**
    *   Projetado primariamente para criar um conjunto de dados de compostos baseados em Ferro (Fe) usando a API do Materials Project.
    *   **`fetch_mp_data.py`**: Este script consulta a API do Materials Project.
        *   **Requisito**: Você **deve** definir uma variável de ambiente chamada `MP_API_KEY` com sua chave de API válida do Materials Project. Você pode obter uma chave registrando-se em [materialsproject.org](https://materialsproject.org).
        *   Ele busca dados brutos para materiais contendo Ferro (Fe) e os salva em `mp_raw_data.json`.
    *   **`process_raw_data.py`**: Este script processa o `mp_raw_data.json`.
        *   Ele usa `pymatgen` para analisar strings CIF e extrair características estruturais.
        *   Combina essas características com dados obtidos da API e salva o resultado em `Fe_materials_dataset.csv`.
    *   **Conjunto de Dados Placeholder**: Um arquivo `Fe_materials_dataset.csv` de exemplo está incluído no repositório. Isso permite que a GUI e o script de treinamento de modelos funcionem para fins de demonstração, mesmo que você não busque dados novos da API imediatamente.

2.  **Treinamento de Modelos (`train_model.py`):**
    *   Este script carrega o `Fe_materials_dataset.csv`.
    *   Ele treina vários modelos de machine learning para prever propriedades de materiais:
        *   Gap de Energia (Band Gap - Regressor)
        *   Energia de Formação por Átomo (Regressor)
        *   Metalicidade (É Metal - Classificador)
        *   Densidade de Estados (DOS) no Nível de Fermi (Regressor, apenas para metais)
    *   O script realiza pré-processamento básico (imputação, escalonamento, one-hot encoding) e salva os modelos treinados e os pré-processadores como arquivos `.joblib`:
        *   `model_target_band_gap.joblib`
        *   `model_target_formation_energy.joblib`
        *   `model_target_is_metal.joblib`
        *   `model_dos_at_fermi.joblib`
        *   `preprocessor_main.joblib` (para características gerais)
        *   `preprocessor_dos_at_fermi.joblib` (especificamente para características do modelo DOS)
    *   **Uso**: `python train_model.py`

3.  **Aplicação GUI (`material_predictor_gui.py`):**
    *   Uma interface gráfica de usuário baseada em Tkinter com duas abas principais.
    *   **Aba "Prever a partir de CIF":**
        *   Permite aos usuários selecionar um arquivo CIF local.
        *   Extrai características estruturais usando `pymatgen`.
        *   Usa os modelos `.joblib` pré-treinados (carregados na inicialização) para prever propriedades: Gap de Energia, Energia de Formação, Metalicidade (com pontuação de confiança) e DOS no Nível de Fermi (se previsto como metal).
        *   Se algum modelo/pré-processador necessário não for encontrado (ex.: se `train_model.py` não foi executado), as previsões para essas propriedades específicas aparecerão como "N/A (modelo não carregado)".
    *   **Aba "Entrada Manual de Dados":**
        *   Fornece um formulário para inserir manualmente dados para todas as características definidas no esquema do projeto.
        *   Botão **"Carregar CIF para Extração de Características"**: Permite selecionar um arquivo CIF para preencher automaticamente os campos derivados pelo `pymatgen` (ex.: fórmula, densidade, parâmetros de rede).
        *   Botão **"Salvar no Conjunto de Dados"**: Adiciona os dados inseridos como uma nova linha ao `Fe_materials_dataset.csv`. Isso permite aos usuários aumentar o conjunto de dados ou construir um se o acesso à API não estiver disponível.
        *   Botão **"Limpar Campos"**: Reinicia todos os campos de entrada.

## Configuração e Uso

1.  **Clone o repositório:**
    ```bash
    git clone <url_do_repositorio>
    # Substitua <url_do_repositorio> pela URL real do repositório
    cd <diretorio_do_repositorio>
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows use: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    Certifique-se de que você tem o Python 3.x instalado. Os pacotes necessários estão listados em `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Isso inclui `pymatgen`, `scikit-learn`, `pandas`, `numpy`, `mp-api` e `joblib`.

4.  **Executando a Aplicação e Fluxo de Trabalho:**
    *   **Opção A: Usar dados de exemplo e modelos pré-treinados (se fornecidos)**
        1.  O repositório pode incluir um `Fe_materials_dataset.csv` de exemplo e arquivos `.joblib` pré-treinados.
        2.  Execute a GUI: `python material_predictor_gui.py`
        3.  Use a aba "Prever a partir de CIF" com seus próprios arquivos CIF ou explore a aba "Entrada Manual de Dados".
    *   **Opção B: Gerar conjunto de dados e treinar modelos localmente**
        1.  **Defina a Chave de API (Crucial para `fetch_mp_data.py`):**
            Defina a variável de ambiente `MP_API_KEY`:
            ```bash
            # Linux/macOS
            export MP_API_KEY="SUA_CHAVE_DE_API_REAL"
            # Windows Command Prompt
            set MP_API_KEY="SUA_CHAVE_DE_API_REAL"
            # Windows PowerShell
            $Env:MP_API_KEY="SUA_CHAVE_DE_API_REAL"
            ```
        2.  Execute a busca de dados: `python fetch_mp_data.py`
        3.  Processe os dados brutos: `python process_raw_data.py` (Isso cria/atualiza `Fe_materials_dataset.csv`)
        4.  Treine os modelos: `python train_model.py` (Isso cria os arquivos de modelo `.joblib`)
        5.  Execute a GUI: `python material_predictor_gui.py`

## Tratamento de Erros e Disponibilidade dos Modelos
*   A GUI exibirá avisos se os arquivos de modelo (`.joblib`) não forem encontrados durante a inicialização, e as previsões correspondentes serão desabilitadas.
*   O script de busca de dados (`fetch_mp_data.py`) avisará se a `MP_API_KEY` não estiver definida e poderá falhar ou recuperar dados limitados.
*   Mensagens de erro básicas são exibidas para problemas de análise de CIF ou arquivos de conjunto de dados ausentes.

[end of README_pt.md]
