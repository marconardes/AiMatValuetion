# Protótipo de Previsor de Propriedades de Materiais

## Descrição

Este projeto é uma aplicação GUI Tkinter desenhada como um protótipo para prever propriedades de materiais a partir de Arquivos de Informação Cristalográfica (CIF). Ele permite que os usuários selecionem um arquivo CIF, visualizem alguns dados básicos extraídos do material e vejam previsões de placeholder para propriedades eletrônicas chave.

## Funcionalidades Atuais

*   **Seleção de Arquivo CIF:** Usuários podem navegar e selecionar um arquivo `.cif` do seu sistema local.
*   **Extração de Dados do Material:** Utiliza a biblioteca `pymatgen` para analisar o arquivo CIF selecionado e extrai:
    *   Fórmula Química (reduzida)
    *   Densidade
    *   Volume da Célula Unitária
*   **Previsões Placeholder:** Exibe valores placeholder para:
    *   Gap de Energia (Band Gap)
    *   Densidade de Estados (DOS)
    *   Energia de Formação
    (Observação: Estas previsões são atualmente baseadas em uma consulta a dados pré-definidos usando a fórmula química do material e são apenas para fins de demonstração.)
*   **Tratamento de Erros:** Mensagens de erro básicas são exibidas se um arquivo CIF não puder ser analisado.

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
    Certifique-se de que você tem o Python 3.x instalado.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    python material_predictor_gui.py
    ```
    Isso iniciará a GUI Tkinter. Você poderá então selecionar um arquivo CIF e clicar em "Prever". Arquivos CIF de exemplo não são fornecidos no repositório, então você precisará usar os seus próprios.
