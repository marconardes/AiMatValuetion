# Roadmap de Desenvolvimento do Modelo de IA: Status Atual

Este documento acompanha o progresso do projeto em relação ao roadmap inicial para construir um modelo de IA para prever propriedades de materiais.

- `[x]` Implementado
- `[~]` Parcialmente Implementado ou Versão Básica Existente
- `[ ]` Não Implementado

---

## 1. Definir o Problema e Escopo:

- `[x]` **Qual propriedade eletrônica específica você quer prever?** (Implementado: Gap de energia, características da densidade de estados, energia de formação. Estabilidade não diretamente, embora a energia de formação esteja relacionada.)
- `[~]` **Para qual classe de materiais?** (Anotação: Conjunto de dados inicial focado em compostos baseados em Fe do Materials Project. A ferramenta GUI em si é genérica para qualquer CIF.)
- `[ ]` **Qual é o nível de precisão desejado e o custo computacional aceitável?** (Anotação: A precisão é avaliada, mas nenhum alvo específico como "80%" foi definido ou otimizado.)

## 2. Aquisição e Preparação de Dados:

- `[x]` **Fonte de Dados: Bancos de Dados Públicos** (Anotação: Implementada a busca na API do Materials Project via `fetch_mp_data.py`. AFLOW foi discutido, mas não implementado como fonte direta.)
- `[~]` **Fonte de Dados: Seus Próprios Cálculos** (Anotação: Cálculos DFT diretos pelo agente não são viáveis. No entanto, a aba "Entrada Manual de Dados" na GUI permite aos usuários inserir dados de seus próprios cálculos.)
- `[~]` **Limpeza e Validação:** (Anotação: O tratamento básico de NaN é feito no script de treinamento. Nenhum método abrangente de limpeza ou validação de dados foi implementado ainda.)
- `[~]` **Volume de Dados:** (Anotação: O framework busca ~50 materiais para demonstração. Os scripts podem ser adaptados para mais, e a entrada manual é possível. O conjunto de dados atual não é de "grande volume" no contexto de aprendizado profundo.)
- `[x]` **Controle Aprimorado de Busca de Dados:** Implementada uma opção para buscar todos os materiais disponíveis que correspondam aos critérios definidos, configurando `max_total_materials` para `-5` no `config.yml`, contornando os limites de busca padrão.

## 3. Engenharia de Atributos (Descritores de Materiais):

- `[x]` **Esta é uma das etapas mais cruciais. Você precisa converter as informações do material (composição química, estrutura cristalina) em um formato numérico que o modelo de IA possa entender.** (Anotação: Implementado via `process_raw_data.py` usando `pymatgen`.)
- `[x]` **Tipos de Atributos: Baseados na Composição** (Anotação: Inclui fórmula reduzida, número de elementos, lista de elementos.)
- `[x]` **Tipos de Atributos: Baseados na Estrutura** (Anotação: Inclui densidade, volume da célula, volume por átomo, número do grupo espacial, sistema cristalino, parâmetros de rede, número de sítios.)
- `[x]` **Ferramentas: Bibliotecas como pymatgen (integrada com o Materials Project) e Matminer são extremamente úteis para gerar uma ampla variedade de descritores de materiais.** (Anotação: `pymatgen` é usado extensivamente. `Matminer` não é usado.)

## 4. Seleção e Treinamento do Modelo de IA:

- `[x]` **Divisão de Dados: Separe seus dados em conjuntos de treinamento, validação e teste.** (Anotação: `train_test_split` do `scikit-learn` é usado.)
- `[~]` **Seleção de Algoritmo: Redes Neurais (Aprendizado Profundo)** (Anotação: Aprendizado de Máquina Clássico (Random Forest) do `scikit-learn` é implementado. Redes Neurais ou Redes Neurais em Grafos não são implementadas.)
- `[~]` **Treinamento:**
    - `[~]` Escolha uma função de perda (ex: Erro Quadrático Médio para regressão). (Anotação: Implícito para Random Forest.)
    - `[ ]` Escolha um otimizador (ex: Adam). (Anotação: Não aplicável para Random Forest como implementado.)
    - `[ ]` Ajuste os hiperparâmetros do modelo (ex: taxa de aprendizado, número de camadas/neurônios em redes neurais, profundidade da árvore em Random Forest). (Anotação: Hiperparâmetros padrão usados para Random Forest; nenhum ajuste implementado.)

## 5. Avaliação do Modelo:

- `[x]` **Métricas: Para problemas de regressão (como prever um gap de energia), use métricas como Erro Absoluto Médio (MAE), Raiz do Erro Quadrático Médio (RMSE), R² (coeficiente de determinação).** (Anotação: MAE e R² são implementados para modelos de regressão. RMSE não é, mas informação similar é transmitida.)
- `[ ]` **Validação Cruzada: Uma técnica importante para obter uma estimativa mais robusta do desempenho do modelo.** (Anotação: Não implementado.)
- `[~]` **Análise de Erro: Entenda onde seu modelo está falhando. Ele tem dificuldade com certos tipos de materiais ou faixas de valores?** (Anotação: Métricas básicas de avaliação são impressas. Nenhuma ferramenta ou relatório detalhado de análise de erro é gerado.)

## 6. Iteração e Refinamento:

- `[~]` **Com base na avaliação, você pode precisar voltar às etapas anteriores:** (Anotação: O projeto fornece scripts e uma GUI que permitem a argumentação de dados e o retreinamento, apoiando um processo iterativo. No entanto, nenhum loop automatizado de iteração ou refinamento foi implementado ou executado.)
    - `[~]` Coletar mais dados ou dados de melhor qualidade. (Anotação: Possível via modificação do script da API ou entrada manual.)
    - `[ ]` Projetar novos atributos. (Anotação: O conjunto atual de atributos está fixo por enquanto.)
    *   `[ ]` Tentar diferentes arquiteturas de modelo. (Anotação: Apenas Random Forest implementado.)
    *   `[ ]` Ajustar melhor os hiperparâmetros. (Anotação: Não implementado.)

## Ferramentas e Linguagens Comuns:

- `[x]` **Python:** A linguagem dominante para aprendizado de máquina.
- `[x]` **Bibliotecas Python Essenciais:**
    - `[x]` **scikit-learn:** Para aprendizado de máquina clássico.
    - `[ ]` **TensorFlow ou PyTorch:** Para aprendizado profundo.
    - `[x]` **pymatgen:** Para manipular estruturas cristalinas e dados de materiais.
    - `[ ]` **Matminer:** Para engenharia de atributos de materiais.
    - `[x]` **Pandas:** Para manipular dados tabulares.
    *   `[x]` **NumPy:** Para computação numérica (via pandas/sklearn).
    *   `[ ]` **Matplotlib / Seaborn:** Para visualização de dados. (Anotação: Nenhuma funcionalidade específica de visualização de dados implementada neste projeto.)

## Adiciona uma interface gráfica para a criação de novos materiais.

- `[x]` **(Referindo-se à GUI para inserir candidatos de materiais para predição & entrada manual de dados)** (Anotação: A GUI Tkinter inclui uma aba "Prever a partir de CIF" e uma aba "Entrada Manual de Dados", cumprindo isso.)

---
## Melhorias do Projeto

- `[x]` **Refatoração do Código da GUI:** O arquivo `material_predictor_gui.py` foi significativamente refatorado. Funcionalidades específicas de abas ('Prever a partir de CIF', 'Entrada Manual de Dados') foram movidas para suas próprias classes (`PredictionTab`, `ManualEntryTab`) para modularidade, legibilidade e manutenibilidade aprimoradas.

---
## Considerações Futuras & Próximos Passos Potenciais (Dicas)

Aqui estão algumas áreas potenciais para desenvolvimento e melhoria futuros:

*   **Treinamento Avançado de Modelos:**
    *   Implementar ajuste de hiperparâmetros (ex: usando `GridSearchCV` ou `RandomizedSearchCV` do `scikit-learn`) para os modelos Random Forest existentes para potencialmente melhorar seu desempenho.
    *   Incorporar validação cruzada durante o processo de treinamento de modelos (`train_model.py`) para métricas de avaliação mais robustas.
*   **Explorar Modelos Avançados:**
    *   Se conjuntos de dados maiores e mais diversos se tornarem disponíveis, explorar arquiteturas de modelo mais avançadas, como:
        *   Redes Neurais Feedforward (FNNs) para dados de atributos tabulares.
        *   Redes Neurais em Grafos (GNNs), como CGCNN, que podem aprender diretamente de estruturas cristalinas (exigiria mudanças significativas na engenharia de atributos e representação de dados).
*   **Avaliação & Análise Mais Profundas:**
    *   Desenvolver ferramentas ou saídas de análise de erro mais detalhadas. Por exemplo, identificar tipos de materiais ou faixas de atributos onde os modelos têm desempenho ruim.
    *   Implementar funcionalidade para plotar distribuições de atributos, distribuições de variáveis alvo ou correlações de predição (ex: gráficos de previsto vs. real). Isso pode envolver a integração de bibliotecas como Matplotlib/Seaborn, potencialmente como um script separado ou nova aba na GUI.
*   **Gerenciamento de Dados & Escalabilidade:**
    *   Para conjuntos de dados maiores, considerar a transição de CSVs para soluções de armazenamento mais robustas (ex: banco de dados SQLite, arquivos Parquet).
    *   Se usando conjuntos de dados muito grandes ou modelos complexos, explorar ferramentas para rastreamento de experimentos (ex: MLflow, Weights & Biases).
*   **Estrutura do Código & Projeto:**
    *   `[x]` **Introduzir um arquivo de configuração (ex: YAML ou JSON) para gerenciar configurações como caminhos de modelos, caminhos de arquivos ou parâmetros padrão, em vez de tê-los codificados nos scripts.** (Anotação: Implementado `config.yml` que centraliza todas as principais configurações: chaves de API, caminhos de arquivos para dados/modelos, parâmetros para critérios de busca de dados e treinamento de modelos (ex: test_size, n_estimators). Todos os scripts agora carregam deste arquivo via `utils.config_loader`.)
    *   `[x]` **Desenvolver um conjunto de testes unitários e de integração para garantir a confiabilidade do código e detectar regressões à medida que o projeto evolui.** (Anotação: Implementado um conjunto abrangente de testes no diretório `tests/` usando `pytest`. Testes unitários cobrem a lógica central em `fetch_mp_data` (simulando chamadas de API), `process_raw_data` (simulando pymatgen e E/S de arquivo), `train_model` (simulando sklearn e E/S de arquivo) e módulos utilitários. Um teste de integração verifica o pipeline de dados desde a busca até o treinamento de modelos usando o sistema de configuração. Testes de interação GUI estão pendentes devido a problemas de ambiente `tkinter`.)
    *   `[x]` **Modularizar ainda mais o código, por exemplo, movendo funções utilitárias ou definições de esquema de dados para módulos separados.** (Anotação: Criado um diretório `utils/` contendo `config_loader.py` para carregamento padronizado de configuração e `schema.py` para definições centralizadas de esquema de dados (`DATA_SCHEMA`, `MANUAL_ENTRY_CSV_HEADERS`). Os scripts foram atualizados para usar esses utilitários, reduzindo a redundância.)
*   **Melhorias na Interface do Usuário:**
    *   Permitir a seleção de diferentes modelos treinados se múltiplas versões ou tipos estiverem disponíveis.
    *   Fornecer feedback mais interativo ou visualizações dentro da GUI.

[fim de ROADMAP_STATUS_pt.md]
