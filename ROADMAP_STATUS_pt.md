# Roteiro de Desenvolvimento do Modelo de IA: Status Atual

Este documento acompanha o progresso do projeto em relação ao roteiro inicial para construir um modelo de IA para prever propriedades de materiais.

- `[x]` Implementado
- `[~]` Parcialmente Implementado ou Versão Básica Existente
- `[ ]` Não Implementado

---

## 1. Definir o Problema e o Escopo:

- `[x]` **Qual propriedade eletrônica específica você quer prever?** (Implementado: Gap de energia, características da densidade de estados, energia de formação. Estabilidade não diretamente, embora a energia de formação esteja relacionada.)
- `[~]` **Para qual classe de materiais?** (Anotação: O conjunto de dados inicial focou em compostos baseados em Fe do Materials Project. A ferramenta GUI em si é genérica para qualquer CIF.)
- `[ ]` **Qual é o nível de precisão desejado e o custo computacional aceitável?** (Anotação: A precisão é avaliada, mas nenhum alvo específico como "80%" foi definido ou otimizado.)

## 2. Aquisição e Preparação de Dados:

- `[x]` **Fonte de Dados: Bancos de Dados Públicos** (Anotação: Implementada a busca na API do Materials Project via `fetch_mp_data.py`. AFLOW foi discutido, mas não implementado como fonte direta.)
- `[~]` **Fonte de Dados: Seus Próprios Cálculos** (Anotação: Cálculos DFT diretos pelo agente não são viáveis. No entanto, a aba "Entrada Manual de Dados" na GUI permite que os usuários insiram dados de seus próprios cálculos.)
- `[~]` **Limpeza e Validação:** (Anotação: O tratamento básico de NaN é feito no script de treinamento. Nenhum método abrangente de limpeza ou validação de dados foi implementado ainda.)
- `[~]` **Volume de Dados:** (Anotação: O framework busca ~50 materiais para demonstração. Os scripts podem ser adaptados para mais, e a entrada manual é possível. O conjunto de dados atual não é de "grande volume" no contexto de aprendizado profundo.)

## 3. Engenharia de Características (Descritores de Materiais):

- `[x]` **Esta é uma das etapas mais cruciais. Você precisa converter as informações do material (composição química, estrutura cristalina) em um formato numérico que o modelo de IA possa entender.** (Anotação: Implementado via `process_raw_data.py` usando `pymatgen`.)
- `[x]` **Tipos de Características: Baseadas na Composição** (Anotação: Inclui fórmula reduzida, número de elementos, lista de elementos.)
- `[x]` **Tipos de Características: Baseadas na Estrutura** (Anotação: Inclui densidade, volume da célula, volume por átomo, número do grupo espacial, sistema cristalino, parâmetros de rede, número de sítios.)
- `[x]` **Ferramentas: Bibliotecas como pymatgen (integrada com o Materials Project) e Matminer são extremamente úteis para gerar uma ampla variedade de descritores de materiais.** (Anotação: `pymatgen` é usado extensivamente. `Matminer` não é usado.)

## 4. Seleção e Treinamento do Modelo de IA:

- `[x]` **Divisão dos Dados: Separe seus dados em conjuntos de treinamento, validação e teste.** (Anotação: `train_test_split` do `scikit-learn` é usado.)
- `[~]` **Seleção do Algoritmo: Redes Neurais (Deep Learning)** (Anotação: Machine Learning Clássico (Random Forest) do `scikit-learn` é implementado. Redes Neurais ou Redes Neurais Grafos não são implementadas.)
- `[~]` **Treinamento:**
    - `[~]` Escolha uma função de perda (ex.: Erro Quadrático Médio para regressão). (Anotação: Implícito para Random Forest.)
    - `[ ]` Escolha um otimizador (ex.: Adam). (Anotação: Não aplicável para Random Forest como implementado.)
    - `[ ]` Ajuste os hiperparâmetros do modelo (ex.: taxa de aprendizado, número de camadas/neurônios em redes neurais, profundidade da árvore em Random Forest). (Anotação: Hiperparâmetros padrão usados para Random Forest; nenhum ajuste implementado.)

## 5. Avaliação do Modelo:

- `[x]` **Métricas: Para problemas de regressão (como prever um gap de energia), use métricas como Erro Absoluto Médio (MAE), Raiz do Erro Quadrático Médio (RMSE), R² (coeficiente de determinação).** (Anotação: MAE e R² são implementados para modelos de regressão. RMSE não é, mas informações similares são transmitidas.)
- `[ ]` **Validação Cruzada: Uma técnica importante para obter uma estimativa mais robusta do desempenho do modelo.** (Anotação: Não implementada.)
- `[~]` **Análise de Erros: Entenda onde seu modelo está falhando. Ele tem dificuldade com certos tipos de materiais ou faixas de valores?** (Anotação: Métricas básicas de avaliação são impressas. Nenhuma ferramenta detalhada de análise de erros ou relatórios são gerados.)

## 6. Iteração e Refinamento:

- `[~]` **Com base na avaliação, você pode precisar voltar às etapas anteriores:** (Anotação: O projeto fornece scripts e uma GUI que permitem o aumento de dados e o retreinamento, apoiando um processo iterativo. No entanto, nenhum loop automatizado de iteração ou refinamento foi implementado ou executado.)
    - `[~]` Coletar mais dados ou dados de melhor qualidade. (Anotação: Possível via modificação do script da API ou entrada manual.)
    - `[ ]` Projetar novas características. (Anotação: O conjunto atual de características está fixo por enquanto.)
    *   `[ ]` Tentar diferentes arquiteturas de modelo. (Anotação: Apenas Random Forest implementado.)
    *   `[ ]` Ajustar melhor os hiperparâmetros. (Anotação: Não implementado.)

## Ferramentas e Linguagens Comuns:

- `[x]` **Python:** A linguagem dominante para machine learning.
- `[x]` **Bibliotecas Python Essenciais:**
    - `[x]` **scikit-learn:** Para machine learning clássico.
    - `[ ]` **TensorFlow ou PyTorch:** Para deep learning.
    - `[x]` **pymatgen:** Para manipular estruturas cristalinas e dados de materiais.
    - `[ ]` **Matminer:** Para featurização de materiais.
    - `[x]` **Pandas:** Para manipular dados tabulares.
    *   `[x]` **NumPy:** Para computação numérica (via pandas/sklearn).
    *   `[ ]` **Matplotlib / Seaborn:** Para visualização de dados. (Anotação: Nenhuma característica específica de visualização de dados implementada neste projeto.)

## Adiciona uma interface gráfica para a criação de novos materiais.

- `[x]` **(Referindo-se à GUI para inserir candidatos de materiais para predição e entrada manual de dados)** (Anotação: A GUI Tkinter inclui uma aba "Prever a partir de CIF" e uma aba "Entrada Manual de Dados", cumprindo este requisito.)
