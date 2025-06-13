# Roadmap para Descoberta de Supercondutores com IA: Um Plano de 20 Passos

## Pilha Tecnológica Principal (Core Stack)

Este projeto utiliza as seguintes tecnologias principais:

*   **Linguagem**: Python 3.10+
*   **Aprendizado de Máquina (Machine Learning)**:
    *   PyTorch (`torch`): Framework principal de deep learning. (Instalado)
    *   PyTorch Geometric (PyG) (`torch_geometric`): Biblioteca para deep learning em grafos. (Instalado)
*   **Química/Ciência dos Materiais**:
    *   RDKit (`rdkit-pypi`): Toolkit para quimioinformática. (Instalado)
    *   Pymatgen (`pymatgen`): Biblioteca para análise de materiais. (Instalado)
*   **Gerenciamento de Experimentos (Planejado)**:
    *   Weights & Biases (W&B) ou MLflow. (A ser integrado)
*   **Controle de Versão de Dados (Planejado)**:
    *   DVC (Data Version Control). (A ser integrado)

**Legenda:**
- `[x]` Implementado
- `[~]` Parcialmente Implementado ou Versão Básica Existente
- `[ ]` Não Implementado

---

Este plano transforma as fases estratégicas em um fluxo de trabalho acionável, com prioridades claras do setup inicial até a validação experimental.

## Fase I: 🏗️ Construção da Fundação de Dados (Prioridades 1-6)

O sucesso do projeto depende inteiramente da qualidade e da organização dos seus dados.

- `[~]` **(Prioridade 1/20) Setup do Ambiente de Desenvolvimento:**
    - `[x]` Configurar um repositório de código (Git). (B)
    - `[~]` Instalar bibliotecas essenciais de IA (PyTorch, TensorFlow). (B)
    - `[~]` Instalar bibliotecas de IA para grafos (PyTorch Geometric ou DGL). (B)
    - `[x]` Instalar bibliotecas de química/materiais (Pymatgen, RDKit). (B)

- `[~]` **(Prioridade 2/20) Identificação e Acesso às Fontes de Dados:**
    - `[~]` Acessar Materials Project (API, opcional), SuperCon (arquivos locais), OQMD (API identificada); ICSD pendente. (I)
    - `[x]` Definir critérios de busca (SuperCon para Tc, OQMD/MP para propriedades). (B)

- `[~]` **(Prioridade 3/20) Desenvolvimento de Scripts para Extração de Dados:**
    - `[x]` Scripts para extração de dados do Materials Project existem. (I)
    - `[x]` Desenvolver script para processar dataset local SuperCon (`raw.tsv`) para composições e Tc. (I)
    - `[x]` Desenvolver script para buscar dados complementares da API OQMD para composições do SuperCon. (I)
    - `[x]` Processar dados OQMD buscados (`oqmd_data_raw.json`) para selecionar/filtrar entradas e extrair features. (I)
    - `[x]` Armazenar os dados brutos em um formato organizado (ex: banco de dados local ou data lake). (B)

- `[x]` **(Prioridade 4/20) Limpeza e Normalização dos Dados:** (I)
    - `[x]` Validar os dados extraídos, tratando valores faltantes e inconsistências. (I)
    - `[x]` Unificar unidades e formatos. Por exemplo, garantir que todas as estruturas cristalinas estejam em um formato padrão como arquivos CIF. (I)

- `[x]` **(Prioridade 5/20) Definição e Implementação da Representação em Grafo:** (A)
    - `[x]` Definir formalmente como uma estrutura cristalina será convertida em um grafo. (A)
        - `[x]` Nós: Átomos (com features como número atômico, eletronegatividade). (I)
        - `[x]` Arestas: Ligações ou vizinhança (com features como distância). (I)
    - `[x]` Implementar a função de conversão Estrutura -> Grafo. (A)

- `[~]` **(Prioridade 6/20) Pré-processamento e Divisão do Dataset:** (I)
    - `[ ]` Processar todos os dados limpos, convertendo-os em objetos de grafo. (A)
    - `[ ]` Salvar este dataset processado para acesso rápido. (B)
    - `[~]` Dividir o dataset em conjuntos de Treinamento (70%), Validação (15%) e Teste (15%). (B)

## Fase II: 🤖 Desenvolvimento do Modelo Preditivo "OracleNet" (Prioridades 7-10)

Com os dados prontos, construímos a ferramenta que irá guiar nosso gerador.

- `[ ]` **(Prioridade 7/20) Design e Implementação da Arquitetura GNN Preditiva:** (A)
    - `[ ]` Escolher e implementar uma arquitetura GNN (ex: SchNet, GAT, MEGNet) para o OracleNet. (A)
    - `[ ]` O modelo deve aceitar um grafo como entrada e produzir um valor numérico (a Tc) como saída. (A)

- `[~]` **(Prioridade 8/20) Treinamento do Modelo Preditivo:** (A)
    - `[ ]` Escrever o loop de treinamento para o OracleNet. (A)
    - `[~]` Treinar o modelo no conjunto de treinamento, usando o conjunto de validação para ajustar hiperparâmetros (taxa de aprendizado, tamanho das camadas, etc.). (I)

- `[~]` **(Prioridade 9/20) Avaliação Rigorosa do OracleNet:** (I)
    - `[~]` Medir o desempenho do modelo treinado no conjunto de teste (que o modelo nunca viu). (I)
    - `[~]` Métricas importantes: Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE). (B)
    - `[ ]` Ponto de verificação crítico: O OracleNet deve ter um poder preditivo significativamente melhor que um baseline aleatório. Se não, volte para a Fase I ou melhore a arquitetura. (I)

- `[ ]` **(Prioridade 10/20) Análise de Erros e Interpretabilidade:** (I)
    - `[ ]` Analisar onde o OracleNet mais erra. Ele tem dificuldade com alguma família específica de materiais? (I)
    - `[ ]` Usar técnicas de explicabilidade (XAI para GNNs) para entender quais subestruturas o modelo considera importantes para a supercondutividade. (A)

## Fase III: ✨ Desenvolvimento do Modelo Gerativo "Creator" (Prioridades 11-16)

Agora, a parte mais inovadora: criar novos materiais.

- `[ ]` **(Prioridade 11/20) Design da Arquitetura GAN para Grafos:** (A)
    - `[ ]` Projetar as duas redes principais: (A)
        - `[ ]` Gerador: Uma GNN que recebe ruído e gera um novo grafo de material. (A)
        - `[ ]` Discriminador: Uma GNN que recebe um grafo e o classifica como real ou falso. (A)

- `[ ]` **(Prioridade 12/20) Implementação da Função de Perda (Loss) Composta:** (A)
    - `[ ]` Esta é a lógica central. A função de perda do Gerador será uma soma ponderada de: (A)
        - `[ ]` Perda Adversária: Quão bem ele engana o Discriminador. (A)
        - `[ ]` Perda Preditiva: Quão alta é a Tc prevista pelo OracleNet para o material gerado (o objetivo é maximizar isso). (A)
        - `[ ]` (Opcional) Termos de regularização para garantir validade química. (I)

- `[ ]` **(Prioridade 13/20) Implementação do Loop de Treinamento da GAN:** (A)
    - `[ ]` Escrever o script que alterna entre o treinamento do Discriminador (com dados reais e falsos) e o do Gerador (usando a loss composta). Este ciclo é mais complexo que o da Fase II. (A)

- `[ ]` **(Prioridade 14/20) Treinamento do Sistema GAN Completo:** (A)
    - `[ ]` Executar o treinamento da GAN. Este passo é computacionalmente intensivo e pode exigir GPUs potentes. (A)
    - `[ ]` Monitorar as perdas do Gerador e do Discriminador para garantir que o treinamento está estável. (I)

- `[ ]` **(Prioridade 15/20) Geração do Lote Inicial de Candidatos:** (I)
    - `[ ]` Usar o Gerador treinado para criar um grande número (milhares) de novas estruturas moleculares que não existem na base de dados. (I)

- `[ ]` **(Prioridade 16/20) Filtragem e Ranqueamento dos Candidatos Gerados:** (I)
    - `[ ]` Criar um pipeline para avaliar os candidatos gerados: (I)
        - `[ ]` Verificar validade química básica. (B)
        - `[ ]` Executar o OracleNet para prever a Tc de cada um. (I)
        - `[ ]` Ranquer os candidatos da Tc mais alta para a mais baixa. (B)

## Fase IV: 🧪 Validação e Fechamento do Ciclo (Prioridades 17-20)

Onde a IA encontra o mundo real.

- `[ ]` **(Prioridade 17/20) Triagem com Simulações Computacionais Clássicas:** (A)
    - `[ ]` Pegar o top ~100 da lista ranqueada. (B)
    - `[ ]` Realizar simulações mais precisas, porém mais lentas (como DFT), para verificar a estabilidade e as propriedades eletrônicas desses candidatos. (A)

- `[ ]` **(Prioridade 18/20) Seleção dos Candidatos Finais para Síntese:** (I)
    - `[ ]` Com base nos resultados da IA e da triagem computacional, selecionar um pequeno número (1 a 5) de candidatos "campeões" para validação experimental. (I)

- `[ ]` **(Prioridade 19/20) Colaboração para Síntese e Teste em Laboratório:** (B)
    - `[ ]` Este passo requer colaboração com um laboratório de física ou química de materiais. (B)
    - `[ ]` Os parceiros tentarão sintetizar os materiais propostos e medir suas propriedades reais, incluindo a Tc. (B)

- `[ ]` **(Prioridade 20/20) Fechamento do Ciclo de "Active Learning":** (I)
    - `[ ]` O passo mais importante para o sucesso a longo prazo. (B)
    - `[ ]` Pegar os resultados experimentais (seja sucesso ou falha) da Fase 19. (B)
    - `[ ]` Adicionar esses novos pontos de dados à sua base de dados original. (B)
    - `[ ]` Re-treinar o OracleNet e, opcionalmente, o sistema GAN com esses novos dados. (A)
    - `[ ]` Repetir o ciclo a partir da Fase III/IV. (B)
