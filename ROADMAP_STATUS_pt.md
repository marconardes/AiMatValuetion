# Roadmap para Descoberta de Supercondutores com IA: Um Plano de 20 Passos

Este plano transforma as fases estratégicas em um fluxo de trabalho acionável, com prioridades claras do setup inicial até a validação experimental.

## Fase I: 🏗️ Construção da Fundação de Dados (Prioridades 1-6)

O sucesso do projeto depende inteiramente da qualidade e da organização dos seus dados.

- `[~]` **(Prioridade 1/20) Setup do Ambiente de Desenvolvimento:**
    - `[x]` Configurar um repositório de código (Git).
    - `[ ]` Instalar bibliotecas essenciais de IA (PyTorch, TensorFlow).
    - `[ ]` Instalar bibliotecas de IA para grafos (PyTorch Geometric ou DGL).
    - `[~]` Instalar bibliotecas de química/materiais (Pymatgen, RDKit).

- `[~]` **(Prioridade 2/20) Identificação e Acesso às Fontes de Dados:**
    - `[~]` Obter chaves de API e permissões para acessar bancos de dados como Materials Project, ICSD e SuperCon.
    - `[~]` Definir os critérios de busca para materiais relevantes.

- `[x]` **(Prioridade 3/20) Desenvolvimento de Scripts para Extração de Dados:**
    - `[x]` Escrever e executar scripts para baixar sistematicamente os dados estruturais e de propriedades dos materiais selecionados.
    - `[x]` Armazenar os dados brutos em um formato organizado (ex: banco de dados local ou data lake).

- `[~]` **(Prioridade 4/20) Limpeza e Normalização dos Dados:**
    - `[~]` Validar os dados extraídos, tratando valores faltantes e inconsistências.
    - `[~]` Unificar unidades e formatos. Por exemplo, garantir que todas as estruturas cristalinas estejam em um formato padrão como arquivos CIF.

- `[ ]` **(Prioridade 5/20) Definição e Implementação da Representação em Grafo:**
    - `[ ]` Definir formalmente como uma estrutura cristalina será convertida em um grafo.
        - `[ ]` Nós: Átomos (com features como número atômico, eletronegatividade).
        - `[ ]` Arestas: Ligações ou vizinhança (com features como distância).
    - `[ ]` Implementar a função de conversão Estrutura -> Grafo.

- `[~]` **(Prioridade 6/20) Pré-processamento e Divisão do Dataset:**
    - `[ ]` Processar todos os dados limpos, convertendo-os em objetos de grafo.
    - `[ ]` Salvar este dataset processado para acesso rápido.
    - `[~]` Dividir o dataset em conjuntos de Treinamento (70%), Validação (15%) e Teste (15%).

## Fase II: 🤖 Desenvolvimento do Modelo Preditivo "OracleNet" (Prioridades 7-10)

Com os dados prontos, construímos a ferramenta que irá guiar nosso gerador.

- `[ ]` **(Prioridade 7/20) Design e Implementação da Arquitetura GNN Preditiva:**
    - `[ ]` Escolher e implementar uma arquitetura GNN (ex: SchNet, GAT, MEGNet) para o OracleNet.
    - `[ ]` O modelo deve aceitar um grafo como entrada e produzir um valor numérico (a Tc) como saída.

- `[~]` **(Prioridade 8/20) Treinamento do Modelo Preditivo:**
    - `[ ]` Escrever o loop de treinamento para o OracleNet.
    - `[~]` Treinar o modelo no conjunto de treinamento, usando o conjunto de validação para ajustar hiperparâmetros (taxa de aprendizado, tamanho das camadas, etc.).

- `[~]` **(Prioridade 9/20) Avaliação Rigorosa do OracleNet:**
    - `[~]` Medir o desempenho do modelo treinado no conjunto de teste (que o modelo nunca viu).
    - `[~]` Métricas importantes: Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE).
    - `[ ]` Ponto de verificação crítico: O OracleNet deve ter um poder preditivo significativamente melhor que um baseline aleatório. Se não, volte para a Fase I ou melhore a arquitetura.

- `[ ]` **(Prioridade 10/20) Análise de Erros e Interpretabilidade:**
    - `[ ]` Analisar onde o OracleNet mais erra. Ele tem dificuldade com alguma família específica de materiais?
    - `[ ]` Usar técnicas de explicabilidade (XAI para GNNs) para entender quais subestruturas o modelo considera importantes para a supercondutividade.

## Fase III: ✨ Desenvolvimento do Modelo Gerativo "Creator" (Prioridades 11-16)

Agora, a parte mais inovadora: criar novos materiais.

- `[ ]` **(Prioridade 11/20) Design da Arquitetura GAN para Grafos:**
    - `[ ]` Projetar as duas redes principais:
        - `[ ]` Gerador: Uma GNN que recebe ruído e gera um novo grafo de material.
        - `[ ]` Discriminador: Uma GNN que recebe um grafo e o classifica como real ou falso.

- `[ ]` **(Prioridade 12/20) Implementação da Função de Perda (Loss) Composta:**
    - `[ ]` Esta é a lógica central. A função de perda do Gerador será uma soma ponderada de:
        - `[ ]` Perda Adversária: Quão bem ele engana o Discriminador.
        - `[ ]` Perda Preditiva: Quão alta é a Tc prevista pelo OracleNet para o material gerado (o objetivo é maximizar isso).
        - `[ ]` (Opcional) Termos de regularização para garantir validade química.

- `[ ]` **(Prioridade 13/20) Implementação do Loop de Treinamento da GAN:**
    - `[ ]` Escrever o script que alterna entre o treinamento do Discriminador (com dados reais e falsos) e o do Gerador (usando a loss composta). Este ciclo é mais complexo que o da Fase II.

- `[ ]` **(Prioridade 14/20) Treinamento do Sistema GAN Completo:**
    - `[ ]` Executar o treinamento da GAN. Este passo é computacionalmente intensivo e pode exigir GPUs potentes.
    - `[ ]` Monitorar as perdas do Gerador e do Discriminador para garantir que o treinamento está estável.

- `[ ]` **(Prioridade 15/20) Geração do Lote Inicial de Candidatos:**
    - `[ ]` Usar o Gerador treinado para criar um grande número (milhares) de novas estruturas moleculares que não existem na base de dados.

- `[ ]` **(Prioridade 16/20) Filtragem e Ranqueamento dos Candidatos Gerados:**
    - `[ ]` Criar um pipeline para avaliar os candidatos gerados:
        - `[ ]` Verificar validade química básica.
        - `[ ]` Executar o OracleNet para prever a Tc de cada um.
        - `[ ]` Ranquer os candidatos da Tc mais alta para a mais baixa.

## Fase IV: 🧪 Validação e Fechamento do Ciclo (Prioridades 17-20)

Onde a IA encontra o mundo real.

- `[ ]` **(Prioridade 17/20) Triagem com Simulações Computacionais Clássicas:**
    - `[ ]` Pegar o top ~100 da lista ranqueada.
    - `[ ]` Realizar simulações mais precisas, porém mais lentas (como DFT), para verificar a estabilidade e as propriedades eletrônicas desses candidatos.

- `[ ]` **(Prioridade 18/20) Seleção dos Candidatos Finais para Síntese:**
    - `[ ]` Com base nos resultados da IA e da triagem computacional, selecionar um pequeno número (1 a 5) de candidatos "campeões" para validação experimental.

- `[ ]` **(Prioridade 19/20) Colaboração para Síntese e Teste em Laboratório:**
    - `[ ]` Este passo requer colaboração com um laboratório de física ou química de materiais.
    - `[ ]` Os parceiros tentarão sintetizar os materiais propostos e medir suas propriedades reais, incluindo a Tc.

- `[ ]` **(Prioridade 20/20) Fechamento do Ciclo de "Active Learning":**
    - `[ ]` O passo mais importante para o sucesso a longo prazo.
    - `[ ]` Pegar os resultados experimentais (seja sucesso ou falha) da Fase 19.
    - `[ ]` Adicionar esses novos pontos de dados à sua base de dados original.
    - `[ ]` Re-treinar o OracleNet e, opcionalmente, o sistema GAN com esses novos dados. O sistema ficará mais inteligente a cada iteração.
    - `[ ]` Repetir o ciclo a partir da Fase III/IV.
