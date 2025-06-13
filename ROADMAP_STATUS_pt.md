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

- `[x]` **(Prioridade 6/20) Pré-processamento e Divisão do Dataset:** (I)
    - `[x]` Processar todos os dados limpos, convertendo-os em objetos de grafo. (A)
    - `[x]` Salvar este dataset processado para acesso rápido. (B)
    - `[x]` Dividir o dataset em conjuntos de Treinamento (70%), Validação (20%) e Teste (10%). (B)

## Fase II: 🤖 Desenvolvimento do Modelo Preditivo "OracleNet" (Prioridades 7-10)

Com os dados prontos, construímos a ferramenta que irá guiar nosso gerador.

- `[X]` **(Prioridade 7/20) Design e Implementação da Arquitetura GNN Preditiva:** (A)
    - `[X]` Escolher e implementar uma arquitetura GNN (ex: SchNet, GAT, MEGNet) para o OracleNet. (A) (Modelo GCN implementado)
    - `[X]` O modelo deve aceitar um grafo como entrada e produzir um valor numérico (a Tc) como saída. (A) (Modelo implementado, produzindo valor numérico; adaptação para alvo Tc específico é parte da preparação de dados/escolha de treino)

- `[X]` **(Prioridade 8/20) Treinamento do Modelo Preditivo:** (A)
    - `[X]` Escrever o loop de treinamento para o OracleNet. (A)
    - `[X]` Treinar o modelo no conjunto de treinamento, usando o conjunto de validação para ajustar hiperparâmetros (taxa de aprendizado, tamanho das camadas, etc.). (I) (Treinamento inicial e salvamento do melhor modelo com base na perda de validação implementado. Ajuste de hiperparâmetros atualmente manual via config)

- `[X]` **(Prioridade 9/20) Avaliação Rigorosa do OracleNet:** (I)
    - `[X]` Medir o desempenho do modelo treinado no conjunto de teste (que o modelo nunca viu). (I)
    - `[X]` Métricas importantes: Erro Médio Absoluto (MAE), Raiz do Erro Quadrático Médio (RMSE). (B) (Implementado)
    - `[X]` Ponto de verificação crítico: O OracleNet deve ter um poder preditivo significativamente melhor que um baseline aleatório. Se não, volte para a Fase I ou melhore a arquitetura. (I) (Comparação com linha de base aleatória implementada)

- `[~]` **(Prioridade 10/20) Análise de Erros e Interpretabilidade:** (I)
    - `[~]` Analisar onde o OracleNet mais erra. Ele tem dificuldade com alguma família específica de materiais? (I) (Análise de erro básica implementada - mostra N maiores erros. Análise mais profunda de famílias de materiais pendente)
    - `[ ]` Usar técnicas de explicabilidade (XAI para GNNs) para entender quais subestruturas o modelo considera importantes para a supercondutividade. (A)

## Fase III (Revisada): ✨ Desenvolvimento do Modelo Gerativo "Creator" com VAE + LNN (Prioridades 11-16)

Objetivo: Criar um sistema que gera materiais quimicamente válidos e fisicamente estáveis (usando LNN), otimizados para alta Tc (usando VAE e OracleNet).

- `[ ]` **(Prioridade 11/20) Design da Arquitetura Híbrida (VAE + LNN):**
    - `[ ]` Projetar a arquitetura gerativa principal (baseada no VAE):
        - `[ ]` Encoder (GNN): Comprime um grafo de material em um vetor no espaço latente.
        - `[ ]` Decoder (GNN): Gera um novo grafo de material a partir de um vetor do espaço latente.
    - `[ ]` Projetar a rede de validação física:
        - `[ ]` Lagrangian Neural Network (LNN): Uma rede treinada para aprender uma aproximação da energia potencial de uma configuração atômica. Ela receberá um grafo gerado e avaliará sua estabilidade energética.

- `[ ]` **(Prioridade 12/20) Implementação da Função de Perda (Loss) Composta Avançada:**
    - `[ ]` Esta é a lógica que conecta geração, otimização de propriedade e realismo físico. A perda do VAE será uma soma ponderada de:
        - `[ ]` Perda de Reconstrução: Quão bem o VAE reconstrói os dados de entrada.
        - `[ ]` Perda de Divergência KL: Regularização padrão do espaço latente do VAE.
        - `[ ]` Perda Preditiva (OracleNet): Incentiva a geração de materiais com alta Tc prevista pelo OracleNet.
        - `[ ]` Perda de Estabilidade Física (LNN): Penaliza o gerador por criar estruturas que a LNN classifica como tendo alta energia (sendo instáveis ou fisicamente implausíveis). Este é o elo crucial com a LNN.

- `[ ]` **(Prioridade 13/20) Implementação do Loop de Treinamento Híbrido:**
    - `[ ]` Escrever o script que treina o sistema VAE. O treinamento da LNN pode ser feito separadamente com dados de simulações (ex: DFT) ou em conjunto.
    - `[ ]` No loop de treinamento principal do VAE:
        - `[ ]` Gerar um grafo "falso" com o Decoder.
        - `[ ]` Passar o grafo pelo OracleNet para obter a perda preditiva.
        - `[ ]` Passar o mesmo grafo pela LNN pré-treinada para obter a perda de estabilidade.
        - `[ ]` Calcular a perda composta e atualizar os pesos do VAE.

- `[ ]` **(Prioridade 14/20) Treinamento dos Modelos:**
    - `[ ]` 1. Treinar a LNN: Treinar a rede para prever a energia de configurações atômicas a partir de um banco de dados de materiais conhecidos e suas energias calculadas.
    - `[ ]` 2. Treinar o Sistema VAE: Executar o treinamento do VAE usando a função de perda composta, que agora inclui o feedback da LNN já treinada. Monitorar todas as componentes da perda.

- `[ ]` **(Prioridade 15/20) Geração do Lote de Candidatos Fisicamente Válidos:**
    - `[ ]` Usar o Decoder do VAE treinado para gerar milhares de novas estruturas.
    - `[ ]` Por construção, essas estruturas já foram otimizadas durante o treino para serem candidatas a terem alta Tc e estabilidade física.

- `[ ]` **(Prioridade 16/20) Filtragem e Ranqueamento Avançado dos Candidatos:**
    - `[ ]` Criar um pipeline final de avaliação, agora mais robusto:
        - `[ ]` Verificação final de validade química.
        - `[ ]` Re-executar a LNN para obter uma pontuação de estabilidade energética precisa para cada candidato finalista.
        - `[ ]` Executar o OracleNet para prever a Tc de cada um.
        - `[ ]` Ranquear os candidatos usando um critério combinado: maior Tc prevista E menor energia (maior estabilidade).

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
