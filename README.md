# Otimizações e Melhorias - Classificação de RX de Tórax

Repositório com as correções e melhorias aplicadas sobre o pipeline base de CNNs para classificação de raios-X de tórax (Normal, COVID e Pneumonia).

## Principais Ajustes
- **Ambiente Local e AlexNet:** Migração completa do Google Colab para execução local, atualização das importações do Keras e ajustes estruturais no pipeline da **AlexNet** (como caminhos de diretórios, carregamento do dataset e redimensionamento das amostras).
- **Correção na VGG16:** Ajuste da camada de saída (`Dense(3)`, `softmax`, `sparse_categorical_crossentropy`) e substituição do limiar binário por `np.argmax` para prever corretamente as três classes.
- **Correção no Xception:** Resolução de um erro crítico de desalinhamento entre classes reais e previsões, configurando `shuffle=False` e predição direta via gerador.
- **Otimização de Treino:** Aumento do número de lotes por época para garantir o processamento completo do conjunto de dados, além de melhorias no pré-processamento e visualização de amostras via `matplotlib` e `OpenCV`.
