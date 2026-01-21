# Face Mask Detector para ESP32-CAM

Detector de máscaras faciais utilizando uma versão otimizada da MobileNetV2 treinada do zero, projetado para execução em hardware de baixo consumo e memória restrita como a ESP32-CAM.

## Descrição

Este projeto implementa um classificador binário de imagens capaz de detectar se uma pessoa está utilizando máscara facial. Ao contrário de abordagens convencionais, este modelo foi desenvolvido sem o uso de Transfer Learning. Isso permitiu a redução do parâmetro Alpha para valores inferiores aos suportados por pesos pré-treinados, resultando em uma rede extremamente leve e eficiente.

Principais objetivos:
- Otimização extrema para sistemas embarcados (foco em ESP32-CAM)
- Redução do tamanho do modelo para menos de 400KB via quantização INT8
- Alta acurácia de generalização validada em datasets externos

## Características

- Treinamento do zero (No Transfer Learning) para flexibilidade total de arquitetura
- MobileNetV2 com Alpha reduzido (0.25) para economia de memória
- Data Augmentation robusto (flip, rotação, zoom, contraste)
- Regularização com Dropout para evitar overfitting
- Callbacks de Early Stopping e redução dinâmica de Learning Rate
- Conversão para TFLite com quantização Full Integer (INT8)
- Homologação com dataset externo para garantir robustez no mundo real

## Arquitetura

Input (64x64x3)
    |
MobileNetV2 Base (Alpha=0.25, weights=None)
    |
Global Average Pooling
    |
Dropout (0.2)
    |
Dense (2, Softmax)

## Datasets

### Dataset de Treino e Validação
- Fonte: Face Mask Dataset (Omkar Gurav) via Kaggle
- Composição: 7.553 imagens no total
- Divisão: 80% para treino (6.043 imagens), 20% para validação (1.510 imagens)

### Dataset de Teste Externo
- Fonte: Face Mask Dataset with and without mask (Belsonraja) via Kaggle
- Objetivo: Validar a capacidade de generalização do modelo em um ambiente de dados totalmente independente

## Como Usar

### 1. Instalação de Dependências

Recomenda-se o uso de um ambiente Python 3.10+ com as seguintes bibliotecas:
- tensorflow
- matplotlib
- kagglehub
- numpy
- scikit-learn

### 2. Execução do Projeto

O desenvolvimento foi realizado no notebook PDI_Project_Final.ipynb. Para reproduzir:
- Certifique-se de ter as credenciais do Kaggle configuradas para o download automático via kagglehub.
- Execute as células em sequência para realizar o download dos dados, treinamento e exportação.

### 3. Fases de Desenvolvimento

1. Configuração Inicial: Definição de hiperparâmetros e constantes.
2. Download e Extração: Aquisição dos dados via KaggleHub.
3. Pré-processamento: Redimensionamento para 64x64 e normalização básica.
4. Treinamento: Execução de 30 épocas com monitoramento de val_loss.
5. Avaliação: Teste de acurácia nos dados de validação e nos dados externos.
6. Exportação TFLite: Quantização baseada em amostras reais do dataset (Representative Dataset).

## Performance e Resultados

Abaixo os resultados típicos obtidos com a configuração de Alpha 0.25 e entrada de 64x64:

| Métrica | Treino | Validação | Teste Externo |
|---------|--------|-----------|---------------|
| Accuracy| ~94.49%| ~93.64%   |   ~96.56%     |

O gap reduzido entre treino e validação indica que as técnicas de Data Augmentation e a arquitetura simplificada foram eficazes contra o overfitting.

## Otimização para Microcontroladores

Para viabilizar a execução na ESP32-CAM, o modelo passou por um processo de quantização pós-treinamento (Post-Training Quantization):

1. Formato Original (.keras): Aproximadamente 980 KB.
2. Formato TFLite INT8: Aproximadamente 411.41 KB.
3. Formato .h: Aproximadamente 421.29 KB.

Arquivos Gerados:
- mask_detector_light.keras: Pesos em precisão total.
- mask_detector_light_int8.tflite: Modelo pronto para implementação via TensorFlow Lite Micro.
- model_data.h: Modelo pronto para implementação na ESP32-CAM.

## Técnicas Aplicadas

### Data Augmentation
- RandomFlip: Inversão horizontal.
- RandomRotation: Rotação de até 10%.
- RandomZoom: Zoom de até 10%.
- RandomContrast: Ajustes de contraste para robustez a iluminação.

### Regularização e Treinamento
- Dropout: Redução de co-dependência de neurônios.
- Adam Optimizer: Com learning rate inicial de 1e-3.
- ReduceLROnPlateau: Redução do fator de aprendizado caso a perda estagne por 5 épocas.
- Early Stopping: Interrupção caso não haja melhora por 20 épocas.

## Referências

- MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al.)
- Documentação do TensorFlow Lite para Microcontroladores
- Repositórios de dados do Kaggle (Omkar Gurav e Belsonraja)

## Autor

Projeto desenvolvido como parte da disciplina de Processamento Digital de Imagens (PDI), focado em soluções práticas de Visão Computacional Embarcada.
