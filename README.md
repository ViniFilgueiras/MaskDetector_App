# 😷 Face Mask Detector

Detector de máscaras faciais usando Transfer Learning com MobileNet e TensorFlow, otimizado para evitar overfitting e garantir boa generalização.

## 📋 Descrição

Este projeto implementa um classificador binário de imagens capaz de detectar se uma pessoa está usando máscara facial ou não. O modelo utiliza Transfer Learning com MobileNet pré-treinada na ImageNet e foi desenvolvido com foco em:

- **Evitar overfitting** através de data augmentation e regularização
- **Boa generalização** validado em datasets externos
- **Otimização para embarcados** com conversão para TFLite INT8

## 🎯 Características

- ✅ Treinamento em 2 fases (feature extraction + fine-tuning)
- ✅ Data augmentation forte (flip, rotação, zoom, contraste)
- ✅ Regularização com Dropout
- ✅ Early stopping e learning rate adaptativo
- ✅ Validação cruzada com dataset externo
- ✅ Conversão para TFLite INT8 (otimizado para microcontroladores)
- ✅ Visualização detalhada do treinamento

## 🏗️ Arquitetura

```
Input (64x64x3)
    ↓
MobileNet Base (α=0.5, ImageNet weights)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (2, Softmax)
```

## 📊 Datasets

### Dataset de Treino
- **Fonte**: [Face Mask Dataset - Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Classes**: `with_mask`, `without_mask`
- **Split**: 80% treino, 20% validação

### Dataset de Teste
- **Fonte**: [Face Mask Dataset (External)](https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask)
- **Uso**: Validação externa para medir generalização

## 🚀 Como Usar

### 1. Instalação de Dependências

```bash
pip install tensorflow opencv-python matplotlib kagglehub scikit-learn
```

### 2. Executar o Notebook

```python
# No Google Colab ou Jupyter Notebook
# Execute todas as células em sequência
```

### 3. Estrutura do Código

O código está dividido em seções principais:

1. **Setup e Imports** - Configuração inicial
2. **Download dos Datasets** - Kagglehub automático
3. **Visualização** - Exploração dos dados
4. **Data Augmentation** - Preparação com transformações
5. **Modelo** - Construção da rede neural
6. **Fase 1: Feature Extraction** - Treino com base congelada
7. **Fase 2: Fine-tuning** - Treino com base parcialmente descongelada
8. **Avaliação Externa** - Teste com dataset diferente
9. **Exportação** - Salvamento em Keras e TFLite

## 📈 Resultados Esperados

### Performance

| Métrica | Treino | Validação | Teste Externo |
|---------|--------|-----------|---------------|
| Accuracy | 92-95% | 90-93% | 85-92% |
| Loss | ~0.15 | ~0.20 | ~0.25 |

### Diagnóstico de Overfitting

- **Gap Train-Val**: < 5% ✅
- **Generalização**: Acurácia externa > 85% ✅

## 🔧 Técnicas Anti-Overfitting Aplicadas

### 1. Data Augmentation
```python
- RandomFlip horizontal
- RandomRotation (±20%)
- RandomZoom (±20%)
- RandomContrast (±20%)
```

### 2. Regularização
- Dropout de 50% após pooling
- Dropout de 30% após camada densa

### 3. Treinamento Adaptativo
- Early Stopping (patience=5)
- ReduceLROnPlateau (fator=0.5)
- Learning rate inicial: 1e-3 → 1e-5

### 4. Transfer Learning em 2 Fases
- **Fase 1**: Base congelada, LR = 1e-3
- **Fase 2**: 20 últimas camadas descongeladas, LR = 1e-5

## 📦 Outputs do Modelo

### Arquivos Gerados

```
mask_detector_float.keras          # Modelo completo em Keras
mask_detector_savedmodel/          # SavedModel format
mask_detector_int8.tflite          # TFLite quantizado INT8 (~500 KB)
```

### Uso do Modelo TFLite

```python
import tensorflow as tf

# Carregar modelo
interpreter = tf.lite.Interpreter(model_path="mask_detector_int8.tflite")
interpreter.allocate_tensors()

# Obter detalhes de entrada/saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fazer predição
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

## 🎨 Visualizações

O código gera automaticamente:

1. **Amostra do Dataset** - 9 imagens aleatórias
2. **Imagens com Augmentation** - Visualização das transformações
3. **Gráficos de Treinamento**:
   - Accuracy (Train vs Val)
   - Loss (Train vs Val)
   - Linha de transição entre fases
4. **Predições no Teste** - 16 imagens com:
   - Label real vs predição
   - Confiança do modelo
   - Cores (verde = acerto, vermelho = erro)

## ⚙️ Configurações

### Hiperparâmetros Principais

```python
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 20
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
```

### GPU (Opcional)

O código detecta automaticamente GPUs disponíveis:
```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
```

## 🐛 Troubleshooting

### Problema: Acurácia externa muito baixa (< 70%)

**Solução**: 
- Aumentar data augmentation
- Aumentar dropout (0.6-0.7)
- Treinar por mais épocas

### Problema: Overfitting (gap > 10%)

**Solução**:
- Já implementado no código!
- Verificar se callbacks estão ativos
- Reduzir learning rate

### Problema: Modelo não aprende (acc < 60%)

**Solução**:
- Verificar normalização das imagens
- Aumentar learning rate inicial
- Descongelar mais camadas na fase 2

## 📚 Referências

- [MobileNets: Efficient CNNs for Mobile Vision](https://arxiv.org/abs/1704.04861)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TFLite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

## 👨‍💻 Autor

Projeto desenvolvido para a disciplina de Processamento Digital de Imagens (PDI).

## 📄 Licença

Este projeto é de código aberto para fins educacionais.

---

**💡 Dica**: Para melhores resultados, execute no Google Colab com GPU habilitada!
