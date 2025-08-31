# 🚀 Dashboard CSI - Classificação de Gênero

## 📋 Descrição

Este dashboard Streamlit apresenta uma análise completa de **classificação de gênero** usando dados **CSI (Channel State Information)** coletados de dispositivos Raspberry Pi. O projeto utiliza técnicas avançadas de Machine Learning para identificar o gênero de uma pessoa baseado em sinais de rádio capturados pelo ambiente.

## 🎯 Objetivos

- **Análise Exploratória** dos dados CSI
- **Seleção de Atributos** usando correlação de Pearson e Random Forest
- **Treinamento de Modelos** com 4 algoritmos diferentes
- **Comparação de Performance** com métricas detalhadas
- **Predições em Tempo Real** para novos dados
- **Relatórios e Exportação** de resultados

## 🔬 Metodologia

1. **Coleta de Dados:** 4000 amostras de 10 pessoas (400 por pessoa)
2. **Pré-processamento:** Análise de correlação e seleção de atributos
3. **Modelagem:** 4 algoritmos diferentes com validação cruzada
4. **Avaliação:** Métricas de performance e comparação de modelos

## 🛠️ Tecnologias Utilizadas

- **Streamlit** - Interface web interativa
- **Plotly** - Gráficos interativos e responsivos
- **Scikit-learn** - Algoritmos de Machine Learning
- **Pandas & NumPy** - Manipulação e análise de dados
- **Matplotlib & Seaborn** - Visualizações estáticas

## 📊 Funcionalidades do Dashboard

### 🏠 Visão Geral
- Descrição completa do projeto
- Métricas principais do dataset
- Gráficos de distribuição
- Estatísticas descritivas

### 📈 Análise Exploratória
- **Histogramas** interativos dos atributos
- **Box Plots** para análise de distribuição
- **Scatter Plots** para correlações
- **Matriz de Correlação** visual
- Análise por gênero com estatísticas

### 🎯 Seleção de Atributos
- **Correlação de Pearson** - Top 9 atributos
- **Random Forest** - Importância dos atributos
- **Atributos em Comum** entre os métodos
- **Pair Plot** dos melhores atributos
- Estatísticas dos atributos selecionados

### 🤖 Modelos de Machine Learning
- **Decision Tree** - Grid search em max_depth e criterion
- **kNN** - Grid search em n_neighbors
- **Naive Bayes** - Sem hiperparâmetros
- **SVM Linear** - Grid search em C
- **Validação Cruzada** com 3 folds
- Comparação de métricas (Accuracy, Precision, Recall, F1-Score)

### 🔮 Predições em Tempo Real
- Interface interativa para entrada de dados
- Sliders com estatísticas dos atributos
- Predições com todos os modelos
- Probabilidades e níveis de confiança
- Gráficos de resultados das predições

### 📋 Relatórios e Exportação
- Relatório completo de performance
- Download de dados em CSV
- Gráficos finais (radar chart)
- Estatísticas do dataset

## 🚀 Como Executar

### 1. Instalação das Dependências
```bash
# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar o Dashboard
```bash
# Navegar para a pasta notebooks
cd notebooks

# Executar o Streamlit
streamlit run app.py
```

### 3. Acessar o Dashboard
- Abra seu navegador
- Acesse: `http://localhost:8501`
- O dashboard será carregado automaticamente

## 📁 Estrutura de Arquivos

```
notebooks/
├── app.py                    # Dashboard principal
├── README_DASHBOARD.md      # Este arquivo
└── ...                      # Outros notebooks

DB_Gender/
├── DB_gender_400_fp_10p.csv # Dataset completo (4000 amostras)
├── DB_gender_400_fp_8p.csv  # Dataset treinamento (3200 amostras)
└── DB_gender_400_fp_2p.csv  # Dataset validação (800 amostras)
```

## 🔧 Configurações

### Porta do Servidor
- **Padrão:** 8501
- **Personalizar:** `streamlit run app.py --server.port 8502`

### Layout
- **Modo:** Wide (tela cheia)
- **Sidebar:** Expandida por padrão
- **Tema:** Claro com paleta de cores personalizada

## 📊 Dataset

### Características
- **Total de amostras:** 4000
- **Pessoas:** 10 (5 homens, 5 mulheres)
- **Amostras por pessoa:** 400
- **Atributos:** 234 características CSI + 1 classe (gênero)
- **Classes:** 0 (Mulher), 1 (Homem)

### Atributos Selecionados
Os melhores atributos identificados pela análise:
- `rpi1_sc121` - Subcarrier 121
- `rpi1_sc120` - Subcarrier 120
- `rpi1_sc119` - Subcarrier 119
- `rpi1_sc122` - Subcarrier 122
- `rpi1_sc118` - Subcarrier 118

## 🎨 Visualizações

### Gráficos Interativos
- **Plotly Express** para gráficos básicos
- **Plotly Graph Objects** para gráficos complexos
- **Subplots** para múltiplas visualizações
- **Responsividade** automática

### Tipos de Gráficos
- Gráficos de pizza
- Histogramas
- Box plots
- Scatter plots
- Matriz de correlação
- Pair plots
- Radar charts
- Gráficos de barras

## 🔍 Análise de Performance

### Métricas Utilizadas
- **Accuracy:** Taxa de acerto geral
- **Precision:** Precisão das predições positivas
- **Recall:** Sensibilidade das predições
- **F1-Score:** Média harmônica de precisão e recall
- **CV Score:** Média da validação cruzada

### Validação Cruzada
- **Estratégia:** Stratified K-Fold
- **Folds:** 3
- **Repetição:** 1 vez
- **Métrica:** Accuracy

## 💡 Dicas de Uso

### Para Análise Exploratória
1. Comece pela página "Visão Geral"
2. Use "Análise Exploratória" para entender os dados
3. Experimente diferentes tipos de gráficos
4. Selecione atributos específicos para análise

### Para Modelagem
1. Vá para "Seleção de Atributos" para identificar features importantes
2. Use "Modelos de ML" para treinar os algoritmos
3. Aguarde o treinamento (pode levar alguns minutos)
4. Analise os resultados e métricas

### Para Predições
1. Treine os modelos primeiro
2. Use "Predições" para testar novos dados
3. Ajuste os valores dos atributos com os sliders
4. Compare as predições de todos os modelos

### Para Relatórios
1. Acesse "Relatórios" após treinar os modelos
2. Baixe os resultados em CSV
3. Use os gráficos finais para apresentações

## 🚨 Solução de Problemas

### Erro ao Carregar Dados
- Verifique se os arquivos CSV estão na pasta `DB_Gender/`
- Confirme os nomes dos arquivos
- Verifique permissões de leitura

### Dashboard Não Carrega
- Confirme se o Streamlit está instalado
- Verifique se a porta 8501 está livre
- Use `streamlit --version` para verificar a instalação

### Erro de Memória
- Reduza o número de atributos selecionados
- Use apenas os atributos mais importantes
- Reinicie o dashboard se necessário

### Modelos Não Treinam
- Verifique se há dados suficientes
- Confirme se os atributos são numéricos
- Use menos folds na validação cruzada

## 🔮 Funcionalidades Futuras

- **Upload de novos dados** via interface
- **Salvamento de modelos** treinados
- **Comparação com outros datasets**
- **Análise de outros algoritmos**
- **Exportação de modelos** em diferentes formatos
- **API REST** para predições externas

## 📄 Créditos

Felipe Silveira de Almeida (ITA e Exército Brasileiro),  
Eduardo Fabrício Gomes Trindade (ITA e Exército Brasileiro),  
Gioliano de Oliveira Braga (ITA),  
Ágney Lopes Roth Ferraz (ITA),  
Giovani Hoff da Costa (ITA),  
Gustavo Cavalcanti Morais (ITA) e  
Lourenço Alves Pereira Júnio (ITA).  
