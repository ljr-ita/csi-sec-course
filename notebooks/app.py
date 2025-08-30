import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configuração da página
st.set_page_config(
    page_title="Dashboard CSI - Classificação de Gênero",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração do matplotlib para Streamlit
plt.style.use('default')
sns.set_palette("husl")

# Mapeamento explícito das métricas para evitar KeyError
METRIC_MAPPING = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'F1-Score': 'f1'
}

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega os dados de gênero"""
    try:
        # Dados completos para análise
        df_full = pd.read_csv("/DB_Gender/DB_gender_400_fp_10p.csv")
        
        # Dados para treinamento (8 pessoas)
        df_train = pd.read_csv("/DB_Gender/DB_gender_400_fp_8p.csv")
        
        return df_full, df_train
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        # Tentar caminhos alternativos
        try:
            df_full = pd.read_csv("DB_Gender/DB_gender_400_fp_10p.csv")
            df_train = pd.read_csv("DB_Gender/DB_gender_400_fp_8p.csv")
            return df_full, df_train
        except Exception as e2:
            st.error(f"Erro ao carregar dados (caminho alternativo): {e2}")
            return None, None

# Função para análise de correlação
def analyze_correlations(df):
    """Analisa correlações usando Pearson e Random Forest"""
    
    # Correlação de Pearson
    correlations = df.corr(method='pearson')['gender'].drop('gender')
    top9_pearson = correlations.sort_values(key=abs, ascending=False).head(9)
    
    # Random Forest para importância de atributos
    X = df.drop(columns=['gender'])
    y = df['gender']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top9_rf = importances.sort_values(ascending=False).head(9)
    
    # Atributos em comum
    best_attributes = list(set(top9_pearson.index) & set(top9_rf.index))
    
    return top9_pearson, top9_rf, best_attributes

# Função para treinar modelos
@st.cache_resource
def train_models(X, y):
    """Treina todos os modelos"""
    
    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelos e hiperparâmetros
    models_config = {
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {
            'max_depth': [3, 5, 7, None],
            'criterion': ['gini', 'entropy']
        }),
        "kNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7]
        }),
        "Naive Bayes": (GaussianNB(), {}),
        "SVM Linear": (LinearSVC(random_state=42, dual=False, max_iter=10000), {
            'C': [0.1, 1, 10]
        })
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    # Validação cruzada
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for name, (model, grid) in models_config.items():
        st.info(f"Treinando {name}...")
        
        if grid:
            grid_search = GridSearchCV(model, grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_scaled, y)
            final_model = grid_search.best_estimator_
        else:
            final_model = model.fit(X_scaled, y)
        
        # Validação cruzada
        scores = cross_val_score(final_model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Predição final
        y_pred = final_model.predict(X_scaled)
        
        results[name] = {
            'model': final_model,
            'scaler': scaler,
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'predictions': y_pred
        }
        
        if results[name]['cv_mean'] > best_score:
            best_score = results[name]['cv_mean']
            best_model = name
    
    return results, best_model

# Interface principal
def main():
    st.title("🚀 Dashboard CSI - Classificação de Gênero")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Configurações")
    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["🏠 Visão Geral", "📈 Análise Exploratória", "🎯 Seleção de Atributos", 
         "🤖 Modelos de ML", "🔮 Predições", "📋 Relatórios"]
    )
    
    # Carregar dados
    df_full, df_train = load_data()
    
    if df_full is None:
        st.error("❌ Erro ao carregar dados. Verifique os caminhos dos arquivos.")
        return
    
    # Página selecionada
    if page == "🏠 Visão Geral":
        show_overview(df_full, df_train)
    elif page == "📈 Análise Exploratória":
        show_exploratory_analysis(df_full)
    elif page == "🎯 Seleção de Atributos":
        show_feature_selection(df_full)
    elif page == "🤖 Modelos de ML":
        show_ml_models(df_train)
    elif page == "🔮 Predições":
        show_predictions(df_train)
    elif page == "📋 Relatórios":
        show_reports(df_train)

# Página de visão geral
def show_overview(df_full, df_train):
    st.header("🏠 Visão Geral do Projeto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Sobre o Projeto
        
        Este dashboard apresenta uma análise completa de **classificação de gênero** usando dados **CSI (Channel State Information)** 
        coletados de dispositivos Raspberry Pi.
        
        **Objetivo:** Desenvolver modelos de Machine Learning para identificar o gênero de uma pessoa baseado em 
        sinais de rádio capturados pelo ambiente.
        
        **Tecnologia:** Utilizamos técnicas avançadas de processamento de sinais e algoritmos de ML para 
        extrair padrões únicos de cada gênero.
        """)
        
        st.markdown("""
        ### 🔬 Metodologia
        
        1. **Coleta de Dados:** 4000 amostras de 10 pessoas (400 por pessoa)
        2. **Pré-processamento:** Análise de correlação e seleção de atributos
        3. **Modelagem:** 4 algoritmos diferentes com validação cruzada
        4. **Avaliação:** Métricas de performance e comparação de modelos
        """)
    
    with col2:
        st.metric("📊 Total de Amostras", f"{len(df_full):,}")
        st.metric("👥 Pessoas", "10")
        st.metric("🔢 Atributos", f"{len(df_full.columns)-1}")
        st.metric("🎯 Classes", "2 (Homem/Mulher)")
    
    st.markdown("---")
    
    # Estatísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("📈 Distribuição de Gênero")
        gender_counts = df_full['gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=['Mulher', 'Homem'], 
                    title="Distribuição das Classes")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Estatísticas dos Dados")
        st.dataframe(df_full.describe().round(2))
    
    with col3:
        st.subheader("🔍 Informações do Dataset")
        buffer = st.empty()
        buffer.info(f"""
        **Shape:** {df_full.shape}
        **Colunas:** {len(df_full.columns)}
        **Tipos:** {df_full.dtypes.value_counts().to_dict()}
        **Valores únicos:** {df_full.nunique().sum()}
        """)
    
    with col4:
        st.subheader("📋 Amostras")
        st.dataframe(df_full.head())

# Página de análise exploratória
def show_exploratory_analysis(df):
    st.header("📈 Análise Exploratória dos Dados")
    
    # Seleção de atributos para visualização
    st.subheader("🎨 Visualizações Interativas")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Seleção de atributos
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('gender')
        
        selected_features = st.multiselect(
            "Selecione atributos para visualizar:",
            numeric_cols,
            default=numeric_cols[:5]
        )
        
        plot_type = st.selectbox(
            "Tipo de gráfico:",
            ["Histograma", "Box Plot", "Scatter Plot", "Correlação"]
        )
    
    with col2:
        if selected_features:
            if plot_type == "Histograma":
                fig = make_subplots(rows=len(selected_features), cols=1, 
                                  subplot_titles=selected_features)
                
                for i, feature in enumerate(selected_features, 1):
                    fig.add_trace(
                        go.Histogram(x=df[feature], name=feature, showlegend=False),
                        row=i, col=1
                    )
                
                fig.update_layout(height=300*len(selected_features), title="Distribuição dos Atributos")
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plot":
                fig = px.box(df[selected_features], title="Box Plot dos Atributos")
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Scatter Plot":
                if len(selected_features) >= 2:
                    x_feature = st.selectbox("Atributo X:", selected_features)
                    y_feature = st.selectbox("Atributo Y:", selected_features)
                    
                    fig = px.scatter(df, x=x_feature, y=y_feature, color='gender',
                                   title=f"Scatter Plot: {x_feature} vs {y_feature}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selecione pelo menos 2 atributos para scatter plot")
            
            elif plot_type == "Correlação":
                corr_matrix = df[selected_features + ['gender']].corr()
                fig = px.imshow(corr_matrix, title="Matriz de Correlação",
                               color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
    
    # Análise por gênero
    st.subheader("👥 Análise por Gênero")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Estatísticas por gênero
        if selected_features:
            gender_stats = df.groupby('gender')[selected_features[:3]].agg(['mean', 'std']).round(3)
            st.write("**Estatísticas por Gênero:**")
            st.dataframe(gender_stats)
    
    with col2:
        # Comparação de distribuições
        if selected_features:
            feature = st.selectbox("Atributo para comparação:", selected_features)
            
            fig = px.histogram(df, x=feature, color='gender', barmode='overlay',
                              title=f"Distribuição de {feature} por Gênero")
            st.plotly_chart(fig, use_container_width=True)

# Página de seleção de atributos
def show_feature_selection(df):
    st.header("🎯 Seleção de Atributos")
    
    st.info("🔍 Analisando correlações e importância dos atributos...")
    
    # Análise de correlação
    top9_pearson, top9_rf, best_attributes = analyze_correlations(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Correlação de Pearson")
        fig = px.bar(x=top9_pearson.values, y=top9_pearson.index, 
                    orientation='h', title="Top 9 Atributos - Correlação Pearson")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Top atributos por Pearson:**")
        st.dataframe(top9_pearson.to_frame('Correlação').round(4))
    
    with col2:
        st.subheader("🌳 Random Forest - Importância")
        fig = px.bar(x=top9_rf.values, y=top9_rf.index, 
                    orientation='h', title="Top 9 Atributos - Random Forest")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Top atributos por Random Forest:**")
        st.dataframe(top9_rf.to_frame('Importância').round(4))
    
    # Atributos em comum
    st.subheader("🎯 Atributos Selecionados")
    
    if best_attributes:
        st.success(f"✅ **{len(best_attributes)} atributos selecionados:** {', '.join(best_attributes)}")
        
        # Visualização dos atributos selecionados
        df_selected = df[best_attributes + ['gender']].copy()
        df_selected['gender'] = df_selected['gender'].map({0: 'Mulher', 1: 'Homem'})
        
        fig = px.scatter_matrix(df_selected, dimensions=best_attributes, 
                               color='gender', title="Pair Plot dos Atributos Selecionados")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas dos atributos selecionados
        st.write("**Estatísticas dos atributos selecionados:**")
        st.dataframe(df_selected.describe().round(3))
    else:
        st.warning("⚠️ Nenhum atributo em comum encontrado entre os métodos")

# Página de modelos de ML
def show_ml_models(df):
    st.header("🤖 Modelos de Machine Learning")
    
    # Seleção de atributos
    st.subheader("🔧 Configuração dos Modelos")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Atributos disponíveis:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('gender')
        
        # Usar atributos selecionados ou todos
        use_selected = st.checkbox("Usar apenas atributos selecionados", value=True)
        
        if use_selected:
            # Análise de correlação para obter atributos selecionados
            _, _, best_attributes = analyze_correlations(df)
            if best_attributes:
                selected_features = best_attributes
                st.success(f"✅ Usando {len(selected_features)} atributos selecionados")
            else:
                selected_features = numeric_cols[:10]  # Fallback
                st.warning("⚠️ Usando primeiros 10 atributos")
        else:
            selected_features = st.multiselect(
                "Selecione atributos:",
                numeric_cols,
                default=numeric_cols[:10]
            )
        
        # Botão para treinar
        if st.button("🚀 Treinar Modelos", type="primary"):
            if selected_features:
                st.session_state['training'] = True
                st.session_state['features'] = selected_features
            else:
                st.error("❌ Selecione pelo menos um atributo")
    
    with col2:
        st.write("**Configuração dos algoritmos:**")
        st.info("""
        - **Decision Tree:** Grid search em max_depth e criterion
        - **kNN:** Grid search em n_neighbors
        - **Naive Bayes:** Sem hiperparâmetros
        - **SVM Linear:** Grid search em C
        """)
    
    # Treinamento dos modelos
    if st.session_state.get('training', False) and 'features' in st.session_state:
        st.subheader("🔄 Treinando Modelos...")
        
        # Preparar dados
        X = df[st.session_state['features']]
        y = df['gender']
        
        # Treinar modelos
        with st.spinner("Treinando modelos (isso pode levar alguns minutos)..."):
            results, best_model = train_models(X, y)
            st.session_state['results'] = results
            st.session_state['best_model'] = best_model
            st.session_state['training'] = False
        
        st.success("✅ Treinamento concluído!")
    
    # Resultados dos modelos
    if 'results' in st.session_state:
        st.subheader("📊 Resultados dos Modelos")
        
        # Métricas comparativas
        metrics_df = pd.DataFrame({
            'Modelo': list(st.session_state['results'].keys()),
            'CV Score (Média)': [results['cv_mean'] for results in st.session_state['results'].values()],
            'CV Score (Std)': [results['cv_std'] for results in st.session_state['results'].values()],
            'Accuracy': [results['accuracy'] for results in st.session_state['results'].values()],
            'Precision': [results['precision'] for results in st.session_state['results'].values()],
            'Recall': [results['recall'] for results in st.session_state['results'].values()],
            'F1-Score': [results['f1'] for results in st.session_state['results'].values()]
        }).round(4)
        
        # Ordenar por CV Score
        metrics_df = metrics_df.sort_values('CV Score (Média)', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Métricas de Performance:**")
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.write("**Melhor Modelo:**")
            best = st.session_state['best_model']
            st.success(f"🏆 {best}")
            st.metric("CV Score", f"{st.session_state['results'][best]['cv_mean']:.4f}")
        
        # Gráficos de comparação
        st.subheader("📈 Visualizações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CV Scores
            cv_scores = [results['cv_scores'] for results in st.session_state['results'].values()]
            model_names = list(st.session_state['results'].keys())
            
            fig = go.Figure()
            for i, (name, scores) in enumerate(zip(model_names, cv_scores)):
                fig.add_trace(go.Box(y=scores, name=name, boxpoints='all'))
            
            fig.update_layout(title="Distribuição dos CV Scores", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Métricas principais
            metrics_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            
            for metric in metrics_plot:
                values = [st.session_state['results'][model][METRIC_MAPPING[metric]] 
                         for model in model_names]
                fig.add_trace(go.Bar(name=metric, x=model_names, y=values))
            
            fig.update_layout(title="Métricas por Modelo", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# Página de predições
def show_predictions(df):
    st.header("🔮 Predições em Tempo Real")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Treine os modelos primeiro na página 'Modelos de ML'")
        return
    
    st.subheader("📝 Insira os Valores dos Atributos")
    
    # Obter atributos do modelo treinado
    if 'features' in st.session_state:
        features = st.session_state['features']
    else:
        st.error("❌ Modelos não treinados")
        return
    
    # Interface para entrada de dados
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.write("**Valores dos Atributos:**")
        for i, feature in enumerate(features[:len(features)//2]):
            # Obter estatísticas para normalização
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            input_data[feature] = st.slider(
                f"{feature}:",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                help=f"Média: {mean_val:.2f}, Std: {std_val:.2f}"
            )
    
    with col2:
        st.write("**Valores dos Atributos (continuação):**")
        for i, feature in enumerate(features[len(features)//2:]):
            # Obter estatísticas para normalização
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            input_data[feature] = st.slider(
                f"{feature}:",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                step=float((max_val - min_val) / 100),
                help=f"Média: {mean_val:.2f}, Std: {std_val:.2f}"
            )
    
    # Botão de predição
    if st.button("🔮 Fazer Predição", type="primary"):
        st.subheader("📊 Resultados da Predição")
        
        # Preparar dados para predição
        input_df = pd.DataFrame([input_data])
        
        # Fazer predições com todos os modelos
        predictions = {}
        probabilities = {}
        
        for model_name, model_info in st.session_state['results'].items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Aplicar scaler
            input_scaled = scaler.transform(input_df)
            
            # Predição
            pred = model.predict(input_scaled)[0]
            predictions[model_name] = pred
            
            # Probabilidade (se disponível)
            try:
                prob = model.predict_proba(input_scaled)[0]
                probabilities[model_name] = prob
            except:
                probabilities[model_name] = None
        
        # Exibir resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predições dos Modelos:**")
            for model_name, pred in predictions.items():
                gender = "Homem" if pred == 1 else "Mulher"
                confidence = "Alta" if probabilities[model_name] is not None else "N/A"
                
                if pred == 1:
                    st.success(f"✅ {model_name}: {gender} (Confiança: {confidence})")
                else:
                    st.info(f"✅ {model_name}: {gender} (Confiança: {confidence})")
        
        with col2:
            st.write("**Probabilidades (se disponível):**")
            for model_name, prob in probabilities.items():
                if prob is not None:
                    prob_homem = prob[1] if len(prob) > 1 else prob[0]
                    prob_mulher = prob[0] if len(prob) > 1 else 1 - prob[0]
                    
                    st.write(f"**{model_name}:**")
                    st.progress(prob_homem)
                    st.write(f"Homem: {prob_homem:.3f}")
                    st.progress(prob_mulher)
                    st.write(f"Mulher: {prob_mulher:.3f}")
                else:
                    st.write(f"**{model_name}:** Probabilidade não disponível")
        
        # Resumo estatístico
        st.subheader("📈 Resumo Estatístico")
        
        # Contar predições
        pred_counts = pd.Series(predictions.values()).value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Modelos", len(predictions))
        
        with col2:
            if 1 in pred_counts:
                st.metric("Predições 'Homem'", pred_counts[1])
            else:
                st.metric("Predições 'Homem'", 0)
        
        with col3:
            if 0 in pred_counts:
                st.metric("Predições 'Mulher'", pred_counts[0])
            else:
                st.metric("Predições 'Mulher'", 0)
        
        # Gráfico de predições
        if probabilities:
            fig = go.Figure()
            
            for model_name, prob in probabilities.items():
                if prob is not None:
                    prob_homem = prob[1] if len(prob) > 1 else prob[0]
                    fig.add_trace(go.Bar(name=model_name, x=['Mulher', 'Homem'], 
                                       y=[1-prob_homem, prob_homem]))
            
            fig.update_layout(title="Probabilidades por Modelo", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

# Página de relatórios
def show_reports(df):
    st.header("📋 Relatórios e Exportação")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Treine os modelos primeiro na página 'Modelos de ML'")
        return
    
    st.subheader("📊 Relatório Completo")
    
    # Gerar relatório
    report_data = []
    
    for model_name, model_info in st.session_state['results'].items():
        report_data.append({
            'Modelo': model_name,
            'CV Score (Média)': f"{model_info['cv_mean']:.4f}",
            'CV Score (Std)': f"{model_info['cv_std']:.4f}",
            'Accuracy': f"{model_info['accuracy']:.4f}",
            'Precision': f"{model_info['precision']:.4f}",
            'Recall': f"{model_info['recall']:.4f}",
            'F1-Score': f"{model_info['f1']:.4f}"
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Exibir relatório
    st.write("**Relatório de Performance dos Modelos:**")
    st.dataframe(report_df, use_container_width=True)
    
    # Estatísticas do dataset
    st.subheader("📈 Estatísticas do Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Informações Gerais:**")
        st.info(f"""
        - **Total de amostras:** {len(df):,}
        - **Atributos utilizados:** {len(st.session_state.get('features', []))}
        - **Distribuição de classes:** {df['gender'].value_counts().to_dict()}
        - **Melhor modelo:** {st.session_state.get('best_model', 'N/A')}
        """)
    
    with col2:
        st.write("**Atributos Selecionados:**")
        if 'features' in st.session_state:
            features = st.session_state['features']
            for i, feature in enumerate(features):
                st.write(f"{i+1}. {feature}")
        else:
            st.write("Nenhum atributo selecionado")
    
    # Download de dados
    st.subheader("💾 Download de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download do relatório
        csv_report = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Relatório CSV",
            data=csv_report,
            file_name="relatorio_modelos.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download dos dados de treinamento
        if 'features' in st.session_state:
            features = st.session_state['features']
            df_selected = df[features + ['gender']]
            csv_data = df_selected.to_csv(index=False)
            st.download_button(
                label="📥 Download Dados Selecionados",
                data=csv_data,
                file_name="dados_selecionados.csv",
                mime="text/csv"
            )
    
    # Gráficos finais
    st.subheader("🎨 Visualizações Finais")
    
    if 'results' in st.session_state:
        # Radar chart das métricas
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        for model_name in st.session_state['results'].keys():
            values = []
            for metric in metrics:
                metric_key = METRIC_MAPPING[metric]
                values.append(st.session_state['results'][model_name][metric_key])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparação de Métricas por Modelo"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Executar aplicação
if __name__ == "__main__":
    main()
