import pandas as pd
import plotly.express as px
import streamlit as st
from predict import predict

# Data
cp_df = {
    1: "Angina típica",
    2: "Angina atípica",
    3: "Dor não anginosa",
    4: "Assintomático",
}
sex_df = {0: "Feminino", 1: "Masculino"}
restecg_df = {
    0: "Normal",
    1: "Tendo anormalidade na onda ST-T",
    2: "Mostrando hipertrofia ventricular esquerda provável ou definitiva",
}
slope_df = {1: "Inclinando-se para cima", 2: "Plano", 3: "Inclinando-se para baixo"}
thal_df = {3: "Normal", 6: "Defeito fixo", 7: "Defeito reversível"}

# config
st.set_page_config(page_title="Predição de doenças cardíacas", page_icon="🫀")


# caching dataframe
@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df


# cachingg model
@st.cache_resource
def run_model(inputs):
    return predict(inputs)


df = load_data("model/Heart_disease_cleveland_new.csv")

st.markdown(
    """
Este MVP foi construído baseado em um conjunto de dados de doenças cardíacas feita em Cleveland Clinic Foundation. Esse conjunto de dados é muito conhecido na comunidade científica, e por isso nos proporcia outros experiementos para comparar o real desempenho do nosso modelo. como resultado obtivemos um modelo com acuracia de 94% 

Esse é um MVP para *auxílio* de diagnóstico de doenças cardíacas. Não deve ser usado por não profissionais da área médica, nem substituir um médico profissinal.
            """
)
st.subheader("Prencha abaixo os campos para saber o diagnóstico")

with st.form("Formulario"):
    col1, col2 = st.columns(2)
    with col1:

        age = st.number_input("Digite a idade", min_value=1, key="age")
        sex = st.radio(
            "Sexo", options=[0, 1], format_func=lambda x: sex_df[x], key="sex"
        )
        cp = st.selectbox(
            "Tipo de dor no peito:",
            options=[1, 2, 3, 4],
            format_func=lambda x: cp_df[x],
            key="cp",
        )
        trestbps = st.number_input(
            "Pressão arterial em repouso (mmHg)", min_value=0, key="trestbps"
        )
        chol = st.number_input("Colesterol sérico em mg/dl", min_value=0, key="chol")
        fbs = st.checkbox("Glicemia de jejum > 120 mg/dl ?", value=0, key="fbs")
        restecg = st.selectbox(
            "Resultados do eletrocardiograma em repouso",
            options=[0, 1, 2],
            format_func=lambda x: restecg_df[x],
            key="restecg",
        )
    with col2:
        thalach = st.number_input(
            "Frequência cardíaca máxima", min_value=1, key="thalach"
        )
        exang = st.checkbox("Angina induzida por exercício ?", value=0, key="exang")
        oldpeak = st.number_input(
            "Depressão do segmento ST induzida pelo exercício em relação ao repouso.",
            min_value=1.0,
            key="oldpeak",
        )
        slope = st.radio(
            "Inclinação do segmento ST de pico do exercício",
            options=[1, 2, 3],
            format_func=lambda x: slope_df[x],
            key="slope",
        )
        ca = st.number_input(
            "Número de grandes vasos (0-3) coloridos por fluoroscopia.",
            min_value=0,
            max_value=3,
            key="ca",
        )
        thal = st.radio(
            "Resultados do teste de estresse nuclear",
            options=[3, 6, 7],
            format_func=lambda x: thal_df[x],
            key="thal",
        )

    submit = st.form_submit_button("Descubra")
    result = st.subheader("")

if submit:
    result_df = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    result.subheader(
        f"Resultado foi {'Ausência de doença cardíaca.' if run_model(result_df) == 0 else 'Presença de doença cardíaca.'}"
    )

# Visualização da distribuição das idades
st.markdown(
    "Este histograma mostra a distribuição da idade dos pacientes no dataset. A boxplot marginal ajuda a visualizar a dispersão e os outliers na distribuição da idade. Podemos observar que a maioria dos pacientes está na faixa etária entre 50 e 60 anos, o que pode indicar que doenças cardíacas são mais comuns nesta faixa etária."
)
fig = px.histogram(df, x="age", nbins=20, title="Distribuição da Idade", marginal="box")
st.plotly_chart(fig, use_container_width=True)

# Matriz de correlação
st.markdown(
    "A matriz de correlação mostra a relação entre todas as variáveis do dataset. Cores mais escuras representam correlações mais fortes (positivas ou negativas). Observamos que a variável thalach (frequência cardíaca máxima) tem uma correlação negativa moderada com a presença de doença cardíaca (num), enquanto oldpeak (depressão do ST) tem uma correlação positiva moderada com a doença cardíaca. Isso sugere que pacientes com frequência cardíaca máxima menor e maior depressão do ST têm maior probabilidade de ter doença cardíaca."
)
corr = df.corr()
fig_corr = px.imshow(corr, text_auto=True, title="Matriz de Correlação")
st.plotly_chart(fig_corr, use_container_width=True)

# Gráfico de dispersão interativo
st.markdown(
    "Neste gráfico de dispersão interativo mostra a relação entre idade e colesterol, com pontos coloridos pela presença de doença cardíaca (num). Pacientes com doença cardíaca (indicados pela cor) tendem a ter níveis de colesterol mais elevados e podem ser encontrados em várias faixas etárias. No entanto, não há uma clara distinção de idade e colesterol para a presença de doença cardíaca, indicando que outros fatores também podem influenciar a condição."
)
fig_scatter = px.scatter(
    df,
    x="age",
    y="chol",
    color="target",
    title="Relação entre Idade e Colesterol colorido por Diagnóstico de Doença Cardíaca",
    labels={"target": "Diagnóstico de Doença Cardíaca (0 = Não, 1 = Sim)"},
    hover_data=["trestbps", "thalach", "oldpeak"],
)

fig_scatter.update_layout(legend_title_text="Doença Cardíaca")
st.plotly_chart(fig_scatter, use_container_width=True)
