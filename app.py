import pickle
import pandas as pd
import streamlit as st
import json
import os
import string
import emoji
import re
import numpy as np
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from kisaltmalar import kisaltmalar
import nltk
nltk.download('punkt')
from nltk import word_tokenize


fi = open('./assets/languages.json')
config = json.load(fi)
fi.close()

model = pickle.load(open("./assets/logreg_model.pickle", "rb"))
le = pickle.load(open("./assets/le.pickle", "rb"))
cv = pickle.load(open("./assets/cv.pickle", "rb"))
stop_words = [x.strip() for x in open('./assets/stop-words.txt','r', encoding="UTF8").read().split('\n')]


def convert_abbrev(word):
    return kisaltmalar[word] if word in kisaltmalar.keys() else word

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

def preprocess(df):
    df["Haber Gövdesi Cleaned"] = df["Haber Gövdesi"].apply(convert_abbrev_in_text)
    df['Haber Gövdesi Cleaned'] = df["Haber Gövdesi"].apply(lambda x: x.lower())
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: re.sub('[0-9]+', '', x))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: x.replace('"', '').replace("’", '').replace("'", '').replace("”", ''))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: re.sub('a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', x))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: emoji.replace_emoji(x))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda x: re.sub('<.*?>', '', x))
    df['Haber Gövdesi Cleaned'] = df['Haber Gövdesi Cleaned'].apply(lambda text: ' '.join([word for word in text.split() if word.lower() not in stop_words]))
    return df["Haber Gövdesi Cleaned"].values[0]

def analyze(df):
    text = preprocess(df)
    output = model.predict(cv.transform([text]))[0]
    result = output.item()
    return st.success(f"Bu haber '{le.classes_[result]}' kategorisine aittir.")

def main():
    PAGE_TITLE = "Türkçe Haber Sınıflandırma"
    PAGE_ICON = ":news:"

    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    with st.sidebar:
        language_picker = st.selectbox("Language", options=["Türkçe", "English"])

        if language_picker == "Türkçe":
            ui = config["TR"]
        elif language_picker == "English":
            ui = config["EN"]

        options = [
            ui["PAGE_OPTIONS_HOME"],
            ui["PAGE_OPTIONS_UPLOAD"],
            ui["PAGE_OPTIONS_MODELS"],
            ui["PAGE_OPTIONS_FEATURES"],
            ui["PAGE_OPTIONS_ABOUT"]
        ]

        selected = option_menu(
            menu_title="TNC",
            options=options,
            icons=['house-fill', 'cpu-fill', 'box-fill', 'collection-fill', 'info-circle-fill'],
            menu_icon='shield-shaded',
            default_index=0,
            styles={
                "container": {
                    "padding": "5 !important",
                    "background-color": "black"
                },
                "icon": {
                    "color": "white",
                    "font-size": "23px"
                },
                "nav-link": {
                    "color": "white",
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "blue"
                },
                "nav-link-selected": {
                    "background-color": "#02ab21"
                }
            }
        )

    if selected == ui["PAGE_OPTIONS_HOME"]:
        st.title(ui["PAGE_OPTIONS_HOME_TITLE"])

        with st.container(border=True):
            st.markdown(ui["PAGE_OPTIONS_HOME_PARAGRAPH"])
            st.markdown(ui["PAGE_OPTIONS_HOME_STEPS_1"])
            st.markdown(ui["PAGE_OPTIONS_HOME_STEPS_2"])
            st.markdown(ui["PAGE_OPTIONS_HOME_STEPS_3"])
            st.markdown(ui["PAGE_OPTIONS_HOME_STEPS_4"])
            st.markdown(ui["PAGE_OPTIONS_HOME_STEPS_5"])

            st.markdown(ui["PAGE_OPTIONS_HOME_OBJECTIVE_TITLE"])
            st.markdown(ui["PAGE_OPTIONS_HOME_OBJECTIVE_PARAGRAPH"])
            st.markdown(ui["PAGE_OPTIONS_HOME_OBJECTIVE_COL1"])
            st.markdown(ui["PAGE_OPTIONS_HOME_OBJECTIVE_COL2"])

            st.markdown(ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_PARAGRAPH"])

            df = pd.DataFrame({
                ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1"]: [
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_1"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_2"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_3"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_4"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_5"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_6"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_7"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_8"],
                    ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL1_9"],
                ],

                ui["PAGE_OPTIONS_HOME_OBJECTIVE_DF_COL2"]: [
                    489951,
                    305452,
                    230427,
                    122075,
                    100019,
                    71182,
                    68257,
                    65414,
                    33232
                ]
            })

            st.table(df)
            

            st.markdown(ui["PAGE_OPTIONS_HOME_INSTALLATION_TITLE"])
            st.write(ui["PAGE_OPTIONS_HOME_INSTALLATION_PARAGRAPH"])
            st.code("""
                    cd ddi-proje/
                    pip install -r requirements.txt""", language="bash")


    elif selected ==  ui["PAGE_OPTIONS_UPLOAD"]:
        st.title(ui["PAGE_OPTIONS_UPLOAD_TITLE"])

        with st.container(border=True):
            text = st.text_area(ui["PAGE_OPTIONS_UPLOAD_HEADER"])
            res = st.button(ui["PAGE_OPTIONS_UPLOAD_BUTTON"])

            if len(text) > 1 and res:
                analyze(pd.DataFrame({"Haber Gövdesi": [text]}))

                
    elif selected == ui["PAGE_OPTIONS_MODELS"]:
        st.title(ui["PAGE_OPTIONS_MODELS_TITLE"])

        with st.container(border=True):
            st.header(ui["PAGE_OPTIONS_MODELS_HEADER"])
            models_df = pd.read_csv("./assets/scores.csv")
            st.dataframe(models_df)

        with st.container(border=True):
            fig1 = go.Figure(
            data = [
                    go.Bar(name=ui["PAGE_OPTIONS_MODELS_LEGEND_1"], y=models_df["Model Name"], x=models_df["Train Accuracy"], orientation='h'),
                ],
            )
            fig1.update_layout(template='plotly_dark', title=ui["PAGE_OPTIONS_MODELS_GRAPH_TITLE"], width=1000, height=800)
            fig1.update_layout(showlegend=False)
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1)

        with st.container(border=True):
            fig2 = go.Figure(
            data = [
                    go.Bar(name=ui["PAGE_OPTIONS_MODELS_LEGEND_2"], y=models_df["Model Name"], x=models_df["Test Accuracy"], orientation='h'),
                ],
            )
            fig2.update_layout(template='plotly_dark', title=ui["PAGE_OPTIONS_MODELS_GRAPH_TITLE"], width=1000, height=800)
            fig2.update_layout(showlegend=False)
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2)

        with st.container(border=True):
            fig3 = go.Figure(
            data = [
                    go.Bar(name=ui["PAGE_OPTIONS_MODELS_LEGEND_3"], y=models_df["Model Name"], x=models_df["F1"], orientation='h'),
                ],
            )
            fig3.update_layout(template='plotly_dark', title=ui["PAGE_OPTIONS_MODELS_GRAPH_TITLE"], width=1000, height=800)
            fig3.update_layout(showlegend=False)
            fig3.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig3)

        with st.container(border=True):
            fig4 = go.Figure(
            data = [
                    go.Bar(name=ui["PAGE_OPTIONS_MODELS_LEGEND_4"], y=models_df["Model Name"], x=models_df["Precision"], orientation='h'),
                ],
            )
            fig4.update_layout(template='plotly_dark', title=ui["PAGE_OPTIONS_MODELS_GRAPH_TITLE"], width=1000, height=800)
            fig4.update_layout(showlegend=False)
            fig4.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig4)

        with st.container(border=True):
            fig5 = go.Figure(
            data = [
                    go.Bar(name=ui["PAGE_OPTIONS_MODELS_LEGEND_5"], y=models_df["Model Name"], x=models_df["Recall"], orientation='h'),
                ],
            )
            fig5.update_layout(template='plotly_dark', title=ui["PAGE_OPTIONS_MODELS_GRAPH_TITLE"], width=1000, height=800)
            fig5.update_layout(showlegend=False)
            fig5.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig5)

    elif selected == ui["PAGE_OPTIONS_FEATURES"]:
        st.title(ui["PAGE_OPTIONS_FEATURES_TITLE"])

        with st.container(border=True):
            st.markdown(f"- **Kısaltmalar**: {ui['PAGE_OPTIONS_FEATURES_1']}")
            st.markdown(f"- **Harfler**: {ui['PAGE_OPTIONS_FEATURES_2']}")
            st.markdown(f"- **Linkler**: {ui['PAGE_OPTIONS_FEATURES_3']}")
            st.markdown(f"- **Rakamlar**: {ui['PAGE_OPTIONS_FEATURES_4']}")
            st.markdown(f"- **Noktalama işaretleri**: {ui['PAGE_OPTIONS_FEATURES_5']}")
            st.markdown(f"- **Mail adresleri**: {ui['PAGE_OPTIONS_FEATURES_6']}")
            st.markdown(f"- **Emojiler**: {ui['PAGE_OPTIONS_FEATURES_7']}")
            st.markdown(f"- **HTML etiketleri**: {ui['PAGE_OPTIONS_FEATURES_8']}")
            st.markdown(f"- **Etkisiz kelimeler**: {ui['PAGE_OPTIONS_FEATURES_9']}")
            st.markdown(f"- **Frekanslar**: {ui['PAGE_OPTIONS_FEATURES_10']}")
            st.markdown(f"- **Cümleler**: {ui['PAGE_OPTIONS_FEATURES_11']}")
            st.markdown(f"- **Lemmatizasyon**: {ui['PAGE_OPTIONS_FEATURES_12']}")

    elif selected == ui["PAGE_OPTIONS_ABOUT"]:
        st.title(ui["PAGE_OPTIONS_ABOUT_TITLE"])
        with st.container(border=True):
            with st.expander(ui["PAGE_OPTIONS_ABOUT_EXPANDER_TITLE"]):
                st.markdown(ui["PAGE_OPTIONS_ABOUT_EXPANDER_R1"])
                st.markdown(ui["PAGE_OPTIONS_ABOUT_EXPANDER_R2"])


if __name__ == "__main__":
    main()
