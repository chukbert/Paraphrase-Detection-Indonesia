import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import networkx as nx
import matplotlib.pyplot as plt
from mpstemmer import MPStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import sys, os
from amrlib.evaluate.smatch_enhanced import compute_smatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string



stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_sentence(list_of_sentences):
    preprocessed = []
    for sentence in list_of_sentences:
        sentence = sentence.translate(str.maketrans('','',string.punctuation))
        sentence = sentence.lower()
        sentence = stemmer.stem(sentence)
        preprocessed.append(sentence)
    return preprocessed

def extract_js(df):
    all_sentences = np.concatenate([df['preprocessed_sentence1'], df['preprocessed_sentence2']])
    vec = CountVectorizer(binary=True)
    vec.fit(all_sentences)
    X1_train = vec.transform(df['preprocessed_sentence1']).toarray()
    X2_train = vec.transform(df['preprocessed_sentence2']).toarray()
    feat_1_train = [jaccard_score(x1, x2, average='binary') for x1, x2 in zip(X1_train, X2_train)]
    
    df_feat_train = pd.DataFrame(feat_1_train, columns=['Jaccard_Score'])
    
    return df_feat_train

def predict(feature, model):
    y_pred = model.predict(feature)
    return y_pred

def print_validasi(model_option, model, df, validation=False):        
    X_test = df.drop(columns=['label'])
    y_test = df['label']
    if validation:
        X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42, stratify=y_test)
        if model_option != "Model LSA PageRank":
            model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    st.write(f"Akurasi : **{accuracy:.3f}**")
    st.write(f"Presisi : **{precision:.3f}**")
    st.write(f"Recall  : **{recall:.3f}**")
    st.write(f"F1      : **{f1:.3f}**")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

from amr-tst-indo.visualize import render
from PIL import Image

def show_amr(df):
    amr1 = df.iloc[0, 0]
    output_path = "graph.png"
    output_path = render(amr1, output_path)
    with Image.open(output_path) as im:
        st.image(im, caption='AMR kalimat 1', use_column_width=True)
    
    amr2 = df.iloc[0, 1]
    output_path = "graph.png"
    output_path = render(amr2, output_path)
    with Image.open(output_path) as im:
        st.image(im, caption='AMR kalimat 2', use_column_width=True)
    

# Fungsi untuk menampilkan plot graph merged
import networkx as nx
import matplotlib.pyplot as plt
import ast
def get_instance(v, triples):
    for triple in triples:
        if triple[0]==v and triple[1]==":instance":
            return triple[2]
        
def plot_graph_merged(row):
    G = nx.DiGraph()
    merged_amr = ast.literal_eval(row.iloc[0,2])
    for triple in merged_amr:
        print(triple)
        if triple[1] != ':instance':
            G.add_edge(triple[0], triple[2])

    mapping = {}
    for node in G.nodes:
        if (node[0]=="z" or node[0]=="y") and node[1:].isdigit():
            mapping[node] = get_instance(node, merged_amr)
    G = nx.relabel_nodes(G, mapping)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=1000)
    pos = nx.spring_layout(G, k=10) # 'k' mengontrol jarak antar node
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    for node, (x, y) in pos.items():
        plt.text(x, y+0.1, s=f"{pagerank[node]:.3f}", horizontalalignment='center', fontsize=8, color='red')
    plt.title("Merged Graph AMR")
    st.pyplot(plt)


def main():
    # Load Model
    model_jaccard = joblib.load('model/best_model_skor_kesamaan.pkl')
    model_smatch = joblib.load('model/best_model_smatch.pkl')
    model_combined = joblib.load('model/best_model_smatch_js.pkl')
    model_lsa_tfidf = joblib.load('model/best_model_lsa_tfidf.joblib')
    model_lsa_pagerank = joblib.load('model/best_model_lsa_pagerank.pkl')
    model_lsa_pagerank_100 = joblib.load('model/model_lsa_pagerank_100.pkl')
    row_number = None
    
    st.set_page_config(page_title="App Deteksi Parafrasa", page_icon="📝")
    st.subheader("Aplikasi Deteksi Parafrasa Untuk Pasangan Kalimat Bahasa Indonesia Menggunakan Abstract Meaning Representation dan Latent Semantic Analysis 📝")
    
    st.write("""Pilih model yang digunakan dan masukkan dua kalimat atau pilih kalimat dari dataset 
            untuk mendeteksi apakah kedua kalimat merupakan parafrasa atau bukan parafrasa.""")

    # Pilih model untuk digunakan
    model_option = st.selectbox("Pilih model untuk prediksi:", [
        "Model Jaccard",
        "Model Smatch",
        "Model Kombinasi Jaccard dan Smatch",
        "Model LSA TF-IDF",
        "Model LSA PageRank"
    ])
    
    # Dataset yang tersedia
    datasets = {
        "MSRP (latih)": ('data/features/feat_train_total_msrp.csv',
                         'data/features/feat_exp_2_a_train.csv',
                         'data/features/feat_exp_2_c_train.csv',
                         'data/features/feat_exp_3_a_train.csv',
                         'data/features/feat_exp_3_b_train.csv',
                         'data/features/merged_amr_train.csv'),
        "MSRP (terjemahan otomatis)": ('data/features/feat_test_total_msrp.csv',
                                       'data/features/feat_exp_2_a_test.csv',
                                       'data/features/feat_exp_2_c_test.csv',
                                       'data/features/feat_exp_3_a_test.csv',
                                       'data/features/feat_exp_3_b_test.csv',
                                       'data/features/merged_amr_test.csv'),
        "MSRP (terjemahan manusia)": ('data/features/feat_test_100_total_msrp.csv',
                                      'data/features/feat_exp_2_a_test_100.csv',
                                      'data/features/feat_exp_2_c_test_100.csv',
                                      'data/features/feat_exp_3_a_test_100.csv',
                                      'data/features/feat_exp_3_b_test_100.csv',
                                      'data/features/merged_amr_test_100.csv')
    }
    
    # Pilihan antara input manual atau menggunakan dataset
    if model_option == "Model Jaccard":
        option = st.selectbox("Pilih metode input:", ["Masukkan kalimat manual", "Gunakan dataset yang disediakan"])
    else:
        option = "Gunakan dataset yang disediakan"
        
    if option == "Masukkan kalimat manual":
        sentence1 = st.text_area("Masukkan kalimat pertama:", height=100)
        sentence2 = st.text_area("Masukkan kalimat kedua:", height=100)
        label = None
    else:
        dataset_choice = st.selectbox("Pilih dataset:", list(datasets.keys()))
        dataset_df = pd.read_csv(datasets[dataset_choice][0])
        max_rows = len(dataset_df)
        
        row_number = st.number_input(f"Masukkan nomor baris (1-{max_rows}):", min_value=0, max_value=max_rows, step=1)
        if (row_number !=0):
            selected_row = dataset_df.loc[row_number - 1]
            sentence1 = selected_row['sentence1']
            sentence2 = selected_row['sentence2']
            label = selected_row['label']
            st.write("**Kalimat pertama:**", sentence1)
            st.write("**Kalimat kedua:**", sentence2)
            st.write("**Label yang sebenarnya:**", "Parafrasa" if label == 1 else "Bukan Parafrasa")
        else:
            sentence1 = None
            sentence2 = None
                        
    # Tombol untuk melakukan inferensi
    if st.button("Deteksi Parafrasa"):
        if sentence1 and sentence2:
            
            data = {'sentence1': [sentence1], 'sentence2': [sentence2]}
            df = pd.DataFrame(data)
            
            if model_option == "Model Jaccard":
                df['preprocessed_sentence1'] = preprocess_sentence(df['sentence1'])
                df['preprocessed_sentence2'] = preprocess_sentence(df['sentence2'])
                js = extract_js(df)
                paraphrase  = predict(js, model_jaccard)
                st.subheader("Hasil Deteksi:")
                if option == "Masukkan kalimat manual":
                    st.write(sentence1)
                    st.write(sentence2)
                st.write(f"Jaccard Score: **{js.loc[0]['Jaccard_Score']:.5f}**")
                
            elif model_option == "Model Smatch":
                feat = pd.DataFrame([selected_row])
                feat = feat[['feat_smatch']]
                
                amr = pd.read_csv(datasets[dataset_choice][0])
                amr = amr[['amr1', 'amr2']]
                amr = pd.DataFrame([amr.loc[row_number - 1]])
                show_amr(amr)
                
                paraphrase  = predict(feat, model_smatch)
                st.subheader("Hasil Deteksi:")
                st.write(f"Smatch Score: **{feat.iloc[0]['feat_smatch']:.5f}**")
            
            elif model_option == "Model Kombinasi Jaccard dan Smatch":
                feat = pd.read_csv(datasets[dataset_choice][2])
                feat = pd.DataFrame([feat.loc[row_number - 1]])
                feat = feat.drop(columns='label')
                
                amr = pd.read_csv(datasets[dataset_choice][0])
                amr = amr[['amr1', 'amr2']]
                amr = pd.DataFrame([amr.loc[row_number - 1]])
                show_amr(amr)
                
                paraphrase  = predict(feat, model_combined)
                st.subheader("Hasil Deteksi:")
                st.write(f"Smatch Score: **{feat.iloc[0]['feat_smatch']:.5f}**")
                st.write(f"Jaccard Score Konsep AMR: **{feat.iloc[0]['Jaccard_Score']:.5f}**")
            
            elif model_option == "Model LSA TF-IDF":
                feat = pd.read_csv(datasets[dataset_choice][3])
                feat = pd.DataFrame([feat.loc[row_number - 1]])
                feat = feat.drop(columns='label')
            
                amr = pd.read_csv(datasets[dataset_choice][0])
                amr = amr[['amr1', 'amr2']]
                amr = pd.DataFrame([amr.loc[row_number - 1]])
                show_amr(amr)
                
                paraphrase  = predict(feat, model_lsa_tfidf)
                st.subheader("Hasil Deteksi:")
                st.write("Ukuran dictionary: **11961**")
                st.write("Faktor LSA: **3%**")
                st.write("Dimensi Akhir representasi kalimat: **716**")
                
            elif model_option == "Model LSA PageRank":
                feat = pd.read_csv(datasets[dataset_choice][4])
                feat = pd.DataFrame([feat.loc[row_number - 1]])
            
                amr = pd.read_csv(datasets[dataset_choice][5])
                amr = amr[['amr1', 'amr2', 'merged_amr']]
                amr = pd.DataFrame([amr.loc[row_number - 1]])
                show_amr(amr)
                plot_graph_merged(amr)

                if dataset_choice != "MSRP (terjemahan manusia)":                
                    paraphrase  = predict(feat, model_lsa_pagerank)
                    st.subheader("Hasil Deteksi:")
                    st.write("Ukuran dictionary: **14675**")
                    st.write("Faktor LSA: **3%**")
                    st.write("Dimensi Akhir representasi kalimat: **880**")
                else:
                    paraphrase = predict(feat, model_lsa_pagerank_100)
                    st.subheader("Hasil Deteksi:")
                    st.write("Ukuran dictionary: **12112**")
                    st.write("Faktor LSA: **3%**")
                    st.write("Dimensi Akhir representasi kalimat: **726**")

            if paraphrase:
                st.markdown('<p style="color:green; font-size:20px;">Parafrasa</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red; font-size:20px;">Bukan Parafrasa</p>', unsafe_allow_html=True)
        
        elif row_number == 0:
            if model_option == "Model Jaccard":
                validate = False
                if dataset_choice == "MSRP (latih)":
                    validate = True
                temp_df = pd.read_csv(datasets[dataset_choice][1])
                temp_df = temp_df[['Jaccard_Score', 'label']]
                print_validasi(model_option, model_jaccard, temp_df, validate)
                
            elif model_option == "Model Smatch":
                validate = False
                if dataset_choice == "MSRP (latih)":
                    validate = True
                temp_df = pd.read_csv(datasets[dataset_choice][0])
                temp_df = temp_df[['feat_smatch', 'label']]
                print_validasi(model_option, model_smatch, temp_df, validate)
                
            elif model_option == "Model Kombinasi Jaccard dan Smatch":
                validate = False
                if dataset_choice == "MSRP (latih)":
                    validate = True
                temp_df = pd.read_csv(datasets[dataset_choice][2])
                print_validasi(model_option, model_combined, temp_df, validate)
                
            elif model_option == "Model LSA TF-IDF":
                validate = False
                if dataset_choice == "MSRP (latih)":
                    validate = True
                temp_df = pd.read_csv(datasets[dataset_choice][3])
                print_validasi(model_option, model_lsa_tfidf, temp_df, validate)
            elif model_option == "Model LSA PageRank":
                validate = False
                if dataset_choice == "MSRP (latih)":
                    validate = True
                temp_df = pd.read_csv(datasets[dataset_choice][4])
                label_df = pd.read_csv(datasets[dataset_choice][0])
                label_df = label_df[['label']]
                temp_df = pd.concat([temp_df, label_df], axis=1)
                if dataset_choice != "MSRP (terjemahan manusia)":                
                    print_validasi(model_option, model_lsa_pagerank, temp_df, validate)
                else:
                    print_validasi(model_option, model_lsa_pagerank_100, temp_df, validate)
                
        else:
            st.error("Silakan masukkan kedua kalimat.")

if __name__ == "__main__":
    main()
