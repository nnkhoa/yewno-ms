from sentence_transformers import SentenceTransformer
import pandas as pd
from keybert import KeyBERT
import bertopic
import platform
import pickle
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.data import Sentence
import numpy as np
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from matplotlib import cm
import hdbscan

from yellowbrick.cluster import KElbowVisualizer

import time

from tqdm import tqdm


class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_data(self, data):
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, f)


def remove_noise(text):
    if ' -- ' in text:
        text = text[text.index(' -- '):]
    if '. Visit' in text:
        text = text[:text.index('. Visit')]
    if '. visit' in text:
        text = text[:text.index('. visit')]
    if 'Link to Company News' in text:
        text = text[:text.index('Link to Company News')]
    if 'Editor: ' in text:
        text = text[:text.index('Editor: ')]
    while '-0-' in text:
        text = text[:text.index('-0-')]
    while ' SOURCE ' in text:
        text = text[:text.index(' SOURCE ')]
    while 'NOTE:' in text:
        text = text[:text.index('NOTE:')]
    while 'More information can be found' in text:
        text = text[:text.index('More information can be found')]
    while 'CONTACT:' in text:
        text = text[:text.index('CONTACT:')]
    while 'For more information' in text:
        text = text[:text.index('For more information')]
    while 'NOTE TO EDITORS:' in text:
        text = text[:text.index('NOTE TO EDITORS:')]
    while 'can be found at' in text:
        text = text[:text.index('can be found at')]
    while 'View source version' in text:
        text = text[:text.index('View source version')]

    return text


def text_preprocessing(text_df, stop_word=None, writing_style=None):
    corpus = []
    text_list = text_df.text.to_list()

    for text in text_list:
        text = remove_noise(text)
        corpus.append(text)

    token_list = [item.split(' ') for item in corpus]

    text_df['pre_process'] = corpus
    text_df['token'] = token_list

    text_df = text_df[text_df['token'].str.len() > 10]

    text_df.drop(['token'], axis=1, inplace=True)

    return text_df


def plot_doc_emb(doc_emb_data, cluster_label, num_cluster, cluster_center=None):
    defaultcolor = 'black'
    colors_map = cm.get_cmap('viridis', num_cluster)
    colors = colors_map(range(num_cluster))
    axes = [0, 1]

    fig, ax = plt.subplots()
    for i in range(num_cluster):
        xy = doc_emb_data[cluster_label == i]

        if cluster_center:
            xy = np.vstack([xy, cluster_center[i]])

        # print(centroid.shape)
        # print(xy[-1])
        two_dim = PCA(n_components=max(axes)+1).fit_transform(xy)
        print(two_dim[-1])
        color = colors[i] if i > -1 else defaultcolor
        ax.scatter(two_dim[:-1, axes[0]], two_dim[:-1, axes[1]], color=color, label=i, s=10)
        ax.scatter(two_dim[-1, axes[0]], two_dim[-1, axes[1]], color=color, s=40, edgecolor='black')

        annotate_text = f"center#{i}"
        ax.annotate(text=annotate_text, xy=(two_dim[-1, axes[0]], two_dim[-1, axes[1]]), fontsize=5)

    ax.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0), frameon=True)
    plt.show()


def hdbscan_clustering(doc_emb_data):
    start_time = time.time()
    clustering = hdbscan.HDBSCAN().fit(doc_emb_data)
    print('Clustering time: ', time.time() - start_time)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    cluster_cores = clustering.exemplars_

    faux_centroid = [np.mean(core, axis=0) for core in cluster_cores]

    print('Number of Cluster: ', num_clusters)
    print('Number of Noise: ', num_noise)
    print(labels)

    plot_doc_emb(doc_emb_data, labels, num_clusters, faux_centroid)


def kmean_clustering(k_start, k_end, doc_emb_data):
    ss_max = 0
    chosen_k = 0
    chosen_model = None

    for k in range(k_start, k_end+1):
        start_time = time.time()
        clustering = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(doc_emb_data)
        ss = silhouette_score(doc_emb_data, clustering.labels_)
        print(f'k = {k}, clustering time: ', time.time() - start_time)

        if ss > ss_max:
            ss_max = ss
            chosen_k = k
            chosen_model = clustering
            print(f'New best silhouette score = {ss}, k = {chosen_k}')

    # centroid = list(chosen_model.cluster_centers_)

    # plot_doc_emb(doc_emb_data, chosen_model.labels_, chosen_k, cluster_center=centroid)
    # print(cosine_similarity(centroid))

    return chosen_model, chosen_k, ss_max


def kmean_clustering_yellowbrick(k_start, k_end, doc_emb_data, month):
    visualizer = KElbowVisualizer(KMeans(random_state=0), k=(k_start, k_end)).fit(doc_emb_data)

    # print('Optimal k: ', visualizer.elbow_value_)
    visualizer.show(outpath='K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\elbow_kmean_msft\\' + month + '.png')
    # print('Silhouette Score: ', visualizer.k_scores_)
    optimal_k = visualizer.elbow_value_
    elbow_score = visualizer.elbow_score_
    clustering = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(doc_emb_data)

    visualizer.poof()

    print(month)
    print('Optimal k: ', optimal_k)
    print('Elbow Score: ', elbow_score)

    return clustering, optimal_k, silhouette_score(doc_emb_data, clustering.labels_), elbow_score


def keybert_extraction(corpus):
    sentence_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')
    kw_model = KeyBERT(model=sentence_model)

    corpus_kw_list = []

    for doc in tqdm(corpus):
        kw_distance = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
        doc_kw_list = [kw_data[0] for kw_data in kw_distance]
        corpus_kw_list.extend(doc_kw_list)

    return list(set(corpus_kw_list))


# class EDFHandler(DataHandler):
#     def load_data(self):
#         with open(self.data_path, 'rb') as f:
#             self._data = pickle.load(f)
#         return self._data
#
#     def save_data(self, data=None):
#         with open(self.data_path, 'wb') as f:
#             if not data:
#                 pickle.dump(self._data, self.data_path)
#             else:
#                 pickle.dump(data, self.data_path)
#
#     def data_setter(self, data):
#         self._data = data
#
#
# class YewnoHandler(DataHandler):
#     def load_data(self):
#         with open(self.data_path, 'rb') as f:
#             self._data = pickle.load(f)
#         return self._data
#
#     def save_data(self, data=None):
#         with open(self.data_path, 'wb') as f:
#             if not data:
#                 pickle.dump(self._data, self.data_path)
#             else:
#                 pickle.dump(data, self.data_path)
#
#     def data_setter(self, data):
#         self._data = data


if __name__ == '__main__':
    YEWNO_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/ms_yewno_2020.pickle'
    EDF_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/edf_msft/MSFT_Jul2019_2020_linebreak'
    FILTERED_YEWNO_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/filtered_yewno.pickle'
    PERIOD_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/edf_msft/'
    CONCEPT_COUNT_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno-edf_concept_count.pickle'
    YEWNO_CONCEPT_DICT_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno_concept_dict.pickle'
    YEWNO_CONCEPT_EMBEDDINGS_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno_concept_emb.pickle'
    BIGRAM_CONCEPT_COUNT_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno-edf_bigram_concept_count.pickle'
    EDF_EMB_PATH = '/Users/khoanguyen/Workspace/dataset/edf_msft/embeddings/'
    KEYBERT_EMBEDDINGS_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/keybert_emb.pickle'

    monthly_file = ['2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01', '2019-11-01',
                    '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
                    '2020-05-01', '2020-06-01', '2020-07-01']

    # Windows path
    if platform.system() == 'Windows':
        YEWNO_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\yewno_ms_2020.pickle'
        EDF_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\MSFT_Jul2019_2020_linebreak'
        FILTERED_YEWNO_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\filtered_yewno.pickle'
        CONCEPT_COUNT_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\yewno-edf_concept_count.pickle'
        BIGRAM_CONCEPT_COUNT_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\yewno-edf_bigram_concept_count' \
                                    '.pickle '
        YEWNO_CONCEPT_DICT_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\yewno_concept_dict.pickle'
        YEWNO_CONCEPT_EMBEDDINGS_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\yewno_concept_emb.pickle'
        PERIOD_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\'
        EDF_EMB_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\embeddings\\'
        KEYBERT_EMBEDDINGS_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\keybert_emb.pickle'

    # Getting the roberta representation for concept
    '''    
    yewno_dict_handler = DataHandler(YEWNO_CONCEPT_DICT_PATH)
    yewno_dict_df = yewno_dict_handler.load_data()

    yewno_concept_handler = DataHandler(FILTERED_YEWNO_PATH)
    yewno_concept_df = yewno_concept_handler.load_data()

    concept_list = yewno_concept_df['Concept'].tolist()

    concept_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    embeddings = concept_model.encode(concept_list)

    concept_embeddings_df = pd.DataFrame(data=embeddings, index=concept_list)

    yewno_emb_handler = DataHandler(YEWNO_CONCEPT_EMBEDDINGS_PATH)
    yewno_emb_handler.save_data(concept_embeddings_df)
    '''

    # Preprocessing data and getting the roberta representation on EDF Document
    '''
    document_model = SentenceTransformerDocumentEmbeddings('sentence-transformers/stsb-roberta-base-v2')
    
    for month in monthly_file:
        month_data_path = PERIOD_DATA_PATH + month
        edf_handler = DataHandler(month_data_path)
        edf_df = edf_handler.load_data()

        text_preprocessing(edf_df)

        text_list = edf_df['pre_process'].tolist()
        index_list = edf_df.index.tolist()

        doc_emb_list = []
        for text in tqdm(text_list):
            doc_sent = Sentence(text)
            document_model.embed(doc_sent)

            doc_numpy = doc_sent.embedding.detach().numpy()
            doc_emb_list.append(doc_numpy)

        doc_emb_array = np.asarray(doc_emb_list)
        if not os.path.isdir(PERIOD_DATA_PATH + 'embeddings'):
            os.mkdir(PERIOD_DATA_PATH + 'embeddings')

        np.savetxt(PERIOD_DATA_PATH + 'embeddings/' + month + '_no-st-text_emb.csv', doc_emb_array)
    '''

    yewno_emb_handler = DataHandler(YEWNO_CONCEPT_EMBEDDINGS_PATH)
    yewno_emb_df = yewno_emb_handler.load_data()
    yewno_emb_arr = list(yewno_emb_df.to_numpy())

    # similarity_df = pd.DataFrame(index=yewno_emb_df.index)

    # clustering process
    '''
    with open(PERIOD_DATA_PATH + 'clustering_stats_elbow_no-st-text.txt', 'w+') as f:
        f.write('Clustering Stats: \n')

    for month in monthly_file:
        similarity_df = pd.DataFrame(index=yewno_emb_df.index)

        month_doc_emb_path = EDF_EMB_PATH + month + '_no-st-text_emb.csv'
        doc_emb_data = np.genfromtxt(month_doc_emb_path)

        # model, k, sil_score = kmean_clustering(k_start=5, k_end=50, doc_emb_data=doc_emb_data)
        elbow_score = None
        model, k, sil_score, elbow_score = kmean_clustering_yellowbrick(k_start=5, k_end=50, doc_emb_data=doc_emb_data, month=month)

        centroid = list(model.cluster_centers_)

        cluster_labels = model.labels_

        for i in range(len(centroid)):
            similarity_matrix = cosine_similarity(yewno_emb_arr, [centroid[i]])

            similarity_df[i] = similarity_matrix

        save_path = PERIOD_DATA_PATH + month + '_yewno_similarity_elbow.pickle'
        month_similarity_file_handler = DataHandler(save_path)
        month_similarity_file_handler.save_data(similarity_df)

        with open(PERIOD_DATA_PATH + 'clustering_stats_elbow_no-st-text.txt', 'a+') as f:
            f.write(month + '\n')
            f.write(f'Number of clusters: {k}\n')
            f.write(f'Silhouette Score: {sil_score}\n')
            if elbow_score:
                f.write(f'Elbow Score: {elbow_score}\n')
            f.write('\n')

        with open(PERIOD_DATA_PATH + month + '_cluster_elbow-labels.txt', 'w+') as f:
            for label in cluster_labels:
                f.write(f'{label}\n')
    '''
    full_kw_list = []

    for month in monthly_file:
        print(month)
        month_data_path = PERIOD_DATA_PATH + month
        edf_handler = DataHandler(month_data_path)
        edf_df = edf_handler.load_data()

        if 'pre_process' not in edf_df:
            text_preprocessing(edf_df)
            edf_handler.save_data(edf_df)

        text_list = edf_df['pre_process'].tolist()
        index_list = edf_df.index.tolist()

        keyword_list_month = keybert_extraction(text_list)
        full_kw_list.extend(keyword_list_month)

    kw_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    embeddings = kw_model.encode(full_kw_list)

    kw_embeddings_df = pd.DataFrame(data=embeddings, index=full_kw_list)

    keybert_emb_handler = DataHandler(KEYBERT_EMBEDDINGS_PATH)
    keybert_emb_handler.save_data(kw_embeddings_df)
