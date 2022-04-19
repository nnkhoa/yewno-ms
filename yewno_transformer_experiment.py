from sentence_transformers import SentenceTransformer
import pandas as pd
from keybert import KeyBERT
from bertopic import BERTopic
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
import umap

from yellowbrick.cluster import KElbowVisualizer

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import time
import re

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


def hypernym_dict_extraction(data_row):
    hypernym_list = []
    for entry in data_row:
        hypernym_list.append(entry['title'])
        if 'subtopics' in entry:
            for subentry in entry['subtopics']:
                hypernym_list.append(subentry['title'])
    return hypernym_list


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


def text_preprocessing(text_df, stop_word=None, writing_style=None, save_column='pre_process'):
    corpus = []
    text_list = text_df.text.to_list()
    stop_words = set(stopwords.words('english'))

    for text in text_list:
        text = remove_noise(text)

        text = re.sub(r'[^a-zA-Z.]', ' ', text)
        # convert to lower case
        text = text.lower()
        # remove tags
        text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
        # remove special characters and digits
        text = re.sub(r'(\d)',' ', text)
        # removes emails and mentions (words with @)
        text = re.sub(r"\S*@\S*\s?", " ", text)
        # removes URLs with http
        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-])', ' ', text)
        # removes URLs with www
        # text = re.sub(r'www\S*\s?', ' ', text)
        # Convert to list from string
        token_list = text.split(' ')
        # Lemmatization
        lem = WordNetLemmatizer()
        temp = [lem.lemmatize(word) for word in token_list if word not in stop_words]

        text = " ".join(temp)

        corpus.append(text)

    token_list = [item.split(' ') for item in corpus]

    text_df[save_column] = corpus
    text_df['token'] = token_list

    new_text_df = text_df[text_df['token'].str.len() > 10]

    new_text_df.drop(['token'], axis=1, inplace=True)

    return new_text_df


def plot_doc_emb(doc_emb_data, cluster_label, num_cluster, cluster_center=None):
    defaultcolor = 'black'
    colors_map = cm.get_cmap('viridis', num_cluster)
    colors = colors_map(range(num_cluster))
    axes = [0, 1]

    map_data = umap.UMAP(n_neighbors=15,
                         n_components=2,
                         min_dist=0.0,
                         metric='cosine',
                         random_state=10).fit_transform(doc_emb_data)

    fig, ax = plt.subplots()
    # for i in range(num_cluster):
    #     xy = doc_emb_data[cluster_label == i]
    #
    #     if cluster_center:
    #         xy = np.vstack([xy, cluster_center[i]])
    #
    #     # print(centroid.shape)
    #     # print(xy[-1])
    #     two_dim = PCA(n_components=max(axes)+1).fit_transform(xy)
    #
    #     color = colors[i] if i > -1 else defaultcolor
    #     if cluster_center:
    #         ax.scatter(two_dim[:-1, axes[0]], two_dim[:-1, axes[1]], color=color, label=i, s=10, c_map='hsv_r')
    #         ax.scatter(two_dim[-1, axes[0]], two_dim[-1, axes[1]], color=color, s=40, edgecolor='black')
    #
    #         annotate_text = f"center#{i}"
    #         ax.annotate(text=annotate_text, xy=(two_dim[-1, axes[0]], two_dim[-1, axes[1]]), fontsize=5)
    #     else:
    #         ax.scatter(two_dim[:, axes[0]], two_dim[:, axes[1]], color=i, label=i, s=5, c_map='hsv_r')

    # ax.legend(loc='upper right', bbox_to_anchor=(0.0, 1.0), frameon=True)
    plt.scatter(map_data[cluster_label >= 0, 0],
                map_data[cluster_label >= 0, 1],
                c=cluster_label[cluster_label >= 0],
                cmap='hsv', s=5)
    plt.colorbar()
    plt.show()


def hdbscan_clustering(doc_emb_data):
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine',
                                random_state=10).fit_transform(doc_emb_data)

    start_time = time.time()
    clustering = hdbscan.HDBSCAN().fit(umap_embeddings)
    print('Clustering time: ', time.time() - start_time)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)

    cluster_cores = clustering.exemplars_

    faux_centroid = [np.mean(core, axis=0) for core in cluster_cores]

    print('Number of Cluster: ', num_clusters)
    print('Number of Noise: ', num_noise)
    print(labels)

    plot_doc_emb(doc_emb_data, labels, num_clusters)


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
    # visualizer.show(outpath='K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\elbow_kmean_msft\\' + month + '.png')
    # print('Silhouette Score: ', visualizer.k_scores_)
    optimal_k = visualizer.elbow_value_
    elbow_score = visualizer.elbow_score_
    clustering = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(doc_emb_data)

    # visualizer.poof()

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


def bertopic_extraction(corpus, save_path):
    roberta_model = SentenceTransformerDocumentEmbeddings
    topic_model = BERTopic(embedding_model='sentence-transformers/stsb-roberta-base-v2')

    topic, prob = topic_model.fit_transform(corpus)

    topic_model.save(save_path)


def get_top_percentile(df_column, percentile=0.97):
    non_negative_value = df_column[df_column > 0]
    threshold = non_negative_value.quantile(percentile)
    top_percentile = non_negative_value[non_negative_value > threshold].index.tolist()
    return len(top_percentile)


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
    YEWNO_CONCEPT_EMBEDDINGS_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno_concept_with-context_emb.pickle'
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
        YEWNO_CONCEPT_EMBEDDINGS_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\yewno_concept_context_emb.pickle'
        PERIOD_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\'
        EDF_EMB_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\embeddings\\'
        KEYBERT_EMBEDDINGS_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\keybert_emb.pickle'

    # Getting the roberta representation for concept
    # '''
    yewno_dict_handler = DataHandler(YEWNO_CONCEPT_DICT_PATH)
    yewno_dict_df = yewno_dict_handler.load_data()

    yewno_concept_handler = DataHandler(YEWNO_DATA_PATH)
    yewno_concept_df = yewno_concept_handler.load_data()
    '''
    concept_list = yewno_dict_df['Concept'].tolist()
    concept_definition = yewno_dict_df['Definition'].tolist()
    concept_info_misc = yewno_dict_df['Misc'].tolist()

    # update concept with misc info for less ambiguity
    if 'Concept-Misc' not in yewno_dict_df:
        concept_with_misc = []
        # concept_with_context = []
        for concept, misc_info in zip(concept_list, concept_info_misc):
            # if misc_info.lower() in concept.lower():
            #     concept_extra_info = concept
            # elif concept.lower() in misc_info.lower():
            #     concept_extra_info = misc_info
            # else:
            concept_extra_info = concept + ' - ' + misc_info
            # concept_with_context.append(concept + ': ' + definition)

            concept_with_misc.append(concept_extra_info)

        yewno_dict_df['Concept-Misc'] = pd.Series(concept_with_misc)
        yewno_dict_handler.save_data(yewno_dict_df)

    concept_misc = yewno_dict_df['Concept-Misc']
    # concept_with_context = [concept + ' ' + definition
    #                         for concept, definition in zip(concept_list, concept_definition)]

    # concept_with_context = [concept + ' ' + definition
    #                         for concept, definition in zip(concept_misc, concept_definition)]

    concept_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    # embeddings = concept_model.encode(concept_with_context)
    embeddings = concept_model.encode(concept_misc)

    concept_embeddings_df = pd.DataFrame(data=embeddings, index=concept_misc)

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

    # load yewno concept embeddings
    # '''
    yewno_emb_handler = DataHandler(YEWNO_CONCEPT_EMBEDDINGS_PATH)
    yewno_emb_df = yewno_emb_handler.load_data()
    yewno_emb_arr = list(yewno_emb_df.to_numpy())
    # '''

    # similarity_df = pd.DataFrame(index=yewno_emb_df.index)

    # clustering process
    '''
    with open(PERIOD_DATA_PATH + 'clustering_stats_no-context_elbow_no-st-text.txt', 'w+') as f:
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

        save_path = PERIOD_DATA_PATH + month + '_yewno_with-context_similarity_elbow.pickle'
        month_similarity_file_handler = DataHandler(save_path)
        month_similarity_file_handler.save_data(similarity_df)

        with open(PERIOD_DATA_PATH + 'clustering_stats_no-context_elbow_no-st-text.txt', 'a+') as f:
            f.write(month + '\n')
            f.write(f'Number of clusters: {k}\n')
            f.write(f'Silhouette Score: {sil_score}\n')
            if elbow_score:
                f.write(f'Elbow Score: {elbow_score}\n')
            f.write('\n')

        with open(PERIOD_DATA_PATH + month + '_cluster_no-context_elbow-labels.txt', 'w+') as f:
            for label in cluster_labels:
                f.write(f'{label}\n')
    '''

    # TODO keyBERT embeddings creation
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
    '''

    '''
    if not os.path.isdir(PERIOD_DATA_PATH + 'bertopic_models'):
        os.mkdir(PERIOD_DATA_PATH + 'bertopic_models')

    for month in tqdm(monthly_file):
        print(month + '\n')
        month_data_path = PERIOD_DATA_PATH + month
        edf_handler = DataHandler(month_data_path)
        edf_df = edf_handler.load_data()

        if 'pre_process' in edf_df:
            edf_df.drop(['pre_process'], axis=1, inplace=True)

        if 'heavy_pre_process' in edf_df:
            edf_df.drop(['heavy_pre_process'], axis=1, inplace=True)

        if 'heavy_pre_process' not in edf_df:
            text_preprocessing(edf_df, save_column='heavy_pre_process')
            edf_handler.save_data(edf_df)

        text_list = edf_df['heavy_pre_process'].tolist()
        save_path = PERIOD_DATA_PATH + 'bertopic_models/' + month + '_heavy-pre-process' + '_bertopic-model'

        bertopic_extraction(text_list, save_path=save_path)
        # similarity_df = pd.DataFrame(index=yewno_emb_df.index)

        # month_doc_emb_path = EDF_EMB_PATH + month + '_no-st-text_emb.csv'
        # doc_emb_data = np.genfromtxt(month_doc_emb_path)
        #
        # hdbscan_clustering(doc_emb_data)
    '''

    # TODO extract top concepts in each month for each type of embeddings
    '''
    file_prefix = ['no-context_ms',      # concept related to MS extracted from yewno
                   'no-context',         # concept extracted from yewno's dictionary
                   'with-context']       # concept extracted from yewno's dictionary with definition as context

    concept_info = ['Concept', 'Definition', 'Type']
    
    for month in tqdm(monthly_file):
        # get num_cluster in month
        sample_results_file = PERIOD_DATA_PATH + month + '_yewno_' + file_prefix[0] + '_similarity_elbow.pickle'
        with open(sample_results_file, 'rb') as f:
            sample_data = pickle.load(f)
        num_cluster = sample_data.columns.tolist()
        del sample_data

        columns = pd.MultiIndex.from_product([num_cluster, file_prefix, concept_info])
        similar_concept = pd.DataFrame(columns=columns)

        for prefix in file_prefix:
            file_path = PERIOD_DATA_PATH + month + '_yewno_' + prefix + '_similarity_elbow.pickle'
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            for cluster in num_cluster:
                similarity_sort = data[cluster].sort_values(ascending=False)
                # top_concepts = similarity_sort[similarity_sort > 0.5].index.tolist()

                # non_negative_similarity = similarity_sort[similarity_sort > 0]
                # threshold = non_negative_similarity.quantile(0.99)
                # top_concepts = non_negative_similarity[non_negative_similarity > threshold].index.tolist()

                top_concepts = similarity_sort.head(10).index.tolist()

                if len(top_concepts) > len(similar_concept.index):
                    similar_concept = similar_concept.reindex(range(0, len(top_concepts)))
                similar_concept[cluster, prefix, 'Concept'] = pd.Series(top_concepts)

                if prefix in ['no-context', 'with-context']:
                    def_list = yewno_dict_df[yewno_dict_df['Concept-Misc'].isin(top_concepts)]
                    for concept in top_concepts:
                        def_row = def_list.loc[def_list['Concept-Misc'] == concept]
                        definition = def_row['Definition'].values[0]
                        concept_type = hypernym_dict_extraction(def_row['Hypernym'].values[0])
                        row = similar_concept.loc[similar_concept[cluster, prefix, 'Concept'] == concept]
                        try:
                            row_index = row.index[0]
                        except IndexError:
                            pass
                        else:
                            similar_concept.at[row_index, (cluster, prefix, 'Definition')] = definition
                            similar_concept.at[row_index, (cluster, prefix, 'Type')] = concept_type

        save_file = month + '_top10_similar_concepts.pickle'
        with open(PERIOD_DATA_PATH + save_file, 'wb') as f:
            pickle.dump(similar_concept, f)
    '''

    '''
    # TODO Plot concept similarity graph
    file_prefix = ['no-context_ms',      # concept related to MS extracted from yewno
                   'no-context',         # concept extracted from yewno's dictionary
                   'with-context']       # concept extracted from yewno's dictionary with definition as context

    concept_info = ['Concept', 'Definition', 'Type']
    top_concept_count_per_cluster = pd.DataFrame()
    for month in tqdm(monthly_file):
        # get num_cluster in month
        sample_results_file = PERIOD_DATA_PATH + month + '_yewno_' + file_prefix[0] + '_similarity_elbow.pickle'
        with open(sample_results_file, 'rb') as f:
            sample_data = pickle.load(f)

        temp = sample_data.apply(lambda x: get_top_percentile(x, percentile=.99), axis=0)

        if len(temp) > len(top_concept_count_per_cluster.index):
            top_concept_count_per_cluster = top_concept_count_per_cluster.reindex(range(0,len(temp)))

        top_concept_count_per_cluster[month] = temp
    '''

    # '''
    top_concept_cluster_monthly = pd.DataFrame()
    for month in monthly_file:
        save_file = month + '_top10_similar_concepts.pickle'
        with open(PERIOD_DATA_PATH + save_file, 'rb') as f:
            data = pickle.load(f)
        # concept_similarity.append(data)

        num_cluster = list(set([column[0] for column in data.columns.tolist()]))

        concept_all_cluster = pd.concat([data[cluster, 'with-context', 'Concept'] for cluster in num_cluster])
        concept_all_cluster.dropna(inplace=True)
        concept_count = concept_all_cluster.value_counts()

        top_concept_cluster_monthly[month] = concept_count
        # print()
    output = PERIOD_DATA_PATH + 'monthly_top_concepts_all_clusters_top10.pickle'

    with open(output, 'wb') as f:
        pickle.dump(top_concept_cluster_monthly, f)
    # '''

