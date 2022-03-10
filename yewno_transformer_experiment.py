from sentence_transformers import SentenceTransformer
import pandas as pd
# import keybert
import bertopic
import platform
import pickle
from flair.embeddings import TransformerDocumentEmbeddings

import numpy as np


class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        # self._data = None

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_data(self, data):
        with open(self.data_path, 'wb') as f:
            pickle.dump(data, self.data_path)


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
    BIGRAM_CONCEPT_COUNT_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno-edf_bigram_concept_count.pickle'

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
        PERIOD_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\'

    edf_handler = DataHandler(EDF_DATA_PATH)
    edf_df = edf_handler.load_data()

    yewno_dict_handler = DataHandler(YEWNO_CONCEPT_DICT_PATH)
    yewno_dict_df = yewno_dict_handler.load_data()

    yewno_concept_handler = DataHandler(FILTERED_YEWNO_PATH)
    yewno_concept_df = yewno_concept_handler.load_data()

    sample_concept = yewno_concept_df['Concept'].head().tolist()

    print(sample_concept)

    concept_model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')

    embeddings = concept_model.encode(sample_concept)

    print(embeddings)
