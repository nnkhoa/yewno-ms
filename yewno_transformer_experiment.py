import sentence_transformers
import pandas as pd
import keybert
import bertopic
import platform
from flair.embeddings import TransformerDocumentEmbeddings

import numpy as np


def load_data():
    return


def save_data():
    return


if __name__ == '__main__':
    YEWNO_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/ms_yewno_2020.pickle'
    EDF_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/edf_msft/MSFT_Jul2019_2020_linebreak'
    FILTERED_YEWNO_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/filtered_yewno.pickle'
    PERIOD_DATA_PATH = '/Users/khoanguyen/Workspace/dataset/edf_msft/'
    CONCEPT_COUNT_PATH = '/Users/khoanguyen/Workspace/dataset/Yewno/yewno-edf_concept_count.pickle'
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
        PERIOD_DATA_PATH = 'K:\\Lbpam\\DG_Gestion_Quant\\GERANT\\Khoa\\Data\\MSFT\\'

    print('Do shit')
