import pandas as pd
import numpy as np
import ast
import re
import os
import h5py
import pickle
import wfdb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from scipy.signal import resample
from scipy.ndimage import zoom


def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-2] == 'ptbxl':
        # load and convert annotation data
        Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'SPH':
        # load and convert annotation data
        Y = pd.read_csv(path + 'sph_database.csv', index_col='ECG_ID')
        Y.AHA_Code = Y.AHA_Code.apply(lambda x: re.split(r'[;]', x))
        # Load raw signal data
        X = load_raw_data_sph(Y, sampling_rate, path)

    return X, Y


def load_raw_data_sph(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            new_data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            data = [h5py.File(path + 'records500/' + str(f) + '.h5', 'r')['ecg'][()] for f in tqdm(df.index)]
            new_signals = []
            for i in range(len(data)):
                data[i] = data[i][:, :5000]
                new_signals.append(data[i])
            ori_data = np.array(new_signals)
            new_data = []
            for i in range(ori_data.shape[0]):
                down_sig = np.array([zoom(channel.astype(np.float32), .2) for channel in ori_data[i]])
                new_data.append(down_sig)
            new_data = np.array(new_data)
            # print(new_data.shape)  # (25770, 12, 1000)
            pickle.dump(new_data, open(path + 'raw100.npy', 'wb'), protocol=4)
        return new_data

    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [h5py.File(path + 'records500/' + str(f) + '.h5',  'r')['ecg'][()] for f in tqdm(df.index)]
            # print(len(data))  # 25770
            new_signals = []
            for i in range(len(data)):
                data[i] = data[i][:, :5000]
                # print(data[i].shape)  # (12, 5000)
                new_signals.append(data[i])
            data = np.array(new_signals)
            # print(data.shape)  # (25770, 12, 5000)
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
        return data

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path + 'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path + 'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
    return data


def sph_compute_label_aggregations(df, folder, ctype):
    df['aha_codes_len'] = df.AHA_Code.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder + 'code.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        # 44
        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic:
                if "+" in key:
                    key = key.split('+')[0]
                tmp.append(key)
            return list(set(tmp))

        # 11
        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic:
                if "+" in key:
                    key = key.split('+')[0]
                match_item = diag_agg_df[diag_agg_df['Code'] == int(key)]
                category_item = match_item.index.item()
                tmp.append(category_item)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic:
                if "+" in key:
                    key = key.split('+')[0]
                if key == '1':
                    category_item = 'Normal'
                else:
                    category_item = 'AbNormal'
                tmp.append(category_item)
            return list(set(tmp))

        # diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        diag_agg_df = aggregation_df
        if ctype == 'diagnostic':
            df['diagnostic'] = df.AHA_Code.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.AHA_Code.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.AHA_Code.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))

    return df


def select_data(XX, YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass
    # print(mlb.classes_)

    return X, Y, y, mlb
