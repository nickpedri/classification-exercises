from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd


def train_val_test(df, strat, seed=100):
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    return train, val, test


def prep_iris(df):
    new_df = df.drop(columns=['species_id', 'measurement_id'])
    new_df = new_df.rename(columns={'species_name': 'species'})
    return new_df


def prep_titanic(df):
    titanic = df.drop(columns=['passenger_id', 'pclass', 'embarked', 'deck'])
    titanic = pd.get_dummies(titanic, columns=['sex'], drop_first=True)
    titanic = pd.get_dummies(titanic, columns=['class', 'embark_town'])
    return titanic


def prep_telco(df):
    telco = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    telco['total_charges'] = telco['total_charges'].astype(float)
    telco = pd.get_dummies(telco, columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
                                           'online_security', 'online_backup', 'device_protection', 'tech_support',
                                           'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn',
                                           'contract_type', 'internet_service_type', 'payment_type'], drop_first=True)
    return telco


def titanic_dummies(df):
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['class', 'embark_town'])
    return df


def split_x_y(df, target=''):
    x_df = df.drop(columns=target)
    y_df = df[target]
    return x_df, y_df


def impute_df(df, column='', strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(df[[column]])
    df[[column]] = imputer.transform(df[[column]])
    return df


def evaluate(df, data, model='', target='', show_results=True):
    df['baseline'] = df[data].mode()[0]  # Creates a baseline prediction from the mode of the data
    m_acc = (df[data] == df[model]).mean()
    b_acc = (df[data] == df['baseline']).mean()
    sub_rec = df[df[data] == target]  # Subset of all positive cases for recall
    m_rec = (sub_rec[data] == sub_rec[model]).mean()
    b_rec = (sub_rec[data] == sub_rec['baseline']).mean()
    s_pre = df[df[model] == target]  # Subset of all positive guesses
    m_pre = (s_pre[data] == s_pre[model]).mean()
    sub_bas_pre = df[df['baseline'] == target]
    if sub_bas_pre.empty:
        bas_pre = 0
    else:
        bas_pre = (sub_bas_pre[data] == sub_bas_pre['baseline']).mean()
    if show_results:
        print(f'Model accuracy is: {round(m_acc*100,2)}%.')
        print(f'Baseline accuracy is: {round(b_acc*100,2)}%.')
        print()
        print(f'Model recall is: {round(m_rec*100,2)}%.')
        print(f'Baseline recall is: {round(b_rec*100,2)}%.')
        print()
        print(f'Model precision is: {round(m_pre*100,2)}%.')
        print(f'Baseline precision is: {round(bas_pre*100,2)}%.')
        print()
    #    print('Function returns: df, m_acc, b_acc, m_rec, b_rec, m_pre, bas_pre')
    # return df, m_acc, b_acc, m_rec, b_rec, m_pre, bas_pre
