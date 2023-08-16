from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import acquire as ac


def train_val_test(df, strat, seed=100):
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    return train, val, test


def prep_iris(df):
    new_df = df.drop(columns=['species_id', 'measurement_id'])
    new_df = new_df.rename(columns={'species_name': 'species'})
    return new_df


def prep_titanic(df):
    df = df.drop(columns=['passenger_id', 'pclass', 'embarked', 'deck'])
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['class', 'embark_town'])
    return df


def titanic_dummies(df):
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    df = pd.get_dummies(df, columns=['class', 'embark_town'])
    return df


def titanic():
    df = ac.get_titanic_data()
    impute_df(df, 'age', strategy='median')
    df = prep_titanic(df)
    return df


def prep_telco(df):
    telco = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    telco['total_charges'] = telco['total_charges'].astype(float)
    telco = pd.get_dummies(telco, columns=['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
                                           'online_security', 'online_backup', 'device_protection', 'tech_support',
                                           'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn'],
                           drop_first=True)
    telco = pd.get_dummies(telco, columns=['contract_type', 'internet_service_type', 'payment_type'])
    return telco


def split_x_y(df, target=''):
    x_df = df.drop(columns=target)
    y_df = df[target]
    return x_df, y_df


def impute_df(df, column='', strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(df[[column]])
    df[[column]] = imputer.transform(df[[column]])
    return df


def baseline(df, data='', target_value='', show_results=True):
    df['baseline'] = df[data].mode()[0]  # Creates a baseline prediction from the mode of the data
    b_acc = (df[data] == df['baseline']).mean()
    sub_rec = df[df[data] == target_value]  # Subset of all positive cases for recall
    b_rec = (sub_rec[data] == sub_rec['baseline']).mean()
    sub_bas_pre = df[df['baseline'] == target_value]
    if sub_bas_pre.empty:
        bas_pre = 0.0
    else:
        bas_pre = (sub_bas_pre[data] == sub_bas_pre['baseline']).mean()
    if show_results:
        print(f'Baseline accuracy is: {round(b_acc * 100, 2)}%.')
        print(f'Baseline recall is: {round(b_rec * 100, 2)}%.')
        print(f'Baseline precision is: {round(bas_pre * 100, 2)}%.')
        print()
    df.drop(columns=['baseline'], inplace=True)


def evaluate_model(dataframe, data='data column', model='Dataframe', p_class='', show_results=True):
    df = dataframe
    df['model'] = model
    m_acc = (df[data] == df['model']).mean()
    sub_rec = df[df[data] == p_class]  # Subset of all positive cases for recall
    m_rec = (sub_rec[data] == sub_rec['model']).mean()
    s_pre = df[df['model'] == p_class]  # Subset of all positive guesses
    m_pre = (s_pre[data] == s_pre['model']).mean()
    if show_results:
        print(f'Model accuracy is: {round(m_acc * 100, 2)}%.')
        print(f'Model recall is: {round(m_rec * 100, 2)}%.')
        print(f'Model precision is: {round(m_pre * 100, 2)}%.')


''' dataframe is the dataframe with all of the information.
    data is the column name of where the data is under.
    model is the dataframe containing model information
    p_class is the variable that you are trying to predict'''


def importance(training, model):
    imp = {'cols': training.columns,
           'importance': model.feature_importances_}
    return pd.DataFrame(imp).sort_values(by='importance', ascending=False)


''' training is the training data (x_train)
    model is the actual model object'''
