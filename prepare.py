from sklearn.model_selection import train_test_split


def train_val_test(df, strat, seed=100):
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    return train, val, test


def prep_iris(df):
    new_df = df.drop(columns=['species_id', 'measurement_id'])
    new_df = new_df.rename(columns={'species_name': 'species'})
    return new_df


def prep_titanic(df):
    titanic = df.drop(columns=['passenger_id', 'pclass', 'embarked'])
    return titanic


def prep_telco(df):
    telco = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    return telco
