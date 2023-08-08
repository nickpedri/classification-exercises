from sklearn.model_selection import train_test_split
import pandas as pd


def train_val_test(df, strat, seed=100):
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    return train, val, test
