from env import get_connection
import pandas as pd
import os


def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.isfile(filename):  # checks if file exists
        return pd.read_csv(filename)  # if file exists, function will read and return csv file
    else:
        db_url = get_connection('titanic_db')
        query = '''
                SELECT * FROM passengers
                '''
        titanic = pd.read_sql(query, db_url)
        titanic.to_csv('titanic.csv', index=False)
        return titanic


def get_iris_data():
    filename = 'iris.csv'
    if os.path.isfile(filename):  # checks if file exists
        return pd.read_csv(filename)  # if file exists, function will read and return csv file
    else:
        db_url = get_connection('iris_db')
        query = '''SELECT * FROM species as S JOIN measurements as M USING(species_id)'''
        iris = pd.read_sql(query, db_url)
        iris.to_csv(filename, index=False)
        return iris


def get_telco_data():
    filename = 'telco.csv'
    if os.path.isfile(filename):  # checks if file exists
        return pd.read_csv(filename)  # if file exists, function will read and return csv file
    else:
        db_url = get_connection('telco_churn')
        query = '''SELECT * FROM customers AS cd
        LEFT JOIN contract_types as ct USING(contract_type_id)
        LEFT JOIN internet_service_types as ist USING(internet_service_type_id)
        LEFT JOIN payment_types as pt USING(payment_type_id)
        WHERE total_charges != '' '''
        telco = pd.read_sql(query, db_url)
        telco.to_csv(filename, index=False)
        return telco


def sql_query(db='None', query='None'):
    if db == 'None':
        print('Database not specified.')
    elif query == 'None':
        print('No query!')
    else:
        db_url = get_connection(db)
        df = pd.read_sql(query, db_url)
        return df


def show_tables(db='None'):
    if db == 'None':
        print('Database not specified')
    else:
        db_url = get_connection(db)
        query = 'SHOW TABLES'
        tables = pd.read_sql(query, db_url)
        return tables
