from airflow import DAG

# We need to import the operators used in our tasks
from airflow.operators.python import PythonOperator
# We then import the days_ago function
from airflow.utils.dates import days_ago
from datetime import timedelta

import sklearn
import pandas as pd
import numpy as np

# def my_python_script():
#     # Your Python script goes here
#     f = open("/opt/airflow/files/demofile2.txt", "a")
#     f.write("Now the file has more content!")
#     f.close()
# my_first_dag1 = DAG(
#     'first_dag1',
#     default_args=default_args,
#     description='Our DAG 1',
#     schedule_interval=timedelta(days=1),
# )

# task_1 = PythonOperator(
#     task_id='first_task',
#     python_callable=my_python_script,
#     dag=my_first_dag1,
# )
# def get_data01():
#     load_train(data_set='/opt/airflow/files/df_train.csv', dump_model='/opt/airflow/files/filename.joblib')

def gener_colomns():
    np.random.seed(0)
    return np.random.randint(0, 100, size=(10, 3))
def get_data():  #https://scikit-learn.org/stable/model_persistence.html
    # Your Python script goes here
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    df = pd.DataFrame(gener_colomns(),gener_colomns(),gener_colomns(),gener_colomns(),gener_colomns(), columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm))','target'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    df_train = pd.DataFrame(X_train, columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm))'])
    df_train['target'] = y_train
    df_test = pd.DataFrame(X_test, columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm))'])
    df_test['target'] = y_test
    df_train.to_csv('/opt/airflow/files/df_train.csv', index=False)
    df_test.to_csv('/opt/airflow/files/df_test.csv', index=False)
    
    
def train01():
    load_train(data_set='/opt/airflow/files/df_train.csv', dump_model='/opt/airflow/files/filename.joblib')
    
def train02():
    load_train(data_set='/opt/airflow/files/df_train.csv', dump_model='/opt/airflow/files/filename.joblib')

def load_train(data_set='/opt/airflow/files/df_train.csv', dump_model='/opt/airflow/files/filename.joblib'):  #https://scikit-learn.org/stable/model_persistence.html
    # Your Python script goes here
    from sklearn import svm
    from joblib import dump, load
    clf = svm.SVC()
    df_train=pd.read_csv(data_set) #skipcols=usecols=[1,:]
    X_train=df_train.iloc[:,0:3].to_numpy()
    y_train=df_train.iloc[:,4].to_numpy()
    clf.fit(X_train, y_train)
    dump(clf, dump_model) 
    # iris = datasets.load_iris()
    # df = pd.DataFrame.from_dict(iris, orient='index')
    # df.to_csv('/opt/airflow/files/iris.csv')
    
    # from sklearn.externals import joblib #https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
    # joblib.dump(clf, '/opt/airflow/files/filename.pkl')  #написать куда выгружать
    
    # from joblib import dump, load   #https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
    # pickle_file = '/opt/airflow/files/pickle_data.joblib'
    # with open(pickle_file, 'wb') as f:
    #     dump(clf, f)
    
def load_test():
    # from joblib import dump, load
    # pickle_file = '/opt/airflow/files/pickle_data.joblib'
    # with open(pickle_file, 'rb') as f:
    #     load(f)
    import pandas as pd
    import numpy as np
    from sklearn import svm
    from joblib import dump, load
    clf2 = load('/opt/airflow/files/filename.joblib')
    df_test=pd.read_csv('/opt/airflow/files//df_test.csv') 
    X_test=df_test.iloc[:,0:3].to_numpy()
    y_test=df_test.iloc[:,4].to_numpy()
    predict = clf2.predict(X_test)
    df = pd.DataFrame(predict)
    df.to_csv('/opt/airflow/files/predict.csv', index=False)
    # f = open("/opt/airflow/files/predict.csv", "a")
    # f.write(predict)
    # f.close()
    
    

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'email': ['airflow@my_first_dag.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


joblib_skilearn_par = DAG(
    'joblib_skilearn_par',
    default_args=default_args,
    description='Our DAG 1',
    schedule_interval=timedelta(days=1),
)

get_data = PythonOperator(
    task_id='get_data',
    python_callable=get_data, #написать функцию
    dag=joblib_skilearn_par,
)

fit_model01 = PythonOperator(
    task_id='fit_model01',
    python_callable=train01, #написать функцию
    dag=joblib_skilearn_par,
)
fit_model02 = PythonOperator(
    task_id='fit_model02',
    python_callable=train02, #написать функцию
    dag=joblib_skilearn_par,
)

test_model = PythonOperator(
    task_id='ftest_model',
    python_callable=load_test, #написать функцию
    dag=joblib_skilearn_par,
)

get_data >> fit_model01 >> test_model
get_data >> fit_model02 >> test_model