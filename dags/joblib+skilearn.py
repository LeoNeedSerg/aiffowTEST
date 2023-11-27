from airflow import DAG

# We need to import the operators used in our tasks
from airflow.operators.python import PythonOperator
# We then import the days_ago function
from airflow.utils.dates import days_ago
from datetime import timedelta

import sklearn
import pandas as pd

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

def get_data_train():  #https://scikit-learn.org/stable/model_persistence.html
    # Your Python script goes here
    from sklearn import svm
    from sklearn import datasets
    from joblib import dump, load
    clf = svm.SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    dump(clf, '/opt/airflow/files/filename.joblib') 
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
    
    from sklearn import svm
    from sklearn import datasets
    from joblib import dump, load
    clf2 = load('/opt/airflow/files/filename.joblib')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    predict = clf2.predict(X[0:1])
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
    'retry_delay': timedelta(minutes=5),
}


joblib_skilearn = DAG(
    'joblib_skilearn',
    default_args=default_args,
    description='Our DAG 1',
    schedule_interval=timedelta(days=1),
)

fit_model = PythonOperator(
    task_id='fit_model',
    python_callable=get_data_train, #написать функцию
    dag=joblib_skilearn,
)

test_model = PythonOperator(
    task_id='ftest_model',
    python_callable=load_test, #написать функцию
    dag=joblib_skilearn,
)

fit_model >> test_model