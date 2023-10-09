from airflow import DAG

# We need to import the operators used in our tasks
from airflow.operators.python import PythonOperator
# We then import the days_ago function
from airflow.utils.dates import days_ago
from datetime import timedelta


def my_python_script():
    # Your Python script goes here
    f = open("/opt/airflow/files/demofile2.txt", "a")
    f.write("Now the file has more content!")
    f.close()


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(5),
    'email': ['airflow@my_first_dag.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


my_first_dag1 = DAG(
    'first_dag1',
    default_args=default_args,
    description='Our DAG 1',
    schedule_interval=timedelta(days=1),
)

task_1 = PythonOperator(
    task_id='first_task',
    python_callable=my_python_script,
    dag=my_first_dag1,
)