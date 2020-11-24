# use a node base image
FROM python:3

WORKDIR /Mulyiclass_model/src/classification

COPY Multiclass_model-master/requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./run_pipeline.py"]
