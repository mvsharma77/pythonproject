# use a node base image
FROM python:3

WORKDIR /classification

COPY /Multiclass_model-master/src/classification/install_module.py .
COPY /Multiclass_model-master/src/classification/run_pipeline.py .
COPY /Multiclass_model-master/requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "install_module.py"]
CMD ["python", "run_pipeline.py"]
