# use a node base image
FROM python:3

WORKDIR /classification

COPY /pythonproject/Multiclass_model-master/src/classification/install_module.py .
COPY /pythonproject/Multiclass_model-master/src/classification/run_pipeline.py .

# RUN pip install -r requirements.txt

CMD ["python", "install_module.py"]
CMD ["python", "run_pipeline.py"]
