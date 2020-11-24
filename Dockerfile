# use a node base image
FROM python:3

WORKDIR /Multyiclass_model/src/classification

# COPY Multiclass_model-master/requirements.txt .

# RUN pip install -r requirements.txt

CMD ["python", "install_module.py"]
CMD ["python", "run_pipeline.py"]
