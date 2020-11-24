# use a node base image
FROM python:3

WORKDIR /Mulyiclass_model/src/classification

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
	rm requirements.txt
CMD ["python", "./run_pipeline.py"]
