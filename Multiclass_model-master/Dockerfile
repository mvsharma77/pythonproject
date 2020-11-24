# use a node base image
FROM node:7-onbuild

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
	rm requirements.txt

# set maintainer
LABEL maintainer "academy@release.works"

# set a health check
HEALTHCHECK --interval=5s \
            --timeout=5s \
            CMD curl -f http://127.0.0.1:8000 || exit 1
            
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/src/classification/run_pipeline.py"]

# tell docker what port to expose
EXPOSE 8000
