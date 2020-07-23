#FROM tensorflow/tensorflow:latest-py3
FROM tensorflow/tensorflow:latest-gpu

RUN pip install bert-for-tf2 flask 

RUN mkdir /workspace

COPY ./entrypoint.sh /workspace/entrypoint.sh
COPY ./app /workspace/app

RUN chmod u+x /workspace/entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/workspace/entrypoint.sh"]
