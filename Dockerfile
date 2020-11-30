FROM python:3
MAINTAINER Yaroslav Honchar "yaroslav.honchar@nure.ua"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]

