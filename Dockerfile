FROM python:3.7

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt --no-cache-dir

WORKDIR /app

ADD main.py main.py

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]
