FROM python:3
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
EXPOSE 8000
CMD python .\House-price-prediction-using-flask\app.py

