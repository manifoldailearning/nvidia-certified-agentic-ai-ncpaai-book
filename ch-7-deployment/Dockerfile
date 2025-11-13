FROM nvcr.io/nvidia/pytorch:23.10-py3
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]
