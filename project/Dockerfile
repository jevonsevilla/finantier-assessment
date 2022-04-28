# Get the Fast API image with Python version 3.8
FROM python:3.8

# Create the directory for the container
WORKDIR /app

# Copy required files
COPY requirements.txt requirements.txt
COPY app.py app.py

# Install the dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Preprocessing Package
COPY /preprocessing ./preprocessing/

# Copy the serialized model and the vectors
COPY /model/xgboost_model.pickle.dat ./model/xgboost_model.pickle.dat
COPY /preprocessing/encoder.pickle ./preprocessing/encoder.pickle

# Run by specifying the host and port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]