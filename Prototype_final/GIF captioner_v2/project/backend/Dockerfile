FROM python:3.11-slim

WORKDIR /app

# copy backend app files
COPY . /app

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose default huggingface port 7860
EXPOSE 7860

# run uvicorn with the app object 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
