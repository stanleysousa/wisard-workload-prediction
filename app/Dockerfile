FROM stanleysousa/wisard
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
WORKDIR /app
COPY . .
CMD [ "python", "/app/src/script.py" ]