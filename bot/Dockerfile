FROM python:latest
WORKDIR /usr/src/app

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./bot.py ./
CMD [ "python", "./bot.py" ]