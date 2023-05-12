FROM python:3.10

RUN mkdir /graduate-project

WORKDIR graduate-project

COPY requirements.txt .

COPY tensorflow_packages .

RUN pip install --upgrade pip

RUN pip install tensorflow_packages/*

RUN pip install -r requirements.txt

COPY . .

RUN chmod a+x docker/*.sh

CMD [train.sh]