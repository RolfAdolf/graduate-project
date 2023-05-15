FROM python:3.10

RUN mkdir /graduate-project

WORKDIR graduate-project

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

RUN chmod a+x *.sh

CMD [run.sh]