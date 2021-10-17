FROM python:3.9


COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade wheel
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /tako
WORKDIR /tako
COPY . .

CMD [ "python3", "./tako.py", "-g", "-k", "CLASSY_CAMEL" ]
