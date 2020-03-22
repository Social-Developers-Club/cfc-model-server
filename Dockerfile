FROM pytorch/pytorch:latest

# copy requirements descriptor
COPY ./requirements.txt /requirements.txt

# install dependencies
RUN pip install --upgrade pip
RUn pip install --upgrade pipenv
RUN pip install --upgrade -r /requirements.txt

# copy remaining files
COPY . .

# download pretrained models
# TODO: fix
# RUN python scripts/download_models.py

# start server
ENTRYPOINT [ "python" ]
CMD [ "server.py" ]
