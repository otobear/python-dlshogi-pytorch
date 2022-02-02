FROM python:3.9

WORKDIR /home/workspace/python-dlshogi-pytorch
COPY . .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir -e .
RUN apt-get update \
    && apt-get install -y p7zip-full
