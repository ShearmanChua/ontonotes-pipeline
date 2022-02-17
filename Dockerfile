FROM nvcr.io/nvidia/pytorch:19.12-py3

RUN apt-get update -y
RUN apt-get upgrade -y
RUN pip install --upgrade pip

# # Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda

# ENV PATH=$CONDA_DIR/bin:$PATH

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt --ignore-installed

ADD data/ontonotes-release-5.0_LDC2013T19.tgz .
ADD data/conll-formatted-ontonotes-5.0-12.tar.gz .
RUN ls
RUN conda create --name py27 python=2.7.13
RUN source activate py27