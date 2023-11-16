# build arguments
ARG PYTHON=python3.10
ARG TZ=America/Los_Angeles
ARG SSLSUBJ=/C=US/CN=DOCA
ARG USER_NAME=lab
ARG VENV_PATH=/var/opt/venv
ARG PORT=8888

# ---------------------------------------------
# source: ubuntu LTS
FROM ubuntu:20.04 AS source
LABEL stage=builder
# ---------------------------------------------

ARG PYTHON
ARG USER_NAME
ARG VENV_PATH
ARG SSLSUBJ

# cancel user prompts
ARG DEBIAN_FRONTEND=noninteractive

# comment out if using current version
RUN apt-get update && apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:alex-p/tesseract-ocr-devel

# install python
RUN apt-get update && apt-get install --no-install-recommends -y \
    $PYTHON $PYTHON-dev $PYTHON-venv python3-pip python3-wheel build-essential \
    libssl-dev openssl libopencv-dev python3-opencv tesseract-ocr poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
RUN $PYTHON -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
# generate certificate for jupyter
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj $SSLSUBJ \
                -keyout /usr/local/share/ca-certificates/jupyter.key \
                -out /usr/local/share/ca-certificates/jupyter.pem \
                -batch

# install requirements
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir wheel
# install notebooks
RUN python -m pip install --no-cache-dir jupyter
# install torch with GPU (consult https://pytorch.org/get-started/locally/ on proper configuration)
RUN python -m pip install --no-cache-dir "torch>=2.0" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --upgrade
#RUN python -m pip install --no-cache-dir "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --upgrade
RUN python -m pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --upgrade
# install OCR stuck
RUN python -m pip install --no-cache-dir opencv-contrib-python pytesseract
# install other python packages
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------
# target: multi-stage for size and performance
FROM ubuntu:20.04 AS target
# ---------------------------------------------

ARG PYTHON
ARG USER_NAME
ARG VENV_PATH
ARG TZ
ARG PORT
ARG DEBIAN_FRONTEND

# same as above
RUN apt-get update && apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:alex-p/tesseract-ocr-devel

# same as above except we do not need -dev here
RUN apt-get update && apt-get install --no-install-recommends -y $PYTHON python3-venv \
    git curl python3-opencv tesseract-ocr poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# set local timezoone
ENV TZ=$TZ
# set GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# create non-root user and set environment
RUN useradd --create-home -s /bin/bash --no-user-group -u 1000 $USER_NAME
COPY --chown=1000 --from=source $VENV_PATH $VENV_PATH
# if use certificates
COPY --chown=1000 --from=source /usr/local/share/ca-certificates /usr/local/share/ca-certificates

USER $USER_NAME
WORKDIR /home/$USER_NAME/notebooks

# messages always reach console
ENV PYTHONUNBUFFERED=1
# no __pycache__
ENV PYTHONDONTWRITEBYTECODE=1

# activate virtual environment
ENV VIRTUAL_ENV="$VENV_PATH" 
ENV PATH="$VENV_PATH/bin:$PATH"

# run jupyter server mounted into notebooks folder
ENTRYPOINT jupyter notebook --no-browser --ip=0.0.0.0 --port=${PORT} \
                            --NotebookApp.max_buffer_size=${MAX_BUFFER_SIZE} \
                            --NotebookApp.iopub_data_rate_limit=${IOPUB_DATA_RATE_LIMIT} \
                            --NotebookApp.iopub_msg_rate_limit=${IOPUB_MSG_RATE_LIMIT} \
                            --NotebookApp.password="${JUPYTER_PASS}" \
                            --NotebookApp.certfile="/usr/local/share/ca-certificates/jupyter.pem" \
                            --NotebookApp.keyfile="/usr/local/share/ca-certificates/jupyter.key"

