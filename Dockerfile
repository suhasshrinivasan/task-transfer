### base image
FROM ghcr.io/walkerlab/docker-pytorch-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04

### update package managers
RUN apt-get update 
RUN pip3 install --upgrade pip

# graphviz for visualizing pymc models and datajoint schemas
RUN apt install -y graphviz==2.43.0

### python packages
# wandb for visualization, pymc for sampling
# datajoint for experiment and data management
RUN pip3 install wandb==0.17.5 pymc==5.6.1 datajoint==0.14.1 pytest==8.2.2 pyparsing==3.1.2

# stable diffusion
# RUN pip3 install --upgrade diffusers transformers scipy accelerate wandb

### custom packages
# gensn for probabilistic machine learning
RUN git clone https://github.com/suhasshrinivasan/gensn.git /src/gensn &&\
    pip3 install --user -e /src/gensn

# insilico-stimuli for generating image stimuli such as oriented gratings
# use Suhas' fork, since the original repo introduces several undocumented breaking changes
RUN git clone https://github.com/suhasshrinivasan/insilico-stimuli.git /src/insilico-stimuli &&\
    cd /src/insilico-stimuli &&\
    pip3 install /src/insilico-stimuli


### build project as package
COPY . /src/project
RUN pip3 install --user -e /src/project