FROM ubuntu:latest

RUN mkdir -p /tmp/codeBase/ann_model /tmp/codeBase/ols_model
WORKDIR /tmp/codeBase
RUN apt-get update  && apt-get install python3 -y && apt-get install python3-pandas -y && apt-get install git -y && git clone https://github.com/MarcusGrum/pybrain.git
RUN apt-get install python3-statsmodels python3-numpy python3-pil python3-matplotlib python3-seaborn python3-sklearn -y
COPY hybridModel.py /tmp/codeBase/ann_model/hybridModel.py
COPY OLS_model.py /tmp/codeBase/ols_model/OLS_model.py
COPY UE_04_LinearRegDiagnostic.py /tmp/codeBase/ols_model/UE_04_LinearRegDiagnostic.py
COPY README.md /tmp/codeBase/README.md
COPY activateOLS.py /tmp/codeBase/activateOLS.py
COPY activateANN.py /tmp/codeBase/activateANN.py