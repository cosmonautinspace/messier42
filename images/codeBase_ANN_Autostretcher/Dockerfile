FROM busybox:latest

RUN mkdir -p /tmp/codeBase/ann_model /tmp/codeBase/ols_model

COPY hybridModel.py /tmp/codeBase/ann_model/hybridModel.py
COPY OLS_model.py /tmp/codeBase/ols_model/OLS_model.py
COPY UE_04_LinearRegDiagnostic.py /tmp/codeBase/ols_model/UE_04_LinearRegDiagnostic.py
COPY README.md /tmp/codeBase/README.md
