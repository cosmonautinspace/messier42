FROM busybox:latest

RUN mkdir -p /tmp/knowledgeBase/ann_model /tmp/knowledgeBase/ols_model

COPY currentSolution.xml /tmp/knowledgeBase/ann_model/currentSolution.xml
COPY currentOlsSolution_RB.pkl /tmp/knowledgeBase/ols_model/currentOlsSolution_RB.pkl
COPY currentOlsSolution_G.pkl /tmp/knowledgeBase/ols_model/currentOlsSolution_G.pkl
COPY README.md /tmp/knowledgeBase/README.md
