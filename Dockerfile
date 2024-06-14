FROM 172.32.150.15/public/vllm-openai:v0.4.3
RUN pip install fastchat && mkdir -p /vllm_fastchat
ADD . /vllm_fastchat
WORKDIR /vllm_fastchat
ENTRYPOINT [ "python3","api_server.py" ]