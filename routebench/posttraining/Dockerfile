FROM vllm/vllm-openai@sha256:607442e407b0fea97f8a132a78b787c121a996dd4de181fa08e8da06e71ec2db
# Reuse the portfolio's CUDA 12.8 / torch 2.8.0 / transformers 4.56.1 image.
RUN python3 -m pip install --no-cache-dir peft==0.17.1
WORKDIR /experiment
ENTRYPOINT ["python3", "experiment.py"]
