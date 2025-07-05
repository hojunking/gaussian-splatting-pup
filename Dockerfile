# nvidia/cuda 베이스 이미지
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 필수 패키지
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python -m pip install --upgrade pip

# pytorch + torchvision
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 기타 패키지
RUN pip install tqdm plyfile

# submodules 복사
COPY submodules/ /workspace/submodules/

# 각 submodule 빌드
RUN cd /workspace/submodules/compress-diff-gaussian-rasterization && \
    python setup.py install || echo "setup.py missing - please build manually"

RUN cd /workspace/submodules/diff-gaussian-rasterization && \
    python setup.py install || echo "setup.py missing - please build manually"

RUN cd /workspace/submodules/simple-knn && \
    python setup.py install || echo "setup.py missing - please build manually"

RUN cd /workspace/submodules/rasterization_and_pup_fisher && \
    python setup.py install || echo "setup.py missing - please build manually"

# 작업 디렉토리
WORKDIR /workspace

# bash로 진입
CMD ["/bin/bash"]
