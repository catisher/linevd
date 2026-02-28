# 基础镜像：Ubuntu 22.04（GLIBC 2.35，兼容所有工具）
FROM ubuntu:22.04

# 1. 初始化所有变量，消除UndefinedVar警告（核心修复）
ENV LD_LIBRARY_PATH=""
ENV DEBIAN_FRONTEND=noninteractive
ENV SINGULARITY=true
ENV PATH=$PATH:/GloVe/build
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
#待定
ENV PATH=$PATH:/usr/local/cuda/bin

# 复制必要文件（无则创建空文件）
COPY cli.sh /cli.sh
COPY requirements.txt /requirements.txt

# 2. 配置国内源，加速下载（解决超时问题）
# 有待测试
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    # pip国内源
    mkdir -p /root/.pip && \
    echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn" > /root/.pip/pip.conf && \
    # conda国内源
    echo "channels:\n  - defaults\ndefault_channels:\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\ncustom_channels:\n  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" > /root/.condarc

# 3. 版本信息验证
RUN echo "=== 完整版本信息验证 ===" && \
    cat /etc/os-release | grep -E "NAME|VERSION_ID" | tee -a /version_info.txt && \
    ldd --version | head -1 | tee -a /version_info.txt && \
    echo "=== 完整版本信息已写入/version_info.txt ==="

# 4. 安装基础工具
# 执行权限设置 + 系统更新 + 安装依赖
RUN chmod u+x /cli.sh \
    && apt update -y \
    && apt install -y \
        wget \
        build-essential \
        git \
        graphviz \
        zip \
        unzip \
        curl \
        vim \
        libexpat1-dev \
        cmake \
    # 清理apt缓存，减小镜像体积
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# 5. 安装Miniconda（完整路径，无警告）
RUN cd / && \
    wget --timeout=60 --tries=3 --no-check-certificate https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b && rm -f miniconda.sh && \
    /root/miniconda3/bin/conda clean -afy && \
    echo 'export PATH="/root/miniconda3/bin:$PATH"' >> /root/.bashrc

# 6. 全局PATH包含conda
ENV PATH=/root/miniconda3/bin:$PATH

# 7. 安装PyTorch（cu118，国内源加速）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 8. 安装GloVe（无Git，压缩包下载）
RUN cd / && \
    wget --timeout=60 --tries=3 --no-check-certificate https://github.com/stanfordnlp/GloVe/archive/refs/heads/master.zip -O GloVe-master.zip && \
    unzip -q GloVe-master.zip && mv GloVe-master GloVe && rm -f GloVe-master.zip && \
    cd GloVe && make -j$(nproc) && cd / && rm -rf GloVe/.git
    
# 9. 安装cppcheck 2.10（兼容GCC 11+）
RUN cd / && \
    curl -L -k https://github.com/danmar/cppcheck/archive/refs/tags/2.10.tar.gz -o cppcheck2.10.tar.gz && \
    mkdir -p cppcheck && tar -xzvf cppcheck2.10.tar.gz -C cppcheck --strip-components=1 && \
    cd cppcheck && mkdir -p build && cd build && \
    cmake .. && cmake --build . -j$(nproc) && make install && \
    cd / && rm -rf cppcheck cppcheck2.10.tar.gz
    
# 11. 安装RATS（SSL绕过，清理压缩包）
RUN cd / && \
    curl -L -k https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rough-auditing-tool-for-security/rats-2.4.tgz -o rats-2.4.tgz && \
    tar -xzvf rats-2.4.tgz && cd rats-2.4 && \
    ./configure && make -j$(nproc) && make install && \
    cd / && rm -rf rats-2.4 rats-2.4.tgz

# 12. 安装flawfinder（国内源）
RUN pip install flawfinder -i https://pypi.tuna.tsinghua.edu.cn/simple

# 14. 安装pygraphviz和nltk
RUN /root/miniconda3/bin/conda install -y pygraphviz && \
    pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -c 'import nltk; nltk.download("punkt")'
  
# 13. 安装Python依赖（兼容cu118） 前置miniconda
RUN cat /requirements.txt | xargs -n 1 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install dgl -i https://pypi.tuna.tsinghua.edu.cn/simple

 
# 10. 安装Joern（已安装Java 8，无需重复安装）
RUN cd / && \
    wget --timeout=60 --tries=3 --no-check-certificate https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh -O joern-install.sh && \
    chmod +x ./joern-install.sh && \
    printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n' | sudo ./joern-install.sh --interactive && \
    rm -f ./joern-install.sh






# 15. 收尾配置（清理+权限）
RUN chmod +x /cli.sh && \
    apt autoremove -y && apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 恢复交互模式
ENV DEBIAN_FRONTEND=dialog

# 默认工作目录+命令
WORKDIR /
CMD ["/bin/bash"]

