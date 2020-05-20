# Base image
FROM nvidia/cudagl:10.1-runtime-ubuntu18.04

# Disable interactive installation for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
    apt-utils ssh-client nano libcudnn7 \
    python3-dev python3-pip \
    build-essential curl unzip psmisc rsync && \
    apt-get autoremove -y --purge && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade setuptools wheel

# Configure EAI toolkit user and ssh port-forwarding access
EXPOSE 2222
EXPOSE 5900
EXPOSE 6000
EXPOSE 8088
ENV LANG=en_US.UTF-8
RUN apt-get update && \
    apt-get install -y \
        ca-certificates supervisor openssh-server bash ssh \
        curl wget vim procps htop locales nano man && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen && \
    useradd -m -u 13011 -s /bin/bash toolkit && \
    passwd -d toolkit && \
    useradd -m -u 13011 -s /bin/bash --non-unique console && \
    passwd -d console && \
    useradd -m -u 13011 -s /bin/bash --non-unique _toolchain && \
    passwd -d _toolchain && \
    useradd -m -u 13011 -s /bin/bash --non-unique coder && \
    passwd -d coder && \
    chown -R toolkit:toolkit /run /etc/shadow && \
    apt-get autoremove -y --purge && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo ssh >> /etc/securetty && \
    rm -f /etc/legal /etc/motd
COPY --chown=13011:13011 --from=registry.console.elementai.com/shared.image/sshd:base /tk /tk
RUN chmod a+rwx /tk/bin/start.sh

# Install pypi packages:
# - common
# - tensorflow 2.1+
# - ray[rllib] 0.8.5+
RUN pip3 install --upgrade pip && \
    pip3 install \
    psutil gputil setproctitle tensorflow-gpu>=2.1.0 \
    gym tensorflow-probability requests>=2.23.0 msgpack==0.6.2 ray[rllib]>=0.8.5

# Fix missing packages
RUN pip3 install packaging

COPY src/ /

# For now, lock image, to make it easy to docker exec / eai job exec
CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]
