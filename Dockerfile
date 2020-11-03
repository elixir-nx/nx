FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG bazel_bin_url="https://github.com/bazelbuild/bazel/releases/download/3.4.1/bazel-3.4.1-linux-x86_64"
ARG pip_args=""
WORKDIR /etc/apt/sources.list.d
RUN rm cuda.list nvidia-ml.list
WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends wget curl git locales python3 python3-pip
RUN echo ${bazel_bin_url}
RUN curl -L ${bazel_bin_url} -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel \
    && bazel

RUN wget https://packages.erlang-solutions.com/erlang-solutions_2.0_all.deb && dpkg -i erlang-solutions_2.0_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends esl-erlang && \
    apt-get install -y --no-install-recommends elixir && \
    rm erlang-solutions_2.0_all.deb

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN echo $LANG UTF-8 > /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=$LANG

RUN wget https://repo.hex.pm/installs/1.10.0/hex-0.20.6.ez && \
    mix archive.install ./hex-0.20.6.ez --force && \
    rm ./hex-0.20.6.ez

RUN python3 -m install ${pip_args} numpy
