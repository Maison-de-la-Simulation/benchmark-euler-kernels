# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b

ARG LLVM_VERSION=22

ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get --yes update \
 && apt-get --yes --no-install-recommends install \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && wget --output-document=/usr/share/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/apt.llvm.org.asc] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list \
 && apt-get --yes update \
 && apt-get --yes --no-install-recommends install \
    build-essential \
    clang-tidy-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    libclang-${LLVM_VERSION}-dev \
    clang-${LLVM_VERSION} \
    cmake \
    git \
    openssh-client \
 && git clone --branch clang_${LLVM_VERSION} --depth 1 https://github.com/include-what-you-use/include-what-you-use.git \
 && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VERSION} -B build-iwyu -S include-what-you-use \
 && cmake --build build-iwyu \
 && cmake --install build-iwyu \
 && rm --force --recursive build-iwyu include-what-you-use \
 && git clone --branch 5.1.0 --depth 1 https://github.com/kokkos/kokkos.git \
 && export Kokkos_ROOT=/opt/kokkos \
 && cmake -D CMAKE_BUILD_TYPE=Release -D Kokkos_ENABLE_DEPRECATED_CODE_5=OFF -D Kokkos_ENABLE_SERIAL=ON -B build -S kokkos \
 && cmake --build build \
 && cmake --install build --prefix $Kokkos_ROOT \
 && rm --force --recursive build \
 && git clone --branch v1.17.0 https://github.com/google/googletest.git \
 && export GTest_ROOT=/opt/googletest \
 && cmake -D CMAKE_BUILD_TYPE=Release -B build -S googletest \
 && cmake --build build \
 && cmake --install build --prefix $GTest_ROOT \
 && rm --force --recursive build \
 && git clone --branch v1.9.5 https://github.com/google/benchmark.git \
 && export benchmark_ROOT=/opt/benchmark \
 && cmake -D BENCHMARK_ENABLE_TESTING=OFF -D CMAKE_BUILD_TYPE=Release -B build -S benchmark \
 && cmake --build build \
 && cmake --install build --prefix $benchmark_ROOT \
 && rm --force --recursive build \
 && apt-get --yes purge \
    apt-transport-https \
    apt-utils \
    wget \
 && apt-get --yes autoremove \
 && apt-get --yes clean \
 && apt-get --yes autoclean \
 && rm --force --recursive /var/lib/apt/lists/* \
 && useradd --home-dir /ci --create-home --user-group ci \
 && mkdir /data \
 && chown ci:ci /ci /data

USER ci:ci
WORKDIR /ci
ENV HOME=/ci
ENV CMAKE_PREFIX_PATH="/opt/benchmark:/opt/googletest:/opt/kokkos"

CMD ["/bin/bash", "-li"]
