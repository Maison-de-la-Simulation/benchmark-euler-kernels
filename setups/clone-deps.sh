#!/bin/bash

git clone --branch v1.9.4 --depth 1 https://github.com/google/benchmark.git
git clone https://github.com/kokkos/kokkos.git
cd kokkos || exit
git checkout 7f8988b4d
cd .. || exit
git clone --branch v1.17.0 --depth 1 https://github.com/google/googletest.git

echo "Repos cloned"
