#!/bin/bash

git clone --branch v1.9.4 --depth 1 https://github.com/google/benchmark.git
git clone --branch fix-simd-from-4.7.1 --depth 1 https://github.com/tpadioleau/kokkos.git
