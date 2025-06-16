#!/usr/bin/env bash

set -euo pipefail

cd $(git rev-parse --show-toplevel)

OLD_EXTENSION_DIRECTORY=torch_extensions/causal_multihead_self_attention

for i in $(seq 11); do
    EXTENSION_DIRECTORY=torch_extensions/causal_multihead_self_attention_version_${i}
    mkdir -p ${EXTENSION_DIRECTORY}
    for FILENAME in causal_multihead_self_attention.cu setup.py __init__.py; do
        git cat-file -p version-${i}:${OLD_EXTENSION_DIRECTORY}/${FILENAME} > ${EXTENSION_DIRECTORY}/${FILENAME}
    done

    sed -i "s/namespace causal_multihead_self_attention/namespace causal_multihead_self_attention_version_${i}/" ${EXTENSION_DIRECTORY}/causal_multihead_self_attention.cu
    sed -i "s/PYBIND11_MODULE(causal_multihead_self_attention/PYBIND11_MODULE(causal_multihead_self_attention_version_${i}/" ${EXTENSION_DIRECTORY}/causal_multihead_self_attention.cu
    sed -i "s/TORCH_LIBRARY(causal_multihead_self_attention/TORCH_LIBRARY(causal_multihead_self_attention_version_${i}/" ${EXTENSION_DIRECTORY}/causal_multihead_self_attention.cu
    sed -i "s/TORCH_LIBRARY_IMPL(causal_multihead_self_attention/TORCH_LIBRARY_IMPL(causal_multihead_self_attention_version_${i}/" ${EXTENSION_DIRECTORY}/causal_multihead_self_attention.cu
    sed -i "s/\"causal_multihead_self_attention\"/\"causal_multihead_self_attention_version_${i}\"/" ${EXTENSION_DIRECTORY}/setup.py
done
