# Sourced by every Hazel job: everything lives in scratch (home has a 15 GB quota).
export S=/share/nesylearning/$USER
export R=$S/CDCL
export UV_PYTHON_INSTALL_DIR=$S/uv/python UV_CACHE_DIR=$S/uv/cache PIP_CACHE_DIR=$S/pip_cache
export HF_HUB_CACHE=$S/hf_cache            # the HF token stays in ~/.cache/huggingface
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PY=$R/.venv/bin/python
mkdir -p $S/logs
