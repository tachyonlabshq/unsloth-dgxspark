# Unsloth Studio — DGX Spark Optimized
# Base chain: nvcr.io/nvidia/pytorch (NGC) → unsloth/unsloth:dgxspark-latest → this image
# Adds: Unsloth Studio web UI, latest transformers, FLA, causal-conv1d
#
# Built automatically via GitHub Actions on a weekly schedule.
# Pushed to ghcr.io/tachyonlabshq/unsloth-dgxspark

FROM unsloth/unsloth:dgxspark-latest

LABEL org.opencontainers.image.source="https://github.com/tachyonlabshq/unsloth-dgxspark"
LABEL org.opencontainers.image.description="Unsloth Studio for DGX Spark — NGC-based with FLA, Studio UI, and latest model support"
LABEL com.centurylinklabs.watchtower.enable="true"

# Upgrade core ML packages for latest model support
RUN pip install --break-system-packages --no-cache-dir --upgrade \
    transformers \
    unsloth \
    unsloth_zoo \
    trl \
    peft \
    accelerate \
    datasets \
    bitsandbytes

# Install flash-linear-attention + causal-conv1d for MoE fast path (~12x speedup)
RUN pip install --break-system-packages --no-cache-dir \
    flash-linear-attention \
    causal-conv1d

# Install Unsloth Studio (creates venv, builds llama.cpp for SM 12.1)
# --no-torch skips PyTorch install since it's already in the base image
RUN curl -fsSL https://unsloth.ai/install.sh | sh -s -- --no-torch

# Create Studio log directory
RUN mkdir -p /var/log/studio && chown unsloth:runtimeusers /var/log/studio

# Add Studio to supervisord (auto-starts on port 8000)
COPY supervisord-studio.conf /etc/supervisor/conf.d/studio.conf

# Expose ports: Studio (8000), Jupyter (8888), SSH (22)
EXPOSE 8000 8888 22

# Verify installs
RUN python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" && \
    python3 -c "import fla; print('flash-linear-attention: OK')" && \
    python3 -c "import causal_conv1d; print('causal-conv1d: OK')" && \
    python3 -c "import unsloth_cli; print('unsloth-cli: OK')" && \
    test -d /home/unsloth/.unsloth/studio && echo "Studio: OK"
