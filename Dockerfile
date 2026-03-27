# Unsloth Studio — DGX Spark Optimized
# Base chain: nvcr.io/nvidia/pytorch (NGC) → unsloth/unsloth:dgxspark-latest → this image
# Adds: Unsloth Studio web UI, latest transformers, FLA, causal-conv1d
#
# Built automatically via GitHub Actions on a weekly schedule.
# Pushed to ghcr.io/<owner>/unsloth-dgxspark

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

# Run Unsloth Studio first-time setup (installs backend deps + llama.cpp)
# This runs as root during build; the entrypoint switches to unsloth user
RUN /home/unsloth/.local/bin/unsloth studio update || true

# Add Studio to supervisord so it auto-starts alongside Jupyter and SSH
COPY supervisord-studio.conf /etc/supervisor/conf.d/studio.conf

# Expose ports: Jupyter (8888), Studio (8000), SSH (22)
EXPOSE 8888 8000 22

# Verify installs
RUN python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" && \
    python3 -c "import fla; print('flash-linear-attention: OK')" && \
    python3 -c "import causal_conv1d; print('causal-conv1d: OK')" && \
    python3 -c "import unsloth_cli; print('unsloth-cli: OK')"
