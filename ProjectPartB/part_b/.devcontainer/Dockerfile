FROM python:3.12-slim-bookworm

# Git for development and runtime libraries required by VS Code Server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libstdc++6 tar \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m venv "${VIRTUAL_ENV}"

# Rebuild this layer when the project's package requirements change.
COPY requirements.txt /tmp/requirements.txt
# Install CPU PyTorch first so Linux builds avoid CUDA dependencies.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m ipykernel install --sys-prefix --name cascade --display-name "Python (Cascade)" \
    && python -m pip check

RUN groupadd --gid 1000 dev \
    && useradd --uid 1000 --gid dev --create-home --shell /bin/bash dev \
    && chown -R dev:dev "${VIRTUAL_ENV}"

USER dev
WORKDIR /home/dev
