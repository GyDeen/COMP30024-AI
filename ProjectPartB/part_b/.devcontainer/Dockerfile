FROM node:24-bookworm-slim AS node-runtime

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libatomic1 libstdc++6 tar \
    && rm -rf /var/lib/apt/lists/*

# The teacher adapter needs the Node.js runtime
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
RUN node --version

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
