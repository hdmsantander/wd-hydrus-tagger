# Thin wrapper around the published image: pin tag/digest via build args and add a health probe.
ARG HYDRUS_WEB_IMAGE=ghcr.io/floogulinc/hydrus-web
ARG HYDRUS_WEB_TAG=dev
FROM ${HYDRUS_WEB_IMAGE}:${HYDRUS_WEB_TAG}

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
