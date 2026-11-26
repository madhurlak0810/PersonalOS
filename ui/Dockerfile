# syntax=docker/dockerfile:1.7
#
# The operator UI (Phase K), built and served as static files. The compose
# `ui` profile runs it (docker-compose.yml). Build context: ui/.
#
# The API's address is compiled into the bundle (VITE_API_BASE_URL) and into
# the Content-Security-Policy, so it is a build argument, not a run-time
# setting. The served headers match `npm run preview` (vite.config.ts).

ARG NODE_IMAGE=node:22.21.1-bookworm-slim@sha256:25b3eb23a00590b7499f2a2ce939322727fcce1b15fdd69754fcd09536a3ae2c
ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.29.3-alpine@sha256:5aea7cc516b419e3526f47dd1531be31a56a046cfe44754d94f9383e13e2ee99

FROM ${NODE_IMAGE} AS build
WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG VITE_API_BASE_URL=http://127.0.0.1:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
# Source maps are for development; they are not served.
RUN npm run build && find dist -name '*.map' -delete
# The CSP admits the API's origin only, exactly as vite.config.ts does.
RUN API_ORIGIN=$(node -e 'console.log(new URL(process.env.VITE_API_BASE_URL).origin)') \
    && sed "s|@API_ORIGIN@|${API_ORIGIN}|g" docker/nginx.conf > /tmp/default.conf

FROM ${NGINX_IMAGE} AS ui
COPY --from=build /tmp/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist /usr/share/nginx/html
# nginx-unprivileged runs as uid 101 and keeps its pid and temp files in /tmp,
# so the root filesystem can be read-only (docker-compose.yml mounts a tmpfs).
USER 101
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/"]
