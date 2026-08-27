# The frontend image: the SPA built by node, served by nginx. nginx's own
# configuration (server blocks, TLS, proxying to the API) is NOT in this
# image — the host mounts its /etc/nginx (conf + certs) at run time, as the
# deployment always has. Built from this directory as its own context:
#
#   docker build -t turnstile-frontend frontend/

ARG GIT_COMMIT=unknown
ARG DOCKER_TAG=unknown

FROM node:24-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.ts tsconfig*.json ./
COPY public ./public
COPY src ./src
RUN npm run build


FROM nginx:1.27-alpine

ARG GIT_COMMIT
ARG DOCKER_TAG
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.version="${DOCKER_TAG}"

# the built SPA; the host's nginx conf points `root` here
COPY --from=builder /app/dist /usr/share/nginx/html/dist

EXPOSE 80 443
