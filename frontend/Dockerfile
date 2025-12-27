# ==========================================
# Stage 1: Builder
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (cached if package.json unchanged)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

# Build for production
RUN npm run build

# ==========================================
# Stage 2: Runner
# ==========================================
FROM nginx:1.25-alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Security: Run as non-root (optional, but recommended for strict environments)
# check if the user exists first to avoid errors, alpine nginx image acts differently sometimes
# keeping it simple for now standard nginx image runs as root to bind port 80,
# but internally worker processes can drop privs.
# For now, we stick to default behaviors unless requested.

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
