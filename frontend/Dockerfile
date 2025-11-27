# ----- Builder Stage -----
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./
COPY frontend/next.config.js ./
COPY frontend/tailwind.config.js ./
COPY frontend/postcss.config.js ./
COPY frontend/tsconfig.json ./

# Install rsync for sync-docs script and all dependencies
RUN apk add --no-cache rsync && \
    npm install

# Copy source code and docs (for sync-docs script)
COPY frontend/app/ ./app/
COPY frontend/components/ ./components/
COPY frontend/lib/ ./lib/
COPY frontend/src/ ./src/
COPY frontend/public/ ./public/
COPY docs/ ../docs/

# Build the application
RUN npm run build

# ----- Runtime Stage -----
FROM node:20-alpine AS runner

WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Run the application
CMD ["node", "server.js"]
