# Build frontend
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Main image
FROM python:3.10-slim
WORKDIR /app

# Copy frontend build
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Setup backend
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy start script
COPY start.sh /app/
RUN chmod +x /app/start.sh


EXPOSE 8000 5173
ENV DB_ENGINE=sqlite+pysqlite:////app/firstaid.db
ENV VITE_APP_AXIOS_URL=http://localhost:8000

CMD ["/app/start.sh"]
