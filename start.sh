#!/bin/bash

# Start backend
cd /app/backend
uvicorn server:app --host 0.0.0.0 --port 8000 &

# Start frontend
cd /app/frontend
npm install -g serve
serve -s dist -l 5173 &

# Keep container running
wait
