# Docker Container Usage Guide

## Building the Container

To build the Docker container, run:

```bash
docker build -t first-aid .
```

### With Persistent Storage

To persist the database, mount a volume:
```bash
docker run -p 8000:8000 -p 5173:5173 -v $(pwd)/data:/app/data first-aid
```

## Environment Variables

The container supports the following environment variables:

### Backend Configuration
- `PASSWORD_LENGTH`: Length of generated passwords (default: 13)
- `JWT_ALGORITHM`: Algorithm used for JWT tokens (default: "HS256")
- `ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiration time in minutes (default: 30)
- `ADMIN_USER`: Admin username (default: "admin")
- `ADMIN_EMAIL`: Admin email address (default: "admin@localhost.com")
- `ADMIN_DEFAULT_PASSWORD`: Default admin password (default: "N8Lwcs4G7Vbmkp5t5g")
- `USE_MIDDLEWARE`: Enable/disable middleware (default: True)
- `SAVE_PATH`: Path for saved files (default: "../files")

### Email Configuration
- `SMTP_TLS`: Enable SMTP TLS (default: True)
- `SMTP_SSL`: Enable SMTP SSL (default: False)
- `SMTP_PORT`: SMTP port number (default: 587)
- `SMTP_HOST`: SMTP host address (default: "localhost")
- `SMTP_USER`: SMTP username (default: "")
- `SMTP_PASSWORD`: SMTP password (default: "")
- `EMAILS_FROM_EMAIL`: Sender email address (default: "")
- `EMAILS_FROM_NAME`: Sender name (default: "")

### Example with Full Configuration

```bash
docker run -p 8000:8000 -p 5173:5173 \
  -e PASSWORD_LENGTH=16 \
  -e JWT_ALGORITHM="HS256" \
  -e ACCESS_TOKEN_EXPIRE_MINUTES=60 \
  -e ADMIN_USER="admin" \
  -e ADMIN_EMAIL="admin@yourdomain.com" \
  -e ADMIN_DEFAULT_PASSWORD="your_secure_password" \
  -e USE_MIDDLEWARE=true \
  -e SAVE_PATH="/app/data/files" \
  -e SMTP_TLS=true \
  -e SMTP_PORT=587 \
  -e SMTP_HOST="smtp.yourdomain.com" \
  -e SMTP_USER="your_smtp_user" \
  -e SMTP_PASSWORD="your_smtp_password" \
  -e EMAILS_FROM_EMAIL="noreply@yourdomain.com" \
  -e EMAILS_FROM_NAME="First-AID System" \
  -v $(pwd)/data:/app/data \
  first-aid
```

## Accessing the Application

After running the container:
- Frontend UI: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

Default login credentials:
- Username: `admin`
- Password: `N8Lwcs4G7Vbmkp5t5g` (unless changed via `ADMIN_DEFAULT_PASSWORD`)
