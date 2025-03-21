# Cosdata Docker Image

## Running the Container

```bash
docker run -d \
  --name cosdata-server \
  -e COSDATA_ADMIN_KEY=your_admin_key \
  -v cosdata-data:/home/cosdata/data \
  -p 8080:8080 \
  -p 50051:50051 \
  cosdata/cosdata:latest
```

## Environment Variables

- `COSDATA_ADMIN_KEY` (required): Your admin key for Cosdata
- `COSDATA_HOME`: Set to /home/cosdata by default
- `PATH`: Includes /home/cosdata/bin

## Ports

- 8080: HTTP API
- 50051: gRPC service

## Volumes

- `/home/cosdata/data`: Persistent data storage

## Health Check

The container includes a health check that monitors the HTTP endpoint. 