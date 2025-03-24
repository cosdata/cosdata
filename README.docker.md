# Cosdata Docker Image

## Running the Container

```bash
docker run -it \
  --name cosdata-server \
  -p 8443:8443 \
  -p 50051:50051 \
  cosdatateam/cosdata:latest
```

When prompted, enter your admin key.

## Ports

- 8443: HTTP API
- 50051: gRPC service

## Environment Variables

- `COSDATA_HOME`: Set to /opt/cosdata by default
- `PATH`: Includes /opt/cosdata/bin

## Data Storage

By default, data is stored inside the container. For persistent storage, you can mount a volume:

```bash
docker run -it \
  --name cosdata-server \
  -v cosdata-data:/opt/cosdata/data \
  -p 8443:8443 \
  -p 50051:50051 \
  cosdatateam/cosdata:latest
```

## Health Check

The container includes a health check that monitors the HTTP endpoint. 