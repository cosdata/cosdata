# Use Ubuntu 24.04 (Noble) as base
FROM ubuntu:noble

# Install build dependencies, curl, and netcat for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create a non-root user
RUN useradd -m -s /bin/bash cosdata

# Set working directory
WORKDIR /tmp/build

# Clone the repository and verify files
RUN git clone https://github.com/cosdata/cosdata.git . && \
    ls -la && \
    cat config.toml

# Build the project
RUN cargo build --release && \
    ls -la target/release/

# Setup final directory structure with verification
RUN mkdir -p /home/cosdata/bin && \
    mkdir -p /home/cosdata/config && \
    mkdir -p /home/cosdata/data && \
    cp target/release/cosdata /home/cosdata/bin/cosdata && \
    cp config.toml /home/cosdata/config/config.toml && \
    ls -la /home/cosdata/bin && \
    ls -la /home/cosdata/config

# Modify config.toml to use 0.0.0.0 instead of 127.0.0.1
RUN sed -i 's/host = "127.0.0.1"/host = "0.0.0.0"/' /home/cosdata/config/config.toml && \
    sed -i 's/host = "127.0.0.1" # Optional/host = "0.0.0.0" # Modified for Docker/' /home/cosdata/config/config.toml

# Add binary directory to PATH
ENV PATH="/home/cosdata/bin:${PATH}"

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
echo "Starting Cosdata..."\n\
if [ -z "$COSDATA_ADMIN_KEY" ]; then\n\
    echo "Error: COSDATA_ADMIN_KEY environment variable is required"\n\
    exit 1\n\
fi\n\
\n\
# Set COSDATA_HOME environment variable\n\
export COSDATA_HOME="/home/cosdata"\n\
\n\
echo "Environment:"\n\
echo "COSDATA_HOME=$COSDATA_HOME"\n\
echo "PATH=$PATH"\n\
\n\
echo "Directory contents:"\n\
ls -R /home/cosdata\n\
\n\
echo "Config file contents:"\n\
cat /home/cosdata/config/config.toml\n\
\n\
echo "Starting Cosdata with admin key..."\n\
exec /home/cosdata/bin/cosdata --admin-key "$COSDATA_ADMIN_KEY" --confirmed' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Set ownership
RUN chown -R cosdata:cosdata /home/cosdata

# Clean up build directory
RUN rm -rf /tmp/build

# Switch to non-root user and working directory
USER cosdata
WORKDIR /home/cosdata

# Set environment variables
ENV COSDATA_HOME=/home/cosdata
ENV PATH="/home/cosdata/bin:${PATH}"

# Expose the default ports (both HTTP and gRPC)
EXPOSE 8443 50051

# Add health check that considers 404 as healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8443/ | grep -q "404" || exit 1

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]