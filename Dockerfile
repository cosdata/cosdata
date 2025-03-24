FROM alpine:latest

# Install dependencies
RUN apk add --no-cache curl bash

# Install Cosdata server
RUN curl -sL https://cosdata.io/install.sh | bash

# Create symlink by sourcing the bashrc file
RUN mkdir -p /opt && \
    bash -c 'source $HOME/.bashrc && ln -sf $COSDATA_HOME /opt/cosdata'

# Expose ports
EXPOSE 8443
EXPOSE 50051

# Persist the ENV variable for all future Docker commands
ENV COSDATA_HOME=/opt/cosdata

# Use the symlink path to modify the config
RUN sed -i 's/host = "127.0.0.1"/host = "0.0.0.0"/' /opt/cosdata/config/config.toml

# Start using the symlink path
CMD ["/opt/cosdata/bin/start-cosdata"]