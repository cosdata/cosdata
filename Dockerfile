FROM alpine:latest

RUN apk add --no-cache curl bash

# Download and run the install script, then set up symlinks and environment
RUN curl -sL https://cosdata.io/install.sh | bash && \
    REPO="cosdata/cosdata" && \
    LATEST_RELEASE=$(curl -s "https://api.github.com/repos/$REPO/releases" | \
      grep '"tag_name":' | \
      sed -E 's/.*"([^"]+)".*/\1/' | \
      sort -V | \
      tail -n 1) && \
    COSDATA_HOME="$HOME/cosdata-$LATEST_RELEASE" && \
    mkdir -p /opt && \
    ln -sf "$COSDATA_HOME" /opt/cosdata && \
    echo "export COSDATA_HOME=\"$COSDATA_HOME\"" >> /root/.profile && \
    echo "export PATH=\"$COSDATA_HOME/bin:\$PATH\"" >> /root/.profile

ENV SHELL=/bin/bash
WORKDIR /opt/cosdata

CMD ["/bin/bash", "-l", "-c", "start-cosdata"]
