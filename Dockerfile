# ---------------------------
# Builder Stage
# ---------------------------
FROM debian:buster AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    openssl \
    ca-certificates \
    curl \
    git \
    musl-tools \
    gcc \
    libssl-dev \
    pkg-config

# Install Rust via rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable && \
    . "$HOME/.cargo/env" && \
    rustup target add x86_64-unknown-linux-musl

# Set environment variables
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc

# Set the working directory
WORKDIR /app

# Copy the Cargo.toml and Cargo.lock files to cache dependencies
COPY Cargo.toml Cargo.lock ./

# Copy only the src/ directory
COPY src/ ./src/

# Build the application
RUN cargo build --release --target x86_64-unknown-linux-musl

# ---------------------------
# Runtime Stage
# ---------------------------
FROM alpine:latest

# Copy the statically linked binary from the builder stage
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/cnn_from_scratch /cnn_from_scratch

# Set the entrypoint
ENTRYPOINT ["/cnn_from_scratch"]
