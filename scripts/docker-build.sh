#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build script for Verax Docker image with ANTLR4 support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="ghcr.io/facebookexperimental/verax/verax-dev:centos9-antlr"
LOCAL_IMAGE_NAME="verax-dev:centos9-antlr"
DOCKERFILE="docker/centos9-antlr.dockerfile"

# Function to print usage
usage() {
    echo "Usage: $0 [build|push|pull|run|shell|test]"
    echo ""
    echo "Commands:"
    echo "  build  - Build the Docker image locally"
    echo "  push   - Push the Docker image to GHCR"
    echo "  pull   - Pull the Docker image from GHCR"
    echo "  run    - Run the build inside the container"
    echo "  shell  - Start an interactive shell in the container"
    echo "  test   - Run tests inside the container"
    echo ""
}

# Function to build the Docker image
build_image() {
    echo "Building Docker image: ${IMAGE_NAME}"
    cd "${PROJECT_ROOT}"
    docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .
    echo "Docker image built successfully: ${IMAGE_NAME}"
}

# Function to push the Docker image
push_image() {
    echo "Pushing Docker image: ${IMAGE_NAME}"
    docker push "${IMAGE_NAME}"
    echo "Docker image pushed successfully: ${IMAGE_NAME}"
}

# Function to pull the Docker image
pull_image() {
    echo "Pulling Docker image: ${IMAGE_NAME}"
    docker pull "${IMAGE_NAME}"
    echo "Docker image pulled successfully: ${IMAGE_NAME}"
}

# Function to run build in container
run_build() {
    echo "Running build inside Docker container"
    docker run --rm \
        -v "${PROJECT_ROOT}:/verax" \
        -v "${PROJECT_ROOT}/.ccache:/verax/.ccache" \
        -e CCACHE_DIR="/verax/.ccache" \
        -e VELOX_DEPENDENCY_SOURCE=SYSTEM \
        -e ICU_SOURCE=SYSTEM \
        -e NUM_THREADS=8 \
        -w /verax \
        "${IMAGE_NAME}" \
        bash -c "
            git config --global --add safe.directory '/verax' &&
            ccache -sz &&
            make debug &&
            echo 'Build completed successfully!'
        "
}

# Function to start interactive shell
run_shell() {
    echo "Starting interactive shell in Docker container"
    docker run --rm -it \
        -v "${PROJECT_ROOT}:/verax" \
        -v "${PROJECT_ROOT}/.ccache:/verax/.ccache" \
        -e CCACHE_DIR="/verax/.ccache" \
        -e VELOX_DEPENDENCY_SOURCE=SYSTEM \
        -e ICU_SOURCE=SYSTEM \
        -w /verax \
        "${IMAGE_NAME}" \
        bash
}

# Function to run tests
run_tests() {
    echo "Running tests inside Docker container"
    docker run --rm \
        -v "${PROJECT_ROOT}:/verax" \
        -w /verax/_build/debug \
        -e GTEST_FILTER="-TpchPlanTest.q01:TpchPlanTest.q03:TpchPlanTest.q05:HiveQueriesTest.basic:PlanTest.filterImport:PlanTest.intersect:PlanTest.except" \
        "${IMAGE_NAME}" \
        bash -c "ctest -j 8 --output-on-failure --no-tests=error"
}

# Main script logic
case "${1:-}" in
    build)
        build_image
        ;;
    push)
        push_image
        ;;
    pull)
        pull_image
        ;;
    run)
        run_build
        ;;
    shell)
        run_shell
        ;;
    test)
        run_tests
        ;;
    *)
        usage
        exit 1
        ;;
esac
