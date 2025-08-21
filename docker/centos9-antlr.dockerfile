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

# Docker image based on velox centos9 with antlr4 4.13.2 support
FROM ghcr.io/facebookincubator/velox-dev:centos9

# Set environment variables for antlr4
ENV ANTLR4_VERSION=4.13.2
ENV ANTLR4_JAR_PATH=/usr/local/lib/antlr-${ANTLR4_VERSION}-complete.jar
ENV ANTLR4_CPP_RUNTIME_PREFIX=/usr/local

# Install Java, unzip, and uuid (required for antlr4)
RUN dnf install -y java-11-openjdk java-11-openjdk-devel unzip libuuid-devel && \
    dnf clean all

# Download and install antlr4 JAR
RUN curl -o ${ANTLR4_JAR_PATH} \
    https://www.antlr.org/download/antlr-${ANTLR4_VERSION}-complete.jar

# Create antlr4 and grun aliases
RUN echo '#!/bin/bash' > /usr/local/bin/antlr4 && \
    echo "java -Xmx500M -cp \"${ANTLR4_JAR_PATH}:$CLASSPATH\" org.antlr.v4.Tool \$*" >> /usr/local/bin/antlr4 && \
    chmod +x /usr/local/bin/antlr4

RUN echo '#!/bin/bash' > /usr/local/bin/grun && \
    echo "java -Xmx500M -cp \"${ANTLR4_JAR_PATH}:$CLASSPATH\" org.antlr.v4.gui.TestRig \$*" >> /usr/local/bin/grun && \
    chmod +x /usr/local/bin/grun

# Build and install antlr4 C++ runtime
WORKDIR /tmp
RUN curl -L -o antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip \
    https://www.antlr.org/download/antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip && \
    mkdir antlr4-source && \
    cd antlr4-source && \
    unzip ../antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=${ANTLR4_CPP_RUNTIME_PREFIX} \
             -DCMAKE_CXX_STANDARD=17 \
             -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
             -Wno-dev && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip antlr4-source

# Add antlr4 C++ runtime to pkg-config
RUN echo "prefix=${ANTLR4_CPP_RUNTIME_PREFIX}" > /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "exec_prefix=\${prefix}" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "libdir=\${exec_prefix}/lib" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "includedir=\${prefix}/include" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "Name: antlr4-runtime" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "Description: ANTLR4 C++ Runtime Library" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "Version: ${ANTLR4_VERSION}" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "Libs: -L\${libdir} -lantlr4-runtime" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc && \
    echo "Cflags: -I\${includedir}/antlr4-runtime" >> /usr/local/lib/pkgconfig/antlr4-runtime.pc

# Update library cache
RUN echo "${ANTLR4_CPP_RUNTIME_PREFIX}/lib" > /etc/ld.so.conf.d/antlr4.conf && \
    ldconfig

# Set environment variables for build systems
ENV ANTLR4_ROOT=${ANTLR4_CPP_RUNTIME_PREFIX}
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

# Verify installation
RUN antlr4 2>&1 | grep -q "ANTLR Parser Generator" && \
    ls -la ${ANTLR4_CPP_RUNTIME_PREFIX}/lib/libantlr4-runtime.* && \
    ls -la ${ANTLR4_CPP_RUNTIME_PREFIX}/include/antlr4-runtime/ && \
    pkg-config --exists antlr4-runtime && \
    echo "ANTLR4 ${ANTLR4_VERSION} installation verified successfully"

WORKDIR /verax

# Maintain the same entrypoint as the base image
ENTRYPOINT ["/bin/bash", "-c", "source /opt/rh/gcc-toolset-12/enable && exec \"$@\"", "--"]
CMD ["/bin/bash"]
