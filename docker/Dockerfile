FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libsm6 \
        libxext6 \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://bootstrap.pypa.io/get-pip.py && \
       python3 get-pip.py && \
       rm get-pip.py

WORKDIR /
ENV OPENCV_VERSION="4.0.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O contrib.zip \
&& unzip contrib.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=ON \
  -DCUDA_ARCH_BIN="7.5" \
  -DCUDA_ARCH_PTX="7.5" \
  -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DENABLE_AVX=ON \
  -DENABLE_AVX2=ON \
  -DENABLE_SSE=ON \
  -DENABLE_SSE2=ON \
  -DENABLE_SSE41=ON \
  -DENABLE_SSE42=ON \
  -DENABLE_SSSE3=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make -j "$(nproc)" \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION} \
&& rm /contrib.zip \
&& rm -r /opencv_contrib-${OPENCV_VERSION}

RUN ln -s \
  /usr/local/lib/python3.6/dist-packages/cv2/cv2.cpython-36m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.6/dist-packages/cv2.so

WORKDIR /root
ADD requirements.txt /root/
RUN pip3 --no-cache-dir install -r requirements.txt \
&& rm requirements.txt

ENV KERAS_BACKEND=tensorflow

CMD ["/bin/bash"]