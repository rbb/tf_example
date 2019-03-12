Building an Image Classifier with Tensorflow, running on a Pi
==============================================================

Starting with Ubuntu 18.04 to train the neural net, and convert to
tensorflow-lite, which will run on Raspian Stretch.

# Installing TensorFlow

## Things that did not work

### Installing with pip
Installing with pip completes, but on old hardware, this fails to run.
Version 1.50 and earlier are compiled without AVX instructions, which will work
on older hardware. However, TensorFlow Lite requires much newer releases.
    ```
    sudo apt install python3-pip libcublas9.1
    pip3 install tensorflow==1.5.0
    #wget https://github.com/schrepfler/tensorflow-community-wheels/releases/download/v1.12.0/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
    #pip3 install tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
    ```
    
### Running with a docker image
Running Tensorflow from a docker image crashes (on older hardware). Probably 
because of AVX instructions?
    ```
    sudo apt install docker.io
    sudo adduser -aG docker $USER
    # completely logout and log back in, or reboot 
    
    docker pull tensorflow/tensorflow:nightly-devel
    docker volume create mydata      # Create someplace for the container to store stuff
    docker run -v mydata -it tensorflow/tensorflow:nightly-devel
    
    # From within the bash shell running in the docker container...
    # Note: the default for this image is to run as root
    apt-get update
    apt-get install crossbuild-essential-armhf
    cd /tensorflow
    ./tensorflow/lite/tools/make/download_dependencies.sh
    ./tensorflow/lite/tools/make/build_rpi_lib.sh

    # This should compile a static library in: `tensorflow/lite/tools/make/gen/lib/rpi_armv7/libtensorflow-lite.a`.
    ```

## Works: Compiling from source
So, that leaves us with compiling from source...
    ```
    # https://medium.com/@Oysiyl/install-tensorflow-1-8-0-with-gpu-from-source-on-ubuntu-18-04-bionic-beaver-35cfa9df3600
    sudo apt install build-essential cmake python3-dev
    sudo apt install libcupti-dev libcuda-9.1-1 nvidia-cuda-dev

    wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
    chmod +x  bazel-0.22.0-installer-linux-x86_64.sh
    ./bazel-0.22.0-installer-linux-x86_64.sh

    #git clone tensorflow
    cd tensorflow
    ./configure
    # The build process chews up too many resources (memory?), so we need to
    # dial it back wit hthe ram_utilization_factor and/or jobs options.
    #bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel build --ram_utilization_factor=40 --jobs=6 --config=opt //tensorflow/tools/pip_package:build_pip_package

    # Be sure to uninstall (with both pip3 and pip2) all old versions of
    # tensorflow, and probably delete caches in $HOME/.local/lib/python3.6/site-packages/tensorflow/
    # See: https://stackoverflow.com/questions/51299194/importerror-cannot-import-name-abs
    bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
    cd tensorflow_pkg
    pip3 install --upgrade --no-cache tensorflow*.whl

    # Verify the install
    python3
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    ```

Note: *DO NOT* (pip3) install mxnet, it will mess up the tensorflow install

# Install Models

    Building mobilenet doesn't work - missing a workspace
    ```
    #CHECKPOINT_DIR=/tmp/checkpoints
    CHECKPOINT_DIR=./checkpoints
    mkdir ${CHECKPOINT_DIR}
    wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
    tar -xvf mobilenet_v1_1.0_224.tgz
    mv mobilenet_v1_1.0_224.ckpt.* ${CHECKPOINT_DIR}
    ```

    Install tensorflow_hub
    ```
    pip3 install tensorflow_hub
    ```

# Links
- https://www.tensorflow.org/lite/guide/build_rpi
- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
- https://www.tensorflow.org/hub/tutorials/image_retraining 
- https://github.com/schrepfler/tensorflow-community-wheels/releases/tag/v1.12.0

<-- vim: ts=4:sw=4
-->
