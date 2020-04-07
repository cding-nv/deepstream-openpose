# deepstream-openpose


## Run pose demo:  https://github.com/CMU Perceptual Computing Lab/openpose
Platform: xavier, Jetpack 4.3
Notes: 
1. CUDA_cublas_device_LIBRARY NOTFOUND issue
   Solution: https://devtalk.nvidia.com/default/topic/1067111/cuda blas libraries not installed
/?offset=18#5436930
2. Refer to openpose/scripts/ubuntu/install_deps.sh to install deps libs.
3. Refer to openpose/models/getModels.sh to fetch models
4. Build.
  $ cmake D CMAKE_BUILD_TYPE=Debug ..
  $ make -j4
5.These demos can work.
  $ ./build/examples/openpose/openpose.bin
  $ ./build/examples/tutorial_api_cpp/01_body_from_image_default.bin
  $ ...
  Show like this [./COCO_val2014_000000000564_infer.jpg]

<p align="center">
    <img src="./COCO_val2014_000000000564_infer.jpg", width="360">
</p>

## Deploy pose/coco/ "pose_iter_440000.caffemodel" and "pose_deploy_linevec.prototxt" by DeepStream

Pipeline: filesrc -> jpegparse -> nvv4l2decoder -> nvstreammux -> nvinfer (openpose and 18 parts parse)
nvsegvidsual -> nvmultistreamtiler -> (nvegltransform) -> nveglglessink

### Build libnvds_infer.so
$ cd libs/nvinfer
$ make
Backup /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_infer.so
$ sudo ln -sf $(pwd)/libnvds_infer.so /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_infer.so 

### Build openpose-app
$ cd openpose_app
$ make
Change nvinfer_config.txt "model-file" and "proto-file" to be your path
  model-file=
  proto-file=

### Run
$ ./openpose-app ./nvinfer_config.txt COCO_val2014_000000000564.jpg

Show like this. [./COCO_val2014_000000000564_deepstream_infer.jpg]

<p align="center">
    <img src="./COCO_val2014_000000000564_deepstream_infer.jpg", width="360">
</p>

TODO:
Add dsexample plugin after nvinfer and do resize_merge[./], nms[./] and BodyPartConnector and show result by nvosd like as the below. 
<p align="center">
    <img src="./todo.jpg", width="360">
</p>
