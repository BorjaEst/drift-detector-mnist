import groovy.transform.Field
// This file defines variables to be used in the AI4OS-Hub Upstream Jenkins pipeline
// dockerfile : what Dockerfile to use for building, can include path, e.g. docker/Dockerfile
// If *both*, CPU and GPU versions can be built:
// base_cpu_tag : CPU tag for the base docker image (check your Dockerfile)
// base_gpu_tag : GPU tag for the base docker image (check your Dockerfile)

@Field
def dockerfile = 'Dockerfile'


@Field
def base_cpu_tag = '2.3.1-cuda11.8-cudnn8-runtime'

@Field
def base_gpu_tag = '2.3.1-cuda11.8-cudnn8-runtime'



return this;