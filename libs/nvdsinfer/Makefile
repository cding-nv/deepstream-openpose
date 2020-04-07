################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++
SRCS:= nvdsinfer_context_impl.cpp  nvdsinfer_context_impl_capi.cpp \
       nvdsinfer_context_impl_output_parsing.cpp nvdsinfer_conversion.cu \
       nms_cpu.cpp resize_merge_cpu.cpp
INCS:= $(wildcard *.h)
LIB:=libnvds_infer.so

NVDS_VERSION:=4.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -g -std=c++11 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream-4.0/sources/includes/

CFLAGS+= `pkg-config --cflags gstreamer-1.0`

LIBS := -shared -g -Wl,-no-undefined \
	 -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvcaffe_parser \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart \
	-lopencv_objdetect -lopencv_imgproc -lopencv_core

LIBS+= `pkg-config --libs gstreamer-1.0`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_inferutils -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
