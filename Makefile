ARCH =
ARCH += -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

GPUTILS_INCDIR=../gputils/include
GPUTILS_LIBDIR=../gputils/lib

NVCC=nvcc -std=c++17 $(ARCH) -m64 -O3 -I$(GPUTILS_INCDIR) --compiler-options -Wall,-march=native
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/pirate/constants.hpp \
  include/pirate/DedispersionConfig.hpp \
  include/pirate/DedispersionPlan.hpp \
  include/pirate/avx256/downsample.hpp \
  include/pirate/avx256/m64_outbuf.hpp \
  include/pirate/avx256/m128_outbuf.hpp \
  include/pirate/gpu/reduce2.hpp \
  include/pirate/gpu/DownsampleKernel.hpp \
  include/pirate/gpu/TransposeKernel.hpp \
  include/pirate/internals/bitvec.hpp \
  include/pirate/internals/cpu_downsample.hpp \
  include/pirate/internals/dedispersion_kernel_implementation.hpp \
  include/pirate/internals/dedispersion_inbufs.hpp \
  include/pirate/internals/dedispersion_outbufs.hpp \
  include/pirate/internals/gpu_downsample.hpp \
  include/pirate/internals/gpu_transpose.hpp \
  include/pirate/internals/file_utils.hpp \
  include/pirate/internals/inlines.hpp \
  include/pirate/internals/utils.hpp \
  include/pirate/internals/Directory.hpp \
  include/pirate/internals/Epoll.hpp \
  include/pirate/internals/FakeCorrelator.hpp \
  include/pirate/internals/FakeServer.hpp \
  include/pirate/internals/File.hpp \
  include/pirate/internals/GpuDedispersionKernel.hpp \
  include/pirate/internals/ReferenceDedisperser.hpp \
  include/pirate/internals/ReferenceDedispersionKernel.hpp \
  include/pirate/internals/ReferenceLagbuf.hpp \
  include/pirate/internals/ReferenceLaggedDownsamplingKernel.hpp \
  include/pirate/internals/ReferenceTree.hpp \
  include/pirate/internals/Socket.hpp \
  include/pirate/internals/UntypedArray.hpp \
  include/pirate/internals/YamlFile.hpp

XFILES = \
  bin/fake_correlator \
  bin/fake_server \
  bin/scratch \
  bin/show_dedispersion_plan \
  bin/test-avx256-m64-outbuf \
  bin/test-cpu-downsampler \
  bin/test-gpu-dedispersion-kernels \
  bin/test-gpu-downsample \
  bin/test-gpu-lagged-downsampler \
  bin/test-gpu-reduce2 \
  bin/test-gpu-transpose \
  bin/test-reference-dedisperser \
  bin/test-reference-tree \
  bin/time-cpu-downsample \
  bin/time-gpu-dedispersion-kernels \
  bin/time-gpu-downsample \
  bin/time-gpu-lagged-downsampler \
  bin/time-gpu-transpose

OFILES = \
  src_lib/cpu_downsample.o \
  src_lib/file_utils.o \
  src_lib/gpu_downsample.o \
  src_lib/gpu_transpose.o \
  src_lib/utils.o \
  src_lib/DedispersionConfig.o \
  src_lib/DedispersionPlan.o \
  src_lib/Directory.o \
  src_lib/Epoll.o \
  src_lib/FakeCorrelator.o \
  src_lib/FakeServer.o \
  src_lib/File.o \
  src_lib/GpuDedispersionKernel.o \
  src_lib/GpuLaggedDownsamplingKernel.o \
  src_lib/ReferenceDedisperser.o \
  src_lib/ReferenceDedispersionKernel.o \
  src_lib/ReferenceLagbuf.o \
  src_lib/ReferenceLaggedDownsamplingKernel.o \
  src_lib/ReferenceTree.o \
  src_lib/Socket.o \
  src_lib/UntypedArray.o \
  src_lib/YamlFile.o \
  src_lib/template_instantiations/dedisp_simple_float16.o \
  src_lib/template_instantiations/dedisp_simple_float32.o \
  src_lib/template_instantiations/dedisp_simple_nolag_float16.o \
  src_lib/template_instantiations/dedisp_simple_nolag_float32.o \
  src_lib/template_instantiations/dedisp_stage0_float16.o \
  src_lib/template_instantiations/dedisp_stage0_float32.o \
  src_lib/template_instantiations/dedisp_stage1_float16.o \
  src_lib/template_instantiations/dedisp_stage1_float32.o

# Used in 'make clean' and 'make source_files.txt'
SRCDIRS = . src_bin src_lib src_lib/template_instantiations include include/pirate include/pirate/avx256 include/pirate/gpu include/pirate/internals

all: $(XFILES)

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

bin/%: src_bin/%.o $(OFILES)
	mkdir -p bin && $(NVCC) -o $@ $^ $(GPUTILS_LIBDIR)/libgputils.a -lyaml-cpp

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) source_files.txt *~ template_instantiations/*.cu
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done
