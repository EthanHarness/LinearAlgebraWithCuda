GPU_ARCH = compute_80
CUDA_PATH = /usr/local/cuda-11.4/lib64
NVCC_PATH = /usr/local/cuda-11.4/bin/nvcc
DEBUG_FLAG ?= 
#ENZYME_PATH = ../Enzyme/enzyme/.build/Enzyme/ClangEnzyme-16.so

SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./inc
LIBDIR = ./lib

# List of all .cu files in the SRCDIR
CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
# List of all object files that need to be created in OBJDIR
CU_OBJECTS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
# List of all .cpp files in the SRCDIR
CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
# List of all object files that need to be created in OBJDIR
CPP_OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))

# Name of the final executable
EXECUTABLE = MyProgram

# The main .cu file in the current directory
MAIN_CU = kernel.cu
# The object file for the main .cu file
MAIN_OBJ = $(OBJDIR)/kernel.o

ifeq ($(DEBUG), 1)
    DEBUG_FLAG := -g
else
    DEBUG_FLAG :=
endif

# Default rule
all: $(EXECUTABLE)

# Rule to compile with Enzyme (Taken out for now)
#clang++ -c $< -fplugin=$(ENZYME_PATH) -O2 --cuda-gpu-arch=$(GPU_ARCH) -I$(INCDIR) -o $@
# Rule to compile .cu files into .o files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC_PATH) -c $< $(DEBUG_FLAG) -O2 -arch=$(GPU_ARCH) -I$(INCDIR) -o $@

# Rule to compile .cpp files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(NVCC_PATH) -c $< $(DEBUG_FLAG) -O2 -arch=$(GPU_ARCH) -I$(INCDIR) -o $@

# Rule to compile with Enzyme (Taken out for now)
#clang++ -c $< -fplugin=$(ENZYME_PATH) -O2 --cuda-gpu-arch=$(GPU_ARCH) -I$(INCDIR) -o $@
# Rule to compile the main .cu file into an .o file
$(MAIN_OBJ): $(MAIN_CU) | $(OBJDIR)
	$(NVCC_PATH) -c $< $(DEBUG_FLAG) -O2 -arch=$(GPU_ARCH) -I$(INCDIR) -o $@

# Rule to link all object files and create the final executable
$(EXECUTABLE): $(CU_OBJECTS) $(CPP_OBJECTS) $(MAIN_OBJ)
	$(NVCC_PATH) $^ $(DEBUG_FLAG) -L$(CUDA_PATH) -lcudart_static -ldl -lrt -Xcompiler="-pthread" -o $@

# Create the obj directory if it does not exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Make sure the obj directory is created before we try to put files in it
$(CU_OBJECTS) $(MAIN_OBJ): | $(OBJDIR)

# Clean rule for removing .o, .d files and the executable
clean:
	rm -f $(OBJDIR)/*.o $(EXECUTABLE)
