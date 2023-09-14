GPU_ARCH = sm_80
CUDA_PATH = /usr/local/cuda-12.2/lib64
ENZYME_PATH = ../Enzyme/enzyme/.build/Enzyme/ClangEnzyme-16.so

SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./inc
LIBDIR = ./lib

# List of all .cu files in the SRCDIR
CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
# List of all object files that need to be created in OBJDIR
CU_OBJECTS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SOURCES))

# Name of the final executable
EXECUTABLE = MyProgram

# The main .cu file in the current directory
MAIN_CU = kernel.cu
# The object file for the main .cu file
MAIN_OBJ = $(OBJDIR)/kernel.o

# Default rule
all: $(EXECUTABLE)

# Rule to compile .cu files into .o files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	clang++ -c $< -fplugin=$(ENZYME_PATH) -O2 --cuda-gpu-arch=$(GPU_ARCH) -I$(INCDIR) -o $@

# Rule to compile the main .cu file into an .o file
$(MAIN_OBJ): $(MAIN_CU)
	clang++ -c $< -fplugin=$(ENZYME_PATH) -O2 --cuda-gpu-arch=$(GPU_ARCH) -I$(INCDIR) -o $@

# Rule to link all object files and create the final executable
$(EXECUTABLE): $(CU_OBJECTS) $(MAIN_OBJ)
	clang++ $^ -L$(CUDA_PATH) -lcudart_static -ldl -lrt -pthread -o $@

# Create the obj directory if it does not exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Make sure the obj directory is created before we try to put files in it
$(CU_OBJECTS) $(MAIN_OBJ): | $(OBJDIR)

# Clean rule for removing .o files and the executable
clean:
	rm -f $(OBJDIR)/*.o $(EXECUTABLE)
