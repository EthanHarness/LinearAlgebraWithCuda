#include "ExceptionWithDetails.h"

// https://stackoverflow.com/questions/56046062/linux-addr2line-command-returns-0
size_t ExceptionWithDetails::ConvertToVMA(size_t addr) {
    Dl_info info;
    struct link_map* link_map;
    dladdr1((void*)addr, &info, (void**)&link_map, RTLD_DL_LINKMAP);
    return addr - link_map->l_addr;
}

// https://stackoverflow.com/questions/56046062/linux-addr2line-command-returns-0
void ExceptionWithDetails::capture_stack_trace(const std::string& arg) {
    std::cout << "\n\nException: " << arg << "\n";
    std::cout << "Stack trace:\n";

    void* callstack[128];
    int frame_count = backtrace(callstack, sizeof(callstack) / sizeof(callstack[0]));
    for (int i = 0; i < frame_count; i++) {
        char location[1024];
        Dl_info info;
        if (dladdr(callstack[i], &info)) {
            // use addr2line; dladdr itself is rarely useful (see doc)
            char command[256];
            size_t VMA_addr = ConvertToVMA((size_t)callstack[i]);
            std::string binOrLib = info.dli_fname;
            std::cout << "Binary/Library: " << binOrLib << "\n";
            VMA_addr -= 1;    // https://stackoverflow.com/questions/11579509/wrong-line-numbers-from-addr2line/63841497#63841497
            snprintf(command, sizeof(command), "addr2line -e %s -Ci %zx", info.dli_fname, VMA_addr);
            system(command);
            std::cout << "\n";
        }
    }
}

ExceptionWithDetails::ExceptionWithDetails(const std::string& arg, const char* file, int line) : std::runtime_error(arg) {
    capture_stack_trace(arg);
}

ExceptionWithDetails::~ExceptionWithDetails() noexcept {}
