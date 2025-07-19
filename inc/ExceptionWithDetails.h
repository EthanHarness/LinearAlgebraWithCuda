#pragma once
#include <stdexcept>
#include <string>
#include <execinfo.h>
#include <iostream>
#include <link.h>
#include <stdlib.h>
#include <stdio.h>

class ExceptionWithDetails : public std::runtime_error {
    std::string msg;
    void capture_stack_trace(const std::string& arg);
    size_t ConvertToVMA(size_t addr);

public:
    ExceptionWithDetails(const std::string &arg, const char *file, int line);
    ~ExceptionWithDetails() noexcept override;
};

// Macro to include file and line number
#define throw_line(arg) throw ExceptionWithDetails(arg, __FILE__, __LINE__);
