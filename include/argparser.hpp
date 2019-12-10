
#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>

extern char *optarg;
extern int optind, opterr, optopt;

struct Option{
    char opt;
    std::string str;
    bool is_single, is_valid;
};

class Argument{
    private:
    std::string m_args;
    std::vector<Option> m_options;

    public:
    Argument()=default;
    Argument(const std::string& args, int argc, const char* argv[]){
        parse(args, argc, (char* const *)argv);
    }
    Argument(const std::string& args, int argc, char* argv[]){
        parse(args, argc, (char* const *)argv);
    }
    ~Argument(){}

    void parse(const std::string& args, int argc, char* const argv[]);
    uint options_check(const std::string& options) const;
    bool options_exist(const std::string& options) const;
    std::string get_option_value(char option) const;
};
