#include <iostream>
#include <filesystem>
#include <random>
#include "handler.hpp"
#include "struct/System.hpp"
#include "IO/io_gro.hpp"
#include "alg/generator.hpp"
#include "timer.hpp"

int main(int argc, char const *argv[])
{
    double min_dist2 = 0.0064;
    double opt_dist2 = 0.0144;
    int iteration_lim = 10;

    std::string inp_file = "";
    std::string outp_file = "";

    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        std::string arg_str = argv[arg_index];
        if (arg_str == "-f") inp_file = std::string(argv[++arg_index]);
        else if (arg_str == "-o") outp_file = std::string(argv[++arg_index]);
        else if (arg_str == "-mn2") min_dist2 = std::stod(argv[++arg_index]);
        else if (arg_str == "-opt2") opt_dist2 = std::stod(argv[++arg_index]);
        else if (arg_str == "-il") iteration_lim = std::stoi(argv[++arg_index]);
        // else if (arg_str == "-h") {
        //     // PrintUsageInfo();
        //     exit(0);
        // }
        else {
            std::cerr << "Error: Command-line argument '" << arg_str << "' not recognized." << std::endl;
            exit(-1);
        }
    }

    if (inp_file == "") {
        std::cerr << "Give input .gro file" << std::endl;
        exit(-1);
    }

    if (outp_file == "") {
        std::cerr << "Give output .gro file" << std::endl;
        exit(-1);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::filesystem::path inp(inp_file);
    System sys = read_sys(inp);

    push_atoms_apart(sys, gen, min_dist2, opt_dist2, iteration_lim);

    std::filesystem::path outp(outp_file);
    write_sys(sys, outp);

    return 0;
}

