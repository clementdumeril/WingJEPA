// Evaluates the Fixed-Wing C++ advanced_plane model (dynamics.h) on the states of a CSV written by compare.py:
// advanced_aerodynamics (force and moment, body FLU), derivative (dx) and one RK4 step + post_integration (next).
// Build (FW = path to the Fixed-Wing repository):
//   g++ -O2 -std=c++17 -DFW_AIRPLANE=1 -I "$FW/cpp/include" -I stub compare_advanced.cpp -o compare_advanced
// Run:  ./compare_advanced states.csv cpp_out.csv
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <fixed_wing/dynamics.h>

static_assert(FW_AIRPLANE == 1, "compile with -DFW_AIRPLANE=1 (derivative() dispatches on it)");

using namespace fixed_wing;
using T = double;

// input columns (header checked): id, position[3], orientation[4] (w x y z, body FLU to z-up world),
// linear_velocity[3] (world, ground-relative), angular_velocity[3] (FLU), actuators[4] (aileron, elevator, rudder
// [rad], throttle [0, 1]), command[4] (same), wind[3] (world), dt
static const char* INPUT[] = {"id", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
                              "a_ail", "a_ele", "a_rud", "a_thr", "c_ail", "c_ele", "c_rud", "c_thr",
                              "wind_x", "wind_y", "wind_z", "dt"};
constexpr int N_INPUT = sizeof(INPUT) / sizeof(INPUT[0]);

static void write_body(std::FILE* f, const Body<T>& b){
    for(int i = 0; i < 3; i++) std::fprintf(f, ",%.17g", b.position[i]);
    for(int i = 0; i < 4; i++) std::fprintf(f, ",%.17g", b.orientation[i]);
    for(int i = 0; i < 3; i++) std::fprintf(f, ",%.17g", b.linear_velocity[i]);
    for(int i = 0; i < 3; i++) std::fprintf(f, ",%.17g", b.angular_velocity[i]);
    for(int i = 0; i < N_ACTUATORS; i++) std::fprintf(f, ",%.17g", b.actuators[i]);
}

static void body_header(std::FILE* f, const char* prefix){
    const char* names[] = {"px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
                           "a_ail", "a_ele", "a_rud", "a_thr"};
    for(const char* n: names) std::fprintf(f, ",%s%s", prefix, n);
}

int main(int argc, char** argv){
    if(argc != 3){
        std::cerr << "usage: " << argv[0] << " states.csv cpp_out.csv\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if(!in){
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }
    std::string line;
    std::getline(in, line);
    {
        std::stringstream ss(line);
        std::string name;
        for(int i = 0; i < N_INPUT; i++){
            if(!std::getline(ss, name, ',') || name != INPUT[i]){
                std::cerr << "unexpected header column " << i << ": '" << name << "', expected '" << INPUT[i] << "'\n";
                return 1;
            }
        }
    }
    std::FILE* out = std::fopen(argv[2], "w");
    if(!out){
        std::cerr << "cannot write " << argv[2] << "\n";
        return 1;
    }
    std::fprintf(out, "id,Fx,Fy,Fz,Mx,My,Mz");
    body_header(out, "d_");
    body_header(out, "n_");
    std::fprintf(out, "\n");

    Parameters<T> p;
    advanced_plane(p);
    int rows = 0;
    while(std::getline(in, line)){
        if(line.empty()) continue;
        std::vector<T> v;
        std::stringstream ss(line);
        std::string cell;
        while(std::getline(ss, cell, ',')) v.push_back(std::strtod(cell.c_str(), nullptr));
        if((int)v.size() != N_INPUT){
            std::cerr << "row " << rows << ": " << v.size() << " columns\n";
            return 1;
        }
        Body<T> x{};
        int k = 1;
        for(int i = 0; i < 3; i++) x.position[i] = v[k++];
        for(int i = 0; i < 4; i++) x.orientation[i] = v[k++];
        for(int i = 0; i < 3; i++) x.linear_velocity[i] = v[k++];
        for(int i = 0; i < 3; i++) x.angular_velocity[i] = v[k++];
        for(int i = 0; i < N_ACTUATORS; i++) x.actuators[i] = v[k++];
        T command[N_ACTUATORS];
        for(int i = 0; i < N_ACTUATORS; i++) command[i] = v[k++];
        for(int i = 0; i < 3; i++) p.wind[i] = v[k++];
        T dt = v[k++];

        T force[3], moment[3];
        advanced_aerodynamics(p, x, force, moment);
        Body<T> dx, next;
        derivative(p, x, command, dx);
        rk4(p, x, command, dt, next);
        post_integration(p, next);  // quaternion normalized, actuators clipped (as simulate())

        std::fprintf(out, "%d", (int)v[0]);
        for(int i = 0; i < 3; i++) std::fprintf(out, ",%.17g", force[i]);
        for(int i = 0; i < 3; i++) std::fprintf(out, ",%.17g", moment[i]);
        write_body(out, dx);
        write_body(out, next);
        std::fprintf(out, "\n");
        rows++;
    }
    std::fclose(out);
    std::cout << "wrote " << rows << " rows to " << argv[2] << "\n";
    return 0;
}
