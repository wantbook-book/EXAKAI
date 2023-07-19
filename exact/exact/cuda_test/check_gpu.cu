#include <iostream>

int main(){
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::cout<<"device name:  "<<prop.name<<std::endl;
    std::cout<<"max grid size:  "
        <<prop.maxGridSize[0]<<", "
        <<prop.maxGridSize[1]<<", "
        <<prop.maxGridSize[2]<<std::endl;
    std::cout<<"max block size:  "
        <<prop.maxThreadsDim[0]<<", "
        <<prop.maxThreadsDim[1]<<", "
        <<prop.maxThreadsDim[2]<<std::endl;
    return 0;
}