#include "SampleRenderer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

    /*! helper function that initializes optix and checks for errors */
    void initOptix()
    {
        // -------------------------------------------------------
        // check for available optix7 capable devices
        // -------------------------------------------------------
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("#osc: no CUDA capable devices found!");
        std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

        // -------------------------------------------------------
        // initialize optix
        // -------------------------------------------------------
        OPTIX_CHECK(optixInit());
    }


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char** av)
    {
        try {
            SampleRenderer  sample;

            const vec2i fbSize(vec2i(1200, 1024));
            sample.resize(fbSize);
            sample.render();

            std::vector<uint32_t> pixels(fbSize.x * fbSize.y );
            sample.downloadPixels(pixels.data());

            const std::string fileName = "osc.png";
            stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4, pixels.data(), fbSize.x*sizeof(uint32_t));

            std::cout << GDT_TERMINAL_GREEN
                      << std::endl
                      << "Image rendered, and saved to " << fileName << " ... done." << std::endl
                      << GDT_TERMINAL_DEFAULT
                      << std::endl;
        } catch (std::runtime_error& e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                    << GDT_TERMINAL_DEFAULT << std::endl;
          exit(1);
        }
        return 0;
    }
}