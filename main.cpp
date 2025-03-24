#include "SampleRenderer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"
#include "GL/gl.h"
#include "glfWindow/GLFWindow.h"

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


    struct SampleWindow : public GLFWindow
    {
        SampleWindow(const std::string& title)
            : GLFWindow(title)
        {}
        virtual void render() override
        {
            sample.render();
        }
        virtual void draw() override
        {
            sample.downloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());
            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glDisable(GL_DEPTH_TEST);
            glViewport(0, 0, fbSize.x, fbSize.y);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);
            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);
                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);
                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }
        virtual void resize(const vec2i& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }
        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        SampleRenderer        sample;
        std::vector<uint32_t> pixels;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char** av)
    {
        try {
            SampleWindow* window = new SampleWindow("Optix 7 Course Example");
            window->run();
        }
        catch (std::runtime_error& e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
            exit(1);
        }
        return 0;
    }
}