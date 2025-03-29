#include "SampleRenderer.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <imgui_impl_opengl3.h>

#include "3rdParty/stb_image_write.h"
#include "GL/gl.h"
#include "glfWindow/GLFWindow.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
    /*! helper function that initializes optix and checks for errors */
    void initOptix() {
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


    struct SampleWindow : public GLFCameraWindow {
        SampleWindow(const std::string &title, const Model *model, const Camera &camera, const float worldScale)
            : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale), sample(model) {
            sample.setCamera(camera);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();

            // 设置样式（可选）
            ImGui::StyleColorsDark();

            // 绑定到当前 OpenGL 上下文（GLFW）
            ImGui_ImplGlfw_InitForOpenGL(handle, true); // <-- 你需要传入你的 GLFWwindow*
            ImGui_ImplOpenGL3_Init("#version 130");
            glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }

        void mouseButton(int button, int action, int mods) override {
            ImGui_ImplGlfw_MouseButtonCallback(handle, button, action, mods);
            if (ImGui::GetIO().WantCaptureMouse) return;
            GLFCameraWindow::mouseButton(button, action, mods);
        }

        void mouseMotion(const vec2i &newPos) override {
            ImGui_ImplGlfw_CursorPosCallback(handle, newPos.x, newPos.y);
            GLFCameraWindow::mouseMotion(newPos);
        }

        ~SampleWindow() {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }

        virtual void render() override {
            if (cameraFrame.modified) {
                sample.setCamera(Camera{
                    cameraFrame.get_from(),
                    cameraFrame.get_at(),
                    cameraFrame.get_up()
                });
                cameraFrame.modified = false;
            }
            sample.render();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_Always);
            ImGui::Begin("PostProcessing");
            ImGui::SliderFloat("Lightness", &(sample.launchParams.lightness_change), -1.0f, 1.0f);
            ImGui::SliderFloat("Contrast", &(sample.launchParams.contrast_change), -1.0f, 1.0f);
            ImGui::SliderFloat("Saturation", &(sample.launchParams.saturate_change), -1.0f, 1.0f);
            ImGui::End();
        }

        virtual void draw() override {
            sample.downloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_FLOAT;
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
            glOrtho(0.f, (float) fbSize.x, 0.f, (float) fbSize.y, -1.f, 1.f);
            glBegin(GL_QUADS); {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);
                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float) fbSize.y, 0.f);
                glTexCoord2f(1.f, 1.f);
                glVertex3f((float) fbSize.x, (float) fbSize.y, 0.f);
                glTexCoord2f(1.f, 0.f);
                glVertex3f((float) fbSize.x, 0.f, 0.f);
            }
            glEnd();
            ImGui::Render(); // 渲染 ImGui 到当前 OpenGL framebuffer
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        virtual void resize(const vec2i &newSize) {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }

        vec2i fbSize;
        GLuint fbTexture{0};
        SampleRenderer sample;
        std::vector<float4> pixels;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char **av) {
        try {
#ifdef BMW
            Model *model = loadOBJ("../models/bmw/bmw.obj");
            Camera camera = {
                ///*from*/vec3f(-800, 400, -800), /* at */vec3f(0, 0, 0), /* up */vec3f(0.f, 1.f, 0.f)
                vec3f(-82.4492,133.559,-155.28),vec3f(37.1697,165.805,11.0188),vec3f(0,1,0)
            };

            model->meshes.push_back(new TriangleMesh);
            (model->meshes.back())->addCube(vec3f(0.f, 320.f, 0.f), vec3f(800.f, 40.f, 800.f));
            (model->meshes.back())->addCube(vec3f(0.f, -20.f, 0.f), vec3f(800.f, 40.f, 800.f));
            (model->meshes.back())->addCube(vec3f(400.f, 160.f, 000.f), vec3f(10.f, 340.f, 800.f));
            (model->meshes.back())->addCube(vec3f(000.f, 160.f, 400.f), vec3f(800.f, 340.f, 10.f));
#else

            Model *model = loadOBJ("../models/sponza/sponza.obj");
            Camera camera = {
                /*from*/vec3f(-1293.07f, 600+154.681f, -0.7304f),
                /* at */model->bounds.center() - vec3f(0, 400, 0),
                /* up */vec3f(0.f, 1.f, 0.f)
            };
#endif


            const float worldScale = length(model->bounds.span());
            SampleWindow *window = new SampleWindow("PathTracing", model, camera, worldScale);
            window->run();
        } catch (std::runtime_error &e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                    << GDT_TERMINAL_DEFAULT << std::endl;
            exit(1);
        }
        return 0;
    }
}
