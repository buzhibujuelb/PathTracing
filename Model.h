//
// Created by bei on 25-3-25.
//

#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include <map>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
    using namespace gdt;

    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        // material data:
        vec3f diffuse;
        int diffuseTextureID{-1};

        void addCube(const vec3f &center, const vec3f &size);

        void addUnitCube(const affine3f &xfm);
    };

    struct Texture {
        ~ Texture() { if (pixel) delete pixel; }
        uint32_t *pixel{nullptr};
        vec2i resolution = {-1};
    };

    struct Model {
        ~Model() {
            for (auto mesh: meshes) delete mesh;
            for (auto texture: textures) delete texture;
        }

        std::vector<TriangleMesh *> meshes;
        std::vector<Texture *> textures;
        //! bounding box of all vertices in the model
        box3f bounds;
        Texture * envmap=nullptr;
    };

    Model *loadOBJ(const std::string &objFile);

    int loadTexture(Model *model, std::map<std::string, int> &knownTextures, const std::string &textureFileName, const std::string &modelPath);

    int loadEnvmap(Model *model, const std::string &Path);
}
