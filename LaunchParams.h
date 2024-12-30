//
// Created by bei on 24-12-29.
//

#include "gdt/math/vec.h"

namespace osc {
    using namespace gdt;

    struct LaunchParams {
        int frameID {0};
        uint32_t *colorBuffer;
        vec2i fbSize;
    };
}