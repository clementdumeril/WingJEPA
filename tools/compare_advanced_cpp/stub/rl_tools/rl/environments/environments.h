// Minimal stand-in for RLtools' environments.h: fixed_wing.h only derives its Environment tag from this class.
// Lets compare_advanced.cpp include the Fixed-Wing headers (physics only) without fetching RLtools.
#pragma once

namespace rl_tools{
    namespace rl{
        namespace environments{
            template <typename T, typename TI>
            struct Environment{};
        }
    }
}
