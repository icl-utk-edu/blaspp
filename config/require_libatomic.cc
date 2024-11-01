#include <atomic>
#include <cstdint>

int main( int argc, char** argv )
{
    std::atomic<std::int64_t> x = 0;
    for (int i = 1; i < argc; ++i) {
        ++x;
    }
    return x;
}
