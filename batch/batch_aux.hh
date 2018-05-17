#ifndef BATCH_AUX_HH
#define BATCH_AUX_HH

namespace blas{

namespace batch{

template<typename T>
T extract(std::vector<T> const &ivector, const int64_t index){
    return (ivector.size() == 1) ? ivector[0] : ivector[index];
}

}        // namespace batch
}        // namespace blas

#endif    // BATCH_AUX_HH
