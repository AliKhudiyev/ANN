#include <vector>

unsigned max_index(const std::vector<double>& vec){
    unsigned index=0;
    double max=0;

    for(unsigned i=0;i<vec.size();++i){
        if(max<vec[i]){
            max=vec[i];
            index=i;
        }
    }

    return index;
}

bool is_same(const std::vector<double>& vec1, const std::vector<double>& vec2){
    return max_index(vec1)==max_index(vec2);
}