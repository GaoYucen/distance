#include <routingkit/contraction_hierarchy.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using RoutingKit::ContractionHierarchy;
using RoutingKit::ContractionHierarchyQuery;

template<class T>
std::vector<T> read_raw(const std::string& path){
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if(!in) throw std::runtime_error("cannot open " + path);
    auto bytes = in.tellg();
    if(bytes < 0 || (static_cast<unsigned long long>(bytes) % sizeof(T)) != 0)
        throw std::runtime_error("bad raw byte size: " + path);
    std::vector<T> out(static_cast<size_t>(bytes) / sizeof(T));
    in.seekg(0);
    if(!out.empty()) in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()*sizeof(T)));
    if(!in) throw std::runtime_error("short read: " + path);
    return out;
}

template<class T>
void write_raw(const std::string& path, const std::vector<T>& data){
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if(!out) throw std::runtime_error("cannot create " + path);
    if(!data.empty()) out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()*sizeof(T)));
    if(!out) throw std::runtime_error("short write: " + path);
}

int main(int argc, char** argv){
    if(argc != 9){
        std::cerr << "usage: " << argv[0] << " node_count tail.u32 head.u32 weight.u32 query_src.u32 query_dst.u32 output_dist.u32 ch_index\n";
        return 2;
    }
    try{
        const unsigned node_count = static_cast<unsigned>(std::stoul(argv[1]));
        const std::string tail_path=argv[2], head_path=argv[3], weight_path=argv[4];
        const std::string qsrc_path=argv[5], qdst_path=argv[6], out_path=argv[7], ch_path=argv[8];
        auto tail = read_raw<uint32_t>(tail_path);
        auto head = read_raw<uint32_t>(head_path);
        auto weight = read_raw<uint32_t>(weight_path);
        auto qsrc = read_raw<uint32_t>(qsrc_path);
        auto qdst = read_raw<uint32_t>(qdst_path);
        if(tail.size()!=head.size() || tail.size()!=weight.size()) throw std::runtime_error("arc vector length mismatch");
        if(qsrc.size()!=qdst.size()) throw std::runtime_error("query vector length mismatch");
        for(size_t i=0;i<tail.size();++i){
            if(tail[i]>=node_count || head[i]>=node_count) throw std::runtime_error("arc endpoint out of range");
        }
        for(size_t i=0;i<qsrc.size();++i){
            if(qsrc[i]>=node_count || qdst[i]>=node_count || qsrc[i]==qdst[i]) throw std::runtime_error("query endpoint invalid");
        }
        std::cerr << "R4N_FLA_CH_BUILD nodes=" << node_count << " arcs=" << tail.size() << " queries=" << qsrc.size() << "\n";
        auto ch = ContractionHierarchy::build(
            node_count,
            std::vector<unsigned>(tail.begin(), tail.end()),
            std::vector<unsigned>(head.begin(), head.end()),
            std::vector<unsigned>(weight.begin(), weight.end()),
            [](std::string msg){ std::cerr << "ROUTINGKIT_CH " << msg << "\n"; }
        );
        ch.save_file(ch_path);
        std::cerr << "R4N_FLA_CH_INDEX_SAVED " << ch_path << "\n";
        std::vector<uint32_t> dist(qsrc.size());
        #pragma omp parallel
        {
            ContractionHierarchyQuery query(ch);
            #pragma omp for schedule(dynamic, 1024)
            for(long long i=0; i<static_cast<long long>(qsrc.size()); ++i){
                query.reset().add_source(qsrc[static_cast<size_t>(i)]).add_target(qdst[static_cast<size_t>(i)]).run();
                dist[static_cast<size_t>(i)] = query.get_distance();
            }
        }
        write_raw(out_path, dist);
        uint32_t min_d = dist.empty()?0:*std::min_element(dist.begin(), dist.end());
        uint32_t max_d = dist.empty()?0:*std::max_element(dist.begin(), dist.end());
        std::cerr << "R4N_FLA_CH_QUERIES_COMPLETE count=" << dist.size() << " min=" << min_d << " max=" << max_d;
#ifdef _OPENMP
        std::cerr << " omp_max_threads=" << omp_get_max_threads();
#endif
        std::cerr << "\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr << "R4N_FLA_CH_ERROR " << e.what() << "\n";
        return 1;
    }
}
