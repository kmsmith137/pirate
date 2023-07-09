#include "../include/pirate/internals/FakeServer.hpp"
#include <gputils/system_utils.hpp>  // gputils::mlockall_x()

using namespace std;
using namespace pirate;


int main(int argc, char **argv)
{
    gputils::mlockall_x();
    
    FakeServer::Params sp;
    sp.server_name = "Test node validation";
    sp.num_iterations = 20;

    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    sp.memcpy_blocksize = 1024L * 1024L * 1024L;

    sp.nbytes_h2g = 15L * 1024L * 1024L * 1024L;
    sp.nbytes_g2h = 12L * 1024L * 1024L * 1024L;
    sp.nbytes_downsample = 5L * 1024L * 1024L * 1024L;
    
    sp.ipaddr_list = { "10.1.1.1", "10.1.2.1", "10.1.3.1", "10.1.4.1" };
    sp.nconn_per_ipaddr = 4;

#if 0
    sp.nbytes_per_ssd = 4L * 1024L * 1024L * 1024L;
    sp.nbytes_per_file = 16L * 1024L * 1024L;
    sp.nwrites_per_file = 16;
    sp.nthreads_per_ssd = 2;
    sp.ssd_list = { "/ssd1", "/ssd2" };
#endif

    FakeServer::run(sp);
   
    return 0;
}
