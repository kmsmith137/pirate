#include "../include/pirate/internals/FakeServer.hpp"
#include "../include/pirate/internals/utils.hpp"  // make_chord_dedispersion_config()
#include "../include/pirate/DedispersionPlan.hpp"

#include <gputils/system_utils.hpp>  // gputils::mlockall_x()

using namespace std;
using namespace pirate;


void test_01_weird_lucky_unlucky_pcie_behavior()
{
    // This incredibly simple test (just copy 1G host->GPU0 in a loop) already shows bimodal lucky/unlucky behavior!
    FakeServer::Params sp;
    sp.server_name = "Lucky/unlucky PCIe";
    sp.num_iterations = 50;
    // sp.use_hugepages = false;
    
    ssize_t nb = 1024L * 1024L * 1024L;    
    sp.nbytes_h2g = nb;
    sp.nbytes_g2h = 0;
    sp.ngpu = 1;

    FakeServer::run(sp);
}


void test_02_ssds()
{
    FakeServer::Params sp;
    sp.server_name = "Test SSDs";
    sp.num_iterations = 1000;
    
    sp.nbytes_per_ssd = 1024L * 1024L * 1024L;
    sp.nbytes_per_file = 16L * 1024L * 1024L;
    sp.nwrites_per_file = 16;
    sp.nthreads_per_ssd = 2;
    
    // sp.ssd_list = { "/ssd_micron", "/ssd_samsung" };
    // sp.ssd_list = { "/ssd_micron" };
    sp.ssd_list = { "/ssd_samsung" };

    FakeServer::run(sp);
}


void test_03_receive_data()
{
    FakeServer::Params sp;
    sp.server_name = "Receive data";
    sp.num_iterations = 20;
    sp.sleep_usec = 1000 * 1000;
    
    sp.ipaddr_list = { "10.1.1.2", "10.1.2.2" };
    sp.nconn_per_ipaddr = 64;

    FakeServer::run(sp);
}


void test_04_full_server()
{
    // One FakeServer iteration is supposed to represent 128 beams, and one second of data.
    // (In contrast to test_05_dedispersion_plan(), which represents 128 beams and TWO seconds.)
    
    FakeServer::Params sp;
    sp.server_name = "Full server";
    sp.num_iterations = 500;
    // sp.use_hugepages = false;

    // Specifying memcpy_blocksize is important here!
    // Otherwise, we'll use 7-8 GB transfers, and it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    sp.memcpy_blocksize = 1024L * 1024L * 1024L;
    
    sp.nbytes_h2g = 8L * 1024L * 1024L * 1024L;
    sp.nbytes_g2h = 7L * 1024L * 1024L * 1024L;
    sp.nbytes_h2h = 2L * 1024L * 1024L * 1024L;  // will need 1 RW cycle after receiving data
    
    sp.nbytes_downsample = 2L * 1024L * 1024L * 1024L;

#if 1
    // SSDs
    sp.nbytes_per_ssd = 1536L * 1024L * 1024L;
    sp.nbytes_per_file = 16L * 1024L * 1024L;
    sp.nwrites_per_file = 16;
    sp.nthreads_per_ssd = 2;
    sp.ssd_list = { "/ssd_micron", "/ssd_samsung" };
#endif

#if 1
    sp.ipaddr_list = { "10.1.1.2", "10.1.2.2" };
    sp.nconn_per_ipaddr = 64;
#endif
    
    FakeServer::run(sp);
}


#if 0
void test_05_dedispersion_plan()
{
    // One FakeServer iteration is supposed to represent 128 beams, and two seconds of data.
    // (In contrast to test_04_full_server(), which represents 128 beams and ONE second.)
    
    FakeServer::Params sp;
    sp.server_name = "Full server + dedispersion plan";
    sp.num_iterations = 250;
    // sp.use_hugepages = false;

    // FIXME bitrotted -- the function make_chord_dedispersion_config() no longer exists.
    // My plan here is to define a YAML format for the FakeServer. Then the FakeServer YAML
    // file would include dedispersion plan info.
    DedispersionConfig config = make_chord_dedispersion_config();
    
    sp.dedispersion_plan = make_shared<DedispersionPlan> (config);

    FakeServer::run(sp);
}
#endif


void test_06_pcie_gmem()
{
    FakeServer::Params sp;
    sp.server_name = "Full server + dedispersion plan";

    // Note: only using one GPU for now!
    sp.ngpu = 1;

    sp.nbytes_h2g = 5L * 1024L * 1024L * 1024L;
    sp.nbytes_g2h = 5L * 1024L * 1024L * 1024L;
    sp.memcpy_blocksize = 1L * 1024L * 1024L * 1024L;

    sp.nbytes_gmem_kernel = 140L * 1024L * 1024L * 1024L;
    
    FakeServer::run(sp);
}


int main(int argc, char **argv)
{
    gputils::mlockall_x();
    
    // test_01_weird_lucky_unlucky_pcie_behavior();
    // test_02_ssds();
    // test_03_receive_data();
    // test_04_full_server();
    // test_05_dedispersion_plan();
    test_06_pcie_gmem();
    
    return 0;
}
