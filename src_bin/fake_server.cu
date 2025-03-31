#include "../include/pirate/FakeServer.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/system_utils.hpp"
#include "../include/pirate/utils.hpp"     // make_chord_dedispersion_config()

using namespace std;
using namespace pirate;


#if 0

void test_01_weird_lucky_unlucky_pcie_behavior()
{
    // This incredibly simple test (just copy 1G host->GPU0 in a loop) already shows bimodal lucky/unlucky behavior!
    FakeServer::Params sp;
    sp.server_name = "Lucky/unlucky PCIe";
    // sp.use_hugepages = false;

    long nb = 1024L * 1024L * 1024L;    
    sp.nbytes_h2g = nb;
    sp.nbytes_g2h = 0;

    FakeServer::run(sp, 50);
}


void test_02_ssds()
{
    FakeServer::Params sp;
    sp.server_name = "Test SSDs";

    sp.nbytes_per_ssd = 1024L * 1024L * 1024L;
    sp.nbytes_per_file = 16L * 1024L * 1024L;
    sp.nwrites_per_file = 16;
    sp.nthreads_per_ssd = 2;
    
    // sp.ssd_list = { "/ssd_micron", "/ssd_samsung" };
    // sp.ssd_list = { "/ssd_micron" };
    sp.ssd_list = { "/ssd_samsung" };

    FakeServer::run(sp, 1000);
}


void test_03_receive_data()
{
    FakeServer::Params sp;
    sp.server_name = "Receive data";

    sp.sleep_usec = 1000 * 1000;
    sp.add_receiver("10.1.1.2", 64);
    sp.add_receiver("10.1.2.2", 64);

    FakeServer::run(sp, 20);
}

void test_04_full_server()
{
    // One FakeServer iteration is supposed to represent 128 beams, and one second of data.
    // (In contrast to test_05_dedispersion_plan(), which represents 128 beams and TWO seconds.)
    
    FakeServer::Params sp;
    sp.server_name = "Full server";
    // sp.use_hugepages = false;

#if 0
    // Specifying memcpy_blocksize is important here!
    // Otherwise, we'll use 7-8 GB transfers, and it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    sp.memcpy_blocksize = 1024L * 1024L * 1024L;
    sp.nbytes_h2g = 8L * 1024L * 1024L * 1024L;
    sp.nbytes_g2h = 7L * 1024L * 1024L * 1024L;
    sp.nbytes_h2h = 2L * 1024L * 1024L * 1024L;  // will need 1 RW cycle after receiving data
#endif

#if 0
    sp.nbytes_downsample = 2L * 1024L * 1024L * 1024L;
#endif

#if 0
    // SSDs
    sp.nbytes_per_ssd = 1536L * 1024L * 1024L;
    sp.nbytes_per_file = 16L * 1024L * 1024L;
    sp.nwrites_per_file = 16;
    sp.nthreads_per_ssd = 2;
    sp.ssd_list = { "/ssd_micron", "/ssd_samsung" };
#endif

#if 0
    sp.add_receiver("10.1.1.2", 64);
    sp.add_receiver("10.1.2.2", 64);
#endif
    
    FakeServer::run(sp, 500);
}


void test_05_dedispersion_plan()
{
    // One FakeServer iteration is supposed to represent 128 beams, and two seconds of data.
    // (In contrast to test_04_full_server(), which represents 128 beams and ONE second.)
    
    FakeServer::Params sp;
    sp.server_name = "Full server + dedispersion plan";
    // sp.use_hugepages = false;

    // FIXME bitrotted -- the function make_chord_dedispersion_config() no longer exists.
    // My plan here is to define a YAML format for the FakeServer. Then the FakeServer YAML
    // file would include dedispersion plan info.
    DedispersionConfig config = make_chord_dedispersion_config();
    
    sp.dedispersion_plan = make_shared<DedispersionPlan> (config);

    FakeServer::run(sp, 250);
}


void test_06_pcie_gmem()
{
    FakeServer::Params sp;
    sp.server_name = "Full server + dedispersion plan";

    sp.nbytes_h2g = 5L * 1024L * 1024L * 1024L;
    sp.nbytes_g2h = 5L * 1024L * 1024L * 1024L;
    sp.memcpy_blocksize = 1L * 1024L * 1024L * 1024L;

    sp.nbytes_gmem_kernel = 140L * 1024L * 1024L * 1024L;
    
    FakeServer::run(sp, 20);
}
#endif


int main(int argc, char **argv)
{
    sys_mlockall();

    FakeServer fs("New Server");

    for (int igpu = 0; igpu < 2; igpu++) {
	fs.add_memcpy_worker(-1, igpu, 8L*1024L*1024L*1024L, 1024L*1024L*1024L, {});  // h2g
	fs.add_memcpy_worker(igpu, -1, 8L*1024L*1024L*1024L, 1024L*1024L*1024L, {});  // g2h
    }
    
    fs.run(100);
    return 0;
}
