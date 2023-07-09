#include "../include/pirate/internals/FakeCorrelator.hpp"

using namespace pirate;


int main(int argc, char **argv)
{
    // Must be hand-synced with fake_server.cu!
    // FIXME use shared configuration file.
    
    FakeCorrelator::Params params;

    params.ipaddr_list = { "10.1.1.2", "10.1.2.2" };
    params.nconn_per_ipaddr = 64;
    params.gbps_per_ipaddr = 16.0;
    // params.use_zerocopy = false;

    FakeCorrelator::run(params);

    return 0;
}
