#include "../include/pirate/internals/FakeCorrelator.hpp"

using namespace pirate;


int main(int argc, char **argv)
{
    FakeCorrelator::Params params;

    params.ipaddr_list = { "10.1.1.1", "10.1.2.1", "10.1.3.1", "10.1.4.1" };
    params.nconn_per_ipaddr = 4;
    params.gbps_per_ipaddr = 16.0;
    // params.use_zerocopy = false;

    FakeCorrelator::run(params);

    return 0;
}
