#include "../include/pirate/FakeCorrelator.hpp"

using namespace std;
using namespace pirate;


int main(int argc, char **argv)
{
    // Must be hand-synced with fake_server.cu!
    // FIXME use shared configuration file.
    
    vector<string> ipaddr_list = { "10.1.1.2", "10.1.2.2", "10.1.3.2", "10.1.4.2" };
    int tcp_connections_per_ipaddr = 64;
    double gbps_per_ipaddr = 20.0;
    
    FakeCorrelator corr;
    corr.add_endpoint("10.1.1.2", tcp_connections_per_ipaddr, gbps_per_ipaddr, {});
    corr.add_endpoint("10.1.2.2", tcp_connections_per_ipaddr, gbps_per_ipaddr, {});
    corr.add_endpoint("10.1.3.2", tcp_connections_per_ipaddr, gbps_per_ipaddr, {});
    corr.add_endpoint("10.1.4.2", tcp_connections_per_ipaddr, gbps_per_ipaddr, {});
    corr.run();

    return 0;
}
