#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/Dedisperser.hpp"

// asdf-cxx library
#include <asdf/asdf.hxx>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    cout << "Testing asdf-cxx integration..." << endl;
    
    // Check asdf-cxx version
    ASDF_CHECK_VERSION();
    cout << "  asdf-cxx version: " << ASDF::asdf_cxx_version() << endl;
    cout << "  ASDF standard version: " << ASDF::asdf_standard_version() << endl;
    
    // Create a simple ASDF file with some test data
    auto grp = make_shared<ASDF::group>();
    
    // Create a 1D array of integers
    vector<int64_t> data1d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto array1d = make_shared<ASDF::ndarray>(
        data1d,
        ASDF::block_format_t::block,
        ASDF::compression_t::none,
        0,
        vector<bool>(),
        vector<int64_t>{10}
    );
    grp->emplace("integers", array1d);
    
    // Create a 2D array of floats (3x4 matrix)
    vector<double> data2d = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0
    };
    auto array2d = make_shared<ASDF::ndarray>(
        data2d,
        ASDF::block_format_t::block,
        ASDF::compression_t::none,  // No compression (no external dependencies)
        0,  // compression level (unused)
        vector<bool>(),
        vector<int64_t>{3, 4}
    );
    grp->emplace("matrix", array2d);
    
    // Create the ASDF file structure
    auto project = make_shared<ASDF::asdf>(map<string, string>(), grp);
    
    // Write to file
    const string filename = "test_asdf_output.asdf";
    project->write(filename);
    
    cout << "  Wrote test ASDF file: " << filename << endl;
    cout << "  You can inspect it with: python -c \"import asdf; print(asdf.open('" << filename << "').tree)\"" << endl;
    cout << "asdf-cxx test complete!" << endl;
}


}  // namespace pirate

