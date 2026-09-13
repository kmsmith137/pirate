// Minimal .npy reader/writer, so that a spot-test driver can exchange arrays with
// numpy on the pirate side without linking anything.
//
// Supports what a driver needs and no more: version 1.0 files, C order, native
// little-endian, one array per file, dtypes float64/float32/int64/int32/uint8.
// A file it cannot handle is an exception, never a silent misread.
//
// Header-only.  Include it from <test>/driver.cpp as "../npy.hpp".

#ifndef _CHIMEFRB_NPY_HPP
#define _CHIMEFRB_NPY_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {


// The 'descr' string numpy writes for each supported type.  A file whose descr does
// not match the type the driver asked for is rejected rather than reinterpreted.
template<typename T> struct traits;
template<> struct traits<double>   { static const char *descr() { return "<f8"; } };
template<> struct traits<float>    { static const char *descr() { return "<f4"; } };
template<> struct traits<int64_t>  { static const char *descr() { return "<i8"; } };
template<> struct traits<int32_t>  { static const char *descr() { return "<i4"; } };
template<> struct traits<uint8_t>  { static const char *descr() { return "|u1"; } };


template<typename T>
struct array {
    std::vector<size_t> shape;
    std::vector<T> data;

    size_t size() const { return data.size(); }

    // Number of rows, for the (nrows, ncols) shape a driver usually wants.
    size_t rows() const { return shape.empty() ? 0 : shape[0]; }
    size_t cols() const { return (shape.size() >= 2) ? shape[1] : 1; }
};


// -------------------------------------------------------------------------------------------------
//
// Header parsing.  The header is a python dict literal padded with spaces, e.g.
//     {'descr': '<f8', 'fortran_order': False, 'shape': (64, 2), }
// We pull out the three keys by hand rather than writing a parser for a format that
// numpy always emits in the same shape.


inline std::string _find_value(const std::string &hdr, const std::string &key)
{
    size_t i = hdr.find("'" + key + "'");
    if (i == std::string::npos)
        throw std::runtime_error("npy: header has no '" + key + "' key");

    i = hdr.find(':', i);
    if (i == std::string::npos)
        throw std::runtime_error("npy: malformed header near '" + key + "'");

    // Value runs to the next comma at paren depth zero (the shape tuple contains commas).
    size_t j = i + 1;
    int depth = 0;
    for (; j < hdr.size(); j++) {
        char c = hdr[j];
        if (c == '(')
            depth++;
        else if (c == ')')
            depth--;
        else if ((c == ',') && (depth == 0))
            break;
    }

    std::string v = hdr.substr(i+1, j-i-1);

    size_t b = v.find_first_not_of(" \t");
    size_t e = v.find_last_not_of(" \t");
    return (b == std::string::npos) ? std::string() : v.substr(b, e-b+1);
}


inline std::vector<size_t> _parse_shape(const std::string &s)
{
    size_t b = s.find('(');
    size_t e = s.rfind(')');
    if ((b == std::string::npos) || (e == std::string::npos) || (e < b))
        throw std::runtime_error("npy: malformed 'shape' value: " + s);

    std::vector<size_t> shape;
    std::stringstream ss(s.substr(b+1, e-b-1));
    std::string tok;

    while (std::getline(ss, tok, ',')) {
        size_t p = tok.find_first_not_of(" \t");
        if (p == std::string::npos)
            continue;   // trailing comma of a 1-tuple, or an empty 0-d shape
        shape.push_back(strtoul(tok.c_str() + p, nullptr, 10));
    }

    return shape;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
inline array<T> read(const std::string &filename)
{
    std::ifstream f(filename.c_str(), std::ios::binary);
    if (!f)
        throw std::runtime_error("npy: couldn't open " + filename + " for reading");

    char magic[8];
    f.read(magic, 8);
    if (!f || memcmp(magic, "\x93NUMPY", 6))
        throw std::runtime_error("npy: " + filename + " is not a .npy file");
    if (magic[6] != 1)
        throw std::runtime_error("npy: " + filename + " is not format version 1.x");

    uint8_t lo, hi;
    f.read(reinterpret_cast<char *> (&lo), 1);
    f.read(reinterpret_cast<char *> (&hi), 1);

    std::vector<char> hbuf(size_t(lo) | (size_t(hi) << 8));
    f.read(&hbuf[0], hbuf.size());
    if (!f)
        throw std::runtime_error("npy: " + filename + " ended inside its header");

    std::string hdr(hbuf.begin(), hbuf.end());

    std::string descr = _find_value(hdr, "descr");
    if ((descr.size() < 2) || (descr.substr(1, descr.size()-2) != traits<T>::descr()))
        throw std::runtime_error("npy: " + filename + " has dtype " + descr
                                 + ", but this driver asked for '" + traits<T>::descr() + "'");

    if (_find_value(hdr, "fortran_order") != "False")
        throw std::runtime_error("npy: " + filename + " is fortran-ordered, which is not supported");

    array<T> a;
    a.shape = _parse_shape(_find_value(hdr, "shape"));

    size_t n = 1;
    for (size_t i = 0; i < a.shape.size(); i++)
        n *= a.shape[i];

    a.data.resize(n);
    if (n > 0)
        f.read(reinterpret_cast<char *> (&a.data[0]), n * sizeof(T));
    if (!f)
        throw std::runtime_error("npy: " + filename + " ended before its data did");

    return a;
}


template<typename T>
inline void write(const std::string &filename, const std::vector<size_t> &shape, const T *data)
{
    std::stringstream ss;
    ss << "{'descr': '" << traits<T>::descr() << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++)
        ss << shape[i] << ((shape.size() == 1) ? "," : (i+1 < shape.size() ? ", " : ""));
    ss << "), }";

    std::string hdr = ss.str();

    // numpy wants (10 + header length) to be a multiple of 64, with the header
    // space-padded and newline-terminated.
    size_t pad = 64 - ((10 + hdr.size() + 1) % 64);
    if (pad != 64)
        hdr += std::string(pad, ' ');
    hdr += "\n";

    std::ofstream f(filename.c_str(), std::ios::binary);
    if (!f)
        throw std::runtime_error("npy: couldn't open " + filename + " for writing");

    uint8_t len[2] = { uint8_t(hdr.size() & 0xff), uint8_t((hdr.size() >> 8) & 0xff) };
    f.write("\x93NUMPY\x01\x00", 8);
    f.write(reinterpret_cast<char *> (len), 2);
    f.write(hdr.data(), hdr.size());

    size_t n = 1;
    for (size_t i = 0; i < shape.size(); i++)
        n *= shape[i];
    if (n > 0)
        f.write(reinterpret_cast<const char *> (data), n * sizeof(T));

    f.close();
    if (!f)
        throw std::runtime_error("npy: error writing " + filename);
}


template<typename T>
inline void write(const std::string &filename, const std::vector<T> &data)
{
    write<T> (filename, std::vector<size_t> { data.size() }, data.empty() ? nullptr : &data[0]);
}


// -------------------------------------------------------------------------------------------------
//
// The command line.  A driver is invoked as
//     driver <in_1.npy> ... <in_m.npy> <out_1.npy> ... <out_n.npy> [key=value ...]
// with m input files and n output files, and reads the key=value pairs through 'params'.
// Most drivers have m = n = 1: they read argv[1], write argv[2], and construct 'params'
// directly.  A driver with more files uses 'cmdline' below, which also checks the count.


struct params {
    std::vector<std::pair<std::string, std::string>> kv;

    params(int argc, char **argv, int first = 3)
    {
        for (int i = first; i < argc; i++) {
            std::string s = argv[i];
            size_t j = s.find('=');
            if (j == std::string::npos)
                throw std::runtime_error("expected key=value, got: " + s);
            kv.push_back({ s.substr(0,j), s.substr(j+1) });
        }
    }

    std::string get(const std::string &key, const std::string &fallback) const
    {
        for (size_t i = 0; i < kv.size(); i++)
            if (kv[i].first == key)
                return kv[i].second;
        return fallback;
    }

    double get_double(const std::string &key, double fallback) const
    {
        std::string s = get(key, "");
        return s.empty() ? fallback : strtod(s.c_str(), nullptr);
    }

    long get_long(const std::string &key, long fallback) const
    {
        std::string s = get(key, "");
        return s.empty() ? fallback : strtol(s.c_str(), nullptr, 10);
    }
};


// The whole command line of a driver with 'm' input files and 'n' output files.
//
// Throws unless exactly m + n arguments come before the first key=value one.  Nothing on
// the command line marks where the inputs stop and the outputs start, so without this
// check a test that passed the wrong number of arrays would have the driver read the
// wrong file, or overwrite one of its inputs.
struct cmdline {
    std::vector<std::string> inputs;    // m paths
    std::vector<std::string> outputs;   // n paths
    params kv;                          // the key=value arguments

    cmdline(int argc, char **argv, int m, int n) :
        kv(argc, argv, _checked_nfiles(argc, argv, m, n) + 1)
    {
        for (int i = 0; i < m; i++)
            inputs.push_back(argv[1 + i]);
        for (int i = 0; i < n; i++)
            outputs.push_back(argv[1 + m + i]);
    }

    // Returns m + n, after checking that this is the number of file arguments.
    static int _checked_nfiles(int argc, char **argv, int m, int n)
    {
        int nfiles = 0;
        while ((1 + nfiles < argc) && !strchr(argv[1 + nfiles], '='))
            nfiles++;

        if (nfiles != m + n) {
            std::stringstream ss;
            ss << "expected " << m << " input and " << n << " output file(s) before the"
               << " key=value arguments, got " << nfiles;
            throw std::runtime_error(ss.str());
        }

        return m + n;
    }
};


}  // namespace npy

#endif  // _CHIMEFRB_NPY_HPP
