#ifndef _PIRATE_YAML_FILE_HPP
#define _PIRATE_YAML_FILE_HPP

#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

#include <ksgpu/string_utils.hpp>  // tuple_str(), type_name()


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// YamlFile: thin wrapper around YAML::Node, defining an alternate interface
// for reading files.
//
// Compared to YAML::Node, this alternate interface is
//  - much less general
//  - much slower
//  - has more error-checking
//  - throws "verbose" exceptions to help trace parse errors
//
// This alternate interface is useful for reading small, simple config files,
// where quality of error-checking is the primary consideration.
//
//
// Example syntax: if the file contents are as follows:
//
//   x: 15
//   y: [ 1, 2, 3 ]
//
// then the file could be parsed as follows:
//
//   f = YamlFile(filename);
//
//   int x = f.get_scalar<int>("x");
//   vector<int> y = f.get_vector<int>("y");
//
//   f.assert_all_keys_requested();
//
// FIXME to do: implement YamlFile::assert_all_keys_requested().


struct YamlFile {
    // Allowed values: { Type::Undefined, Type::Null, Type::Scalar, Type::Sequence, Type::Map }
    using Type = YAML::NodeType::value;
    
    YamlFile(const std::string &filename);
    YamlFile(const std::string &name, const YAML::Node &node);
    
    std::string name;
    YAML::Node node;

    mutable std::unordered_set<std::string> requested_keys;

    // Throws exception if (type() != Map), or if key 'k' is absent.
    YamlFile operator[](const std::string &k) const;

    // Throws exception if (type() != Sequence), or if index 'ix' is out of range.
    YamlFile operator[](long ix) const;

    long size() const { return node.size(); }
    Type type() const { return node.Type(); }

    template<typename T> inline T as_scalar() const;
    template<typename T> inline std::vector<T> as_vector() const;
                                            
    template<typename T> inline T get_scalar(const std::string &k) const;
    template<typename T> inline T get_scalar(const std::string &k, const T &default_value) const;
    
    template<typename T> inline std::vector<T> get_vector(const std::string &k) const;
    template<typename T> inline std::vector<T> get_vector(const std::string &k, const std::vector<T> &default_value) const;

    // If these asserts fail, they will throw a "verbose" exception, intended to facilitate error-checking.
    void assert_type_is(Type type) const;
    void check_for_invalid_keys() const;

    // Similar to operator[](const string&), but does not throw an exception if key 'k' is absent.
    // (An exception is still thrown if type() != Map.)
    YamlFile _get_child(const std::string &k) const;

    // Intended for debugging
    static std::string type_str(Type t);
};


// -------------------------------------------------------------------------------------------------


template<typename T>
inline bool _yaml_node_to_scalar(const YAML::Node &node, T &val)
{
    try {
        val = node.as<T>();
        return true;
    }
    catch (...) {
        return false;
    }
}


template<typename T>
inline T YamlFile::as_scalar() const
{
    this->assert_type_is(Type::Scalar);
    
    T ret;
    if (!_yaml_node_to_scalar<T> (node, ret))
        throw std::runtime_error(name + ": parse failure (expected type " + ksgpu::type_name<T>() + ")");

    return ret;
}


template<typename T>
inline std::vector<T> YamlFile::as_vector() const
{
    this->assert_type_is(Type::Sequence);

    long n = size();
    std::vector<T> ret(n);

    for (long i = 0; i < n; i++) {
        YAML::Node child_node = node[i];

        if (!child_node.IsScalar() || !_yaml_node_to_scalar<T> (child_node, ret[i]))
            throw std::runtime_error(name + ": parse failure (expected type vector<" + ksgpu::type_name<T>() + ">)");
    }

    return ret;
}


template<typename T>
inline T YamlFile::get_scalar(const std::string &k) const
{
    return (*this)[k].as_scalar<T> ();
}


template<typename T>
inline T YamlFile::get_scalar(const std::string &k, const T &default_value) const
{
    YamlFile child = _get_child(k);

    if (child.node)
        return child.as_scalar<T> ();

    return default_value;
}


template<typename T>
inline std::vector<T> YamlFile::get_vector(const std::string &k) const
{
    return (*this)[k].as_vector<T> ();
}


template<typename T>
inline std::vector<T> YamlFile::get_vector(const std::string &k, const std::vector<T> &default_value) const
{
    YamlFile child = _get_child(k);

    if (child.node)
        return child.as_vector<T> ();

    return default_value;
}


}  // namespace pirate

#endif //  _PIRATE_YAML_FILE_HPP
