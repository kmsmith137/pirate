# YAML Input/Output Patterns

This document summarizes patterns for implementing `to_yaml()` and `from_yaml()` methods in C++ classes, based on the implementation in `DedispersionConfig`.

## Reading YAML with YamlFile

The `YamlFile` class (in `pirate/YamlFile.hpp`) wraps yaml-cpp with better error checking and verbose exceptions.

### Basic Usage

```cpp
#include "../include/pirate/YamlFile.hpp"

MyClass MyClass::from_yaml(const std::string &filename)
{
    YamlFile f(filename);
    return MyClass::from_yaml(f);
}

MyClass MyClass::from_yaml(const YamlFile &f)
{
    MyClass ret;
    
    // Required scalar
    ret.some_int = f.get_scalar<long>("some_int");
    
    // Optional scalar with default
    ret.optional_value = f.get_scalar<long>("optional_value", 0L);
    
    // Required vector
    ret.some_vector = f.get_vector<double>("some_vector");
    
    // Optional vector with default
    ret.optional_vec = f.get_vector<long>("optional_vec", {1, 2, 3});
    
    // Nested sequence of maps
    YamlFile items = f["items"];
    for (long i = 0; i < items.size(); i++) {
        YamlFile item = items[i];
        ret.items.push_back({
            item.get_scalar<long>("field1"),
            item.get_scalar<long>("field2", 0L)  // optional with default
        });
        item.check_for_invalid_keys();  // Important: check nested maps!
    }
    
    f.check_for_invalid_keys();  // Detect typos/unexpected keys
    ret.validate();
    return ret;
}
```

### Key Points for from_yaml()

- Use `get_scalar<T>(key, default)` for optional parameters
- Use `0L` not `0` when the type is `long` (ensures correct template deduction)
- Always call `check_for_invalid_keys()` on:
  - The root YamlFile
  - Any nested maps you access with `operator[]`
- Call `validate()` at the end to ensure the parsed data is valid


## Writing YAML with YAML::Emitter

### Header Declaration

```cpp
namespace YAML { class Emitter; }  // Forward declaration

struct MyClass {
    // If 'verbose' is true, include comments explaining each field.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    void to_yaml(const std::string &filename, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;
};
```

### Implementation

```cpp
#include <yaml-cpp/emitter.h>

void MyClass::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();
    
    emitter << YAML::BeginMap;
    
    // ---- Section with multi-line comment ----
    
    if (verbose) {
        // YAML::Newline before comment prevents horizontal staggering.
        // Double Newline creates a blank line before the section.
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Multi-line comment explaining this section.\n"
            "Second line of comment.\n"
            "Third line."
        ) << YAML::Newline;  // Newline after comment, before data
    }
    
    emitter << YAML::Key << "scalar_field" << YAML::Value << scalar_field;
    
    if (verbose)
        emitter << YAML::Newline;  // Blank line between fields
    
    // Inline sequence using YAML::Flow
    emitter << YAML::Key << "vector_field"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n : vector_field)
        emitter << n;
    emitter << YAML::EndSeq;
    
    // Sequence of maps (each map on one line)
    if (verbose) {
        emitter << YAML::Newline << YAML::Comment("Short inline comment.");
    }
    
    emitter << YAML::Key << "items"
            << YAML::Value
            << YAML::BeginSeq;
    
    for (const auto &item : items) {
        emitter
            << YAML::Flow  // Makes the map inline: {field1: x, field2: y}
            << YAML::BeginMap
            << YAML::Key << "field1" << YAML::Value << item.field1
            << YAML::Key << "field2" << YAML::Value << item.field2
            << YAML::EndMap;
    }
    
    emitter << YAML::EndSeq;
    
    emitter << YAML::EndMap;
}

std::string MyClass::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
}

void MyClass::to_yaml(const std::string &filename, bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    const char *s = emitter.c_str();
    
    File f(filename, O_WRONLY | O_CREAT | O_TRUNC);
    f.write(s, strlen(s));
}
```

### Key Points for to_yaml()

- Call `validate()` at the start to ensure data is valid before serializing
- Use `YAML::Flow` for inline sequences `[1, 2, 3]` and maps `{a: 1, b: 2}`
- Without `YAML::Flow`, sequences/maps are block-style (multi-line)

### Formatting with YAML::Newline

- `YAML::Newline` before a multi-line `YAML::Comment()` prevents the comment from appearing inline after the previous value
- `YAML::Newline` after a comment, before data, improves readability
- Double `YAML::Newline` creates a blank line (useful between major sections)
- Add `YAML::Newline` between fields in verbose mode for visual separation
- Without `YAML::Newline`, comments attach inline to the previous element


## pybind11 Wrapping

```cpp
#include "../include/pirate/MyClass.hpp"

py::class_<MyClass>(m, "MyClass")
    // Use static_cast to disambiguate overloaded static methods
    .def_static("from_yaml", 
        static_cast<MyClass (*)(const std::string &)>(&MyClass::from_yaml),
        py::arg("filename"))
    .def("to_yaml_string", &MyClass::to_yaml_string,
        py::arg("verbose") = false)
;
```

### Python Command-Line Interface

```python
def parse_show_myclass(subparsers):
    parser = subparsers.add_parser("show_myclass", 
        help="Parse a config file and write YAML to stdout")
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Include comments explaining the meaning of each field")

def show_myclass(args):
    config = pirate_pybind11.MyClass.from_yaml(args.config_file)
    print(config.to_yaml_string(args.verbose))
```


## Example Output

**Non-verbose:**
```yaml
zone_nfreq: [16384]
zone_freq_edges: [400, 800]
tree_rank: 15
items:
  - {field1: 1, field2: 2}
  - {field1: 3, field2: 4}
```

**Verbose:**
```yaml

# Frequency channels. The observed frequency band is divided into "zones".
# Within each zone, all frequency channels have the same width.
zone_nfreq: [16384]
zone_freq_edges: [400, 800]

# Core parameters.
# The number of "tree" channels is ntree = 2^tree_rank.
tree_rank: 15

# Items: list of {field1, field2} pairs.
items:
  - {field1: 1, field2: 2}
  - {field1: 3, field2: 4}
```

