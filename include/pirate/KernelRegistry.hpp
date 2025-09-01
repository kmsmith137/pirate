#ifndef _PIRATE_KERNEL_REGISTRY_HPP
#define _PIRATE_KERNEL_REGISTRY_HPP

#include <mutex>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <ksgpu/rand_utils.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// KernelRegistry is header-only (i.e. there is no source file KernelRegistry.cu).
//
// The following functions should be defined:
//
//   bool operator==(const Key &k1, const Key &k2);
//   ostream &operator<<(ostream &os, const Key &k);

template<class Key, class Val>
struct KernelRegistry
{
    struct Entry
    {
	Key key;
	Val val;
	bool debug = false;        // debug kernel "clobbers" non-debug kernel
	bool initialized = false;  // has deferred_initialization() been called?
    };
    
    mutable std::mutex lock;
    std::vector<Entry> entries;
    
    // deferred_initialization(): this optional method performs initializations which
    // are "deferred" from when the kernel is registered, to when the kernel is first
    // used. For an example, see src_lib/GpuDedispersionKernel.cu.
    //
    // It's important to defer calls to the cuda runtime library (e.g. setting shared
    // memory size for a kernel). Such calls may misbehave at library initialization
    // time, but behave normally if "deferred". Warning: this can be hard to debug!
    // Sometimes, the non-deferred call appears to succeed, but an unrelated kernel
    // launch will fail later with error 400 ("invalid resource handle").
    
    virtual void deferred_initialization(Val &val) { }

    
    KernelRegistry() { }

    // Call with lock held!
    // Returned pointer is only valid until lock is dropped.
    Entry *_get_locked(const Key &key)
    {
	for (Entry &e: entries)
	    if (e.key == key)
		return &e;
	return nullptr;
    }

    
    void add(const Key &key, const Val &val, bool debug)
    {
	Entry enew;
	enew.key = key;
	enew.val = val;
	enew.debug = debug;
	
	std::unique_lock<std::mutex> lk(this->lock);
	Entry *e = this->_get_locked(key);

	if (!e) {
	    this->entries.push_back(enew);
	    return;
	}

	if (!e->debug && debug)
	    *e = enew;   // clobber and fall through
	else if (!e->debug || debug) {
	    std::stringstream ss;
	    ss << "KernelRegistry::add() called twice, perhaps you forgot to set the 'debug' flag? Kernel is: " << key;
	    throw std::runtime_error(ss.str());
	}

	// If we get here, then both a debug and non-debug kernel were registered.
	// This is not an error, but we print an informational message.

	std::cout << "\nNote: debug and non-debug kernels were registered; debug kernel takes priority"
		  << "\nKernel is: " << key << "\n\n";
    }


    Val query(const Key &key)
    {
	std::unique_lock<std::mutex> lk(this->lock);
	Entry *e = this->_get_locked(key);

	if (!e) {
	    lk.unlock();
	    std::stringstream ss;
	    ss << "Kernel not found in registry: " << key;
	    throw std::runtime_error(ss.str());
	}

	if (!e->initialized) {
	    this->deferred_initialization(e->val);
	    e->initialized = true;
	}
	
	// Not sure whether "return e->val" drops lock before copying.
	Val ret = e->val;
	return ret;
    }


    Key get_random_key()
    {
	std::unique_lock<std::mutex> lk(this->lock);
	
	if (entries.size() == 0)
	    throw std::runtime_error("KernelRegistry::get_random() called on empty registry");
	    
	long i = ksgpu::rand_int(0, entries.size());

	// Not sure whether "return dd_kernel_registry[i]" drops lock before copying.
	Key ret = entries[i].key;
	return ret;
    }
};


}  // namespace pirate

#endif // _PIRATE_KERNEL_REGISTRY_HPP
