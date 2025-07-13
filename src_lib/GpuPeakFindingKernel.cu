#include "../include/pirate/PeakFindingKernel.hpp"

#include <mutex>
#include <ksgpu/Array.hpp>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Kernel registry


static mutex pf_kernel_lock;
pf_kernel *pf_kernel_registry = nullptr;


void pf_kernel::register_kernel()
{
    // Just check that all members have been initialized.
    // (In the future, I may add more argument checking here.)
    
    xassert(M > 0);
    xassert(E > 0);
    xassert(Dout > 0);
    xassert(Dcore > 0);
    xassert(W > 0);
    xassert(P > 0);
    xassert((E == 0) || (RW > 0));
    xassert(full_kernel != nullptr);
    xassert(reduce_only_kernel != nullptr);
    xassert(next == nullptr);
    
    unique_lock<mutex> lk(pf_kernel_lock);

    // Check whether this (M,E,Dout) triple has been registered before.
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->M != M) || (k->E != E) || (k->Dout != Dout))
	    continue;

	if (k->debug && !this->debug) {
	    cout << "Note: debug and non-debug pf kernels were registered; debug kernel takes priority" << endl;
	    return;
	}

	if (!k->debug && this->debug) {
	    cout << "Note: debug and non-debug pf kernels were registered; debug kernel takes priority" << endl;
	    pf_kernel *save = k->next;
	    *k = *this;
	    k->next = save;
	    return;
	}

	stringstream ss;
	ss << "pf_kernel::register() called twice with (M,E,Dout)=" << "(" << M << "," << E << "," << Dout << ")";
	throw runtime_error(ss.str());
    }
    
    // Memory leak is okay for registry.
    pf_kernel *pf_copy = new pf_kernel(*this);
    pf_copy->next = pf_kernel_registry;  // assign with lock held
    pf_kernel_registry = pf_copy;
}


// Static member function
pf_kernel pf_kernel::get(int M, int E, int Dout)
{
    unique_lock<mutex> lk(pf_kernel_lock);
    
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->M == M) && (k->E == E) && (k->Dout == Dout)) {
	    pf_kernel ret = *k;
	    ret.next = nullptr;
	    return ret;
	}
    }

    stringstream ss;
    ss << "pf_kernel::get(): no kernel found for (M,E,Dout)="
       << "(" << M << "," << E << "," << Dout << ")";
    throw runtime_error(ss.str());
}


// Static member function
vector<pf_kernel> pf_kernel::enumerate()
{
    vector<pf_kernel> ret;
    unique_lock<mutex> lk(pf_kernel_lock);

    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next)
	ret.push_back(*k);

    lk.unlock();

    for (pf_kernel &k: ret)
	k.next = nullptr;

    return ret;
}


}  // namespace pirate
