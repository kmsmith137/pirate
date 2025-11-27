
class ChunkPool
{
public:
    const ssize_t nbytes_per_chunk;
    const ssize_t nchunks_max;
    
    struct Core;
    std::shared_ptr<Core> core;

    // FIXME ChunkPool constructor currently allocates synchronously.
    // Should allocate asynchronously, in separate thread.
    ChunkPool(ssize_t nbytes_per_chunk, ssize_t nchunks);

    // Returns smart possize_ter which returns chunk to pool when last reference is dropped.
    // FIXME currently throws exception if pool is empty -- should have better strategy.
    std::shared_ptr<char> get_chunk();

    ssize_t get_nchunks_free() const;

    // FIXME should add member function which "destroys" the ChunkPool:
    //   - future calls to get_chunk() fail
    //   - all chunks are freed (not returned to pool) when last reference is dropped.
};


// -------------------------------------------------------------------------------------------------


struct ChunkPool::Core
{
    const ssize_t nbytes_per_chunk;
    const ssize_t nchunks_max;
    ssize_t nchunks_free;
    mutex lock;
    
    vector<char *> chunks;

    
    ChunkPool::Core(ssize_t nbytes_per_chunk_, ssize_t nchunks_) :
        nbytes_per_chunk(nbytes_per_chunk_),
        nchunks_max(nchunks_),
        nchunks_free(nchunks_),
        chunks(nchunks_, nullptr)
    {
        for (ssize_t i = 0; i < nchunks; i++) {
            chunks[i] = malloc(xxx);
            if (!chunks[i])
                throw runtime_error(xxx);
        }
    }


    char *get_chunk()
    {
        lock_guard<mutex> lg(lock);
        if (nchunks_free <= 0)
            throw runtime_error();
        return chunks[--nchunks_free];
    }


    void put_chunk(char *p)
    {
        lock_guard<mutex> lg(lock);
        if (nchunks_free >= nchunks_max)
            throw runtime_error();
        chunks[nchunks_free++] = p;
    }

    
    ssize_t get_nchunks_free() const
    {
        lock_guard<mutex> lg(lock);
        return nchunks_free;
    }
};


// -------------------------------------------------------------------------------------------------


struct ChunkDeleter {
    shared_ptr<ChunkPool::Core> core;
    
    ChunkDeleter(const shared_ptr<ChunkPool::Core> &core_) : core(core_) { }

    void operator() (const char *p) { core->put_chunk(); }
};


// -------------------------------------------------------------------------------------------------


ChunkPool::ChunkPool(ssize_t nbytes_per_chunk_, ssize_t nchunks_) :
    nbytes_per_chunk(nbytes_per_chunk_),
    nchunks_max(nchunks_)
{
    core = make_shared<ChunkPool::Core> (nbytes_per_chunk_, nchunks_);
}


shared_ptr<char> ChunkPool::get_chunk()
{
    return shared_ptr<char> (core->get_chunk(), ChunkDeleter(core));
}

ssize_t ChunkPool::get_nchunks_free() const
{
    return core->get_nchunks_free();
}
