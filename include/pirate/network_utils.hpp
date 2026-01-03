#ifndef _PIRATE_NETWORK_UTILS_HPP
#define _PIRATE_NETWORK_UTILS_HPP

#include <string>
#include <vector>
#include <sys/socket.h>  // socklen_t
#include <sys/epoll.h>

namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Socket: RAII wrapper for network socket.
//
// Socketio "cheat sheet"
//
// Sending TCP data:
//
//   Socket s(PF_INET, SOCK_STREAM);
//   s.connect("127.0.0.0", 1370);      // (dst IP address, port)
//   s.get_zerocopy();                  // optional
//   s.set_pacing_rate(bytes_per_sec);  // optional
//   long nbytes_sent = s.send(buf, maxbytes);
//
// Receiving TCP data from single connection:
//
//   Socket s(PF_INET, SOCK_STREAM);
//   s.set_reuseaddr();
//   s.bind("127.0.0.0", 1370);   // (dst IP address, port)
//   s.listen();
//
//   Socket sd = s.accept();
//   long nbytes_received = s.read(buf, maxbytes);


struct Socket
{
    int fd = -1;
    bool zerocopy = false;   // set by set_zerocopy(), supplies MSG_ZEROCOPY on future calls to send().
    bool connreset = false;  // set by send() if 

    // For TCP, use (domain,type) = (PF_INET,SOCK_STREAM). See "cheat sheet" above.
    Socket(int domain, int type, int protocol=0);
    Socket() { }
    
    ~Socket() { this->close(); }
    
    void connect(const std::string &ip_addr, uint16_t port);
    void bind(const std::string &ip_addr, uint16_t port);
    void listen(int backlog=128);
    void close();

    // Reminder: read() returns zero if connection ended, or if socket is nonblocking and no data is ready.
    long read(void *buf, long maxbytes);

    // If receiver closes connection, then send() returns zero and sets Socket::connreset = true.
    // If send() is called subsequently (with Socket::connreset == true), then an exception is thrown.
    // This provides a mechanism for the sender to detect a closed connection.
    long send(const void *buf, long count, int flags=0);

    // FIXME in current API, sender's IP address is thrown away!
    Socket accept();

    // General wrappers for getsockopt(), setsockopt()
    void getopt(int level, int optname, void *optval, socklen_t *optlen);
    void setopt(int level, int optname, const void *optval, socklen_t optlen);

    // Specific options
    void set_reuseaddr();    // setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    void set_nonblocking();  // fcntl(O_NONBLOCK)
    void set_pacing_rate(double bytes_per_sec);  // setsockopt(SOL_SOCKET, SO_MAX_PACING_RATE)
    void set_zerocopy();     // setsockopt(SOL_SOCKET, SO_ZEROCOPY) + (MSG_ZEROCOPY on future send() calls)
    
    // Socket is noncopyable but moveable (can always do shared_ptr<Socket> to avoid copies).
    
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;
    
    Socket(Socket &&s);
    Socket &operator=(Socket &&s);
};


// -------------------------------------------------------------------------------------------------
//
// Epoll: RAII wrapper for epoll file descriptor.
//
// Reminder: 'struct epoll_event' looks like this
//
//   struct epoll_event {
//      uint32_t events;        // bitmask, see below
//      union {                 // union, not struct!!
//           void        *ptr;
//           int          fd;
//           uint32_t     u32;
//           uint64_t     u64;
//      } data;
//   };
//
// Here is a partial list of bits in epoll_event::events (see 'man epoll' for complete list):
//
//   EPOLLIN         fd is ready for read()
//   EPOLLOUT        fd is ready for write()
//   EPOLLRDHUP      peer closed connection, or shut down writing half of connection
//   EPOLLHUP        "hangup", i.e. peer closed connection (what is difference versus EPOLLRDHUP)?
//   EPOLLPRI        "exceptional condition" on fd (see POLLPRI in 'man poll')
//   EPOLLERR        fd has error (also reported for the write end of a pipe when the read end has been closed)
//   EPOLLET         requests edge-triggered notification (see 'man epoll')
//   EPOLLONESHOT    requests one-shot notification (see 'man epoll')
//   EPOLLWAKEUP     ensure system does not "suspend" or "hibernate" while event is being processed (see 'man epoll')
//   EPOLLEXCLUSIVE  sets an exclusive wakeup mode for the epoll file descriptor (see 'man epoll')
//
// Epoll always waits on (EPOLLERR | EPOLLHUP), i.e. no need to include these flags in Epoll::add_fd().


struct Epoll
{
    int epfd = -1;

    // Events returned by wait() are stored here.
    // Note that wait() returns the number of events (which is <= events.size()).
    std::vector<epoll_event> events;

    // If constructor is called with initialize=false, then Epoll::initialize() must be called later.
    Epoll(bool initialize=true, bool close_on_exec=false);
    ~Epoll() { this->close(); }

    void add_fd(int fd, struct epoll_event &ev);
    // To add later: modify_fd(fd,ev), delete_fd(fd).

    // Returns number of events (or zero, if timeout expires).
    // Negative timeout means "blocking". Zero timeout is nonblocking.
    int wait(int timeout_ms=-1);

    void initialize(bool close_on_exec=false);
    void close();
        
    // The Epoll class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<Epoll> ep = make_shared<Epoll> ();
    
    Epoll(const Epoll &) = delete;
    Epoll &operator=(const Epoll &) = delete;
};


}  // namespace pirate

#endif // _PIRATE_NETWORK_UTILS_HPP

