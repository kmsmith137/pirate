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
    bool zerocopy = false;     // set by set_zerocopy(), supplies MSG_ZEROCOPY on future calls to send().
    bool connreset = false;    // set by send() if receiver closes connection, see below.
    bool nonblocking = false;  // set by set_nonblocking(), modifies behavior of read(), send().

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
    // Note: send() also returns zero if socket is nonblocking, and is full (e.g. peer reading too slowly).
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
// Epoll: RAII wrapper for Linux epoll (I/O event notification for multiple file descriptors).
//
// Quick refresher on epoll:
//
//   - epoll is Linux's scalable I/O multiplexing mechanism (like select/poll but O(1) not O(n)).
//   - An epoll instance monitors a set of fds, and reports which fds are "ready" for I/O.
//   - For each fd, caller specifies events of interest (EPOLLIN, EPOLLOUT, etc.)
//   - For each fd, caller also specifies opaque "context" (e.g. pointer to connection object).
//   - When wait() returns, caller iterates over ready fds, using context to dispatch I/O.
//
// Basic usage:
//
//   Epoll ep;
//   struct epoll_event ev;
//   ev.events = EPOLLIN;              // interested in "ready to read"
//   ev.data.ptr = my_connection_ptr;  // opaque context, "replayed" by wait()
//   ep.add_fd(socket_fd, ev);
//
//   while (...) {
//       int n = ep.wait();  // blocks until event (or use timeout)
//       for (int i = 0; i < n; i++) {
//           MyConnection *c = (MyConnection *) ep.events[i].data.ptr;
//           if (ep.events[i].events & EPOLLIN)
//               c->handle_read();
//       }
//   }
//
// The 'struct epoll_event' (passed to add_fd, populated by wait) looks like:
//
//   struct epoll_event {
//       uint32_t events;   // bitmask: requested events (add_fd) or occurred events (wait)
//       union {            // opaque context, specified by caller, replayed by wait()
//           void     *ptr;
//           int       fd;
//           uint32_t  u32;
//           uint64_t  u64;
//       } data;
//   };
//
// Common event flags (see 'man epoll' for full list):
//
//   EPOLLIN      fd is ready for read()
//   EPOLLOUT     fd is ready for write()
//   EPOLLRDHUP   peer closed connection (or shutdown write half)
//   EPOLLHUP     hangup (always monitored, even if not requested)
//   EPOLLERR     error condition (always monitored, even if not requested)
//   EPOLLET      edge-triggered mode (default is level-triggered)


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
    void delete_fd(int fd);

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

