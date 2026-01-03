#include "../include/pirate/network_utils.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <sstream>
#include <iostream>
#include <stdexcept>

#include <ksgpu/xassert.hpp>  // note: defines _unlikely()

using namespace std;


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


inline string errstr(const string &func_name)
{
    stringstream ss;
    ss << func_name << "() failed: " << strerror(errno);
    return ss.str();
}

inline string errstr(int fd, const string &func_name)
{
    if (fd < 0) {
        stringstream ss;
        ss << func_name << "() called on uninitialized object";
        return ss.str();
    }

    return errstr(func_name);
}

static void inet_pton_x(struct sockaddr_in &saddr, const string &ip_addr, uint16_t port)
{
    memset(&saddr, 0, sizeof(saddr));

    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(port);   // note htons() here!
    
    int err = inet_pton(AF_INET, ip_addr.c_str(), &saddr.sin_addr);
    
    if (err < 0) {
        stringstream ss;
        ss << "inet_pton() failed: " << strerror(errno);
        throw runtime_error(ss.str());
    }

    if (err == 0) {
        stringstream ss;
        ss << "invalid IPv4 address: '" << ip_addr << "'";
        throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------


Socket::Socket(int domain, int type, int protocol)
{
    this->fd = socket(domain, type, protocol);

    if (_unlikely(fd < 0))
        throw runtime_error(errstr("socket"));
}


void Socket::connect(const std::string &ip_addr, uint16_t port)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::connect() called on uninitialized socket");

    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);

    int err = ::connect(this->fd, (const struct sockaddr *) &saddr, sizeof(saddr));

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::connect"));
}


void Socket::bind(const std::string &ip_addr, uint16_t port)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::bind() called on uninitialized socket");

    struct sockaddr_in saddr;
    inet_pton_x(saddr, ip_addr, port);

    int err = ::bind(this->fd, (const struct sockaddr *) &saddr, sizeof(saddr));
    
    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::bind"));
}


void Socket::listen(int backlog)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::listen() called on uninitialized socket");

    int err = ::listen(fd, backlog);

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::listen"));
}


void Socket::close()
{
    if (fd < 0)
        return;

    int err = ::close(fd);
    
    this->fd = -1;
    this->zerocopy = false;
    this->connreset = false;

    if (_unlikely(err < 0))
        cout << errstr("Socket::close") << endl;
}

    
long Socket::read(void *buf, long count)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::read() called on uninitialized socket");

    xassert(count > 0);
    long nbytes = ::read(this->fd, buf, count);

    if (_unlikely(nbytes < 0))
        throw runtime_error(errstr(fd, "Socket::read"));

    xassert(nbytes <= count);
    return nbytes;
}


long Socket::send(const void *buf, long count, int flags)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::send() called on uninitialized socket");

    xassert(count > 0);

    if (_unlikely(connreset))
        throw runtime_error("Socket::send() called after connection was reset");
    
    if (zerocopy)
        flags |= MSG_ZEROCOPY;
    
    long nbytes = ::send(this->fd, buf, count, flags);

    if (nbytes < 0) {
        // If receiver closes connection, then send() returns zero and sets Socket::connreset = true.
        // If send() is called subsequently (with Socket::connreset == true), then an exception is thrown.
        // This provides a mechanism for the sender to detect a closed connection.

        if ((errno == ECONNRESET || errno == EPIPE) && !connreset) {
            connreset = true;
            errno = 0;
            return 0;
        }

        throw runtime_error(errstr(fd, "Socket::send"));
    }
    
    // Can send() return zero? If so, then this next line needs removal or rethinking.
    if (_unlikely(nbytes == 0))
        throw runtime_error("send() returned zero?!");

    xassert(nbytes <= count);
    return nbytes;
}


Socket Socket::accept()
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::accept() called on uninitialized socket");

    // FIXME currently throwing away sender's IP address
    sockaddr_in saddr_throwaway;
    socklen_t saddr_len = sizeof(saddr_throwaway);

    Socket ret;
    ret.fd = ::accept(fd, (struct sockaddr *) &saddr_throwaway, &saddr_len);

    if (_unlikely(ret.fd < 0))
        throw runtime_error(errstr(fd, "Socket::accept"));

    return ret;
}


void Socket::getopt(int level, int optname, void *optval, socklen_t *optlen)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::getopt() called on uninitialized socket");

    xassert(optval != nullptr);
    xassert(optlen != nullptr);

    int err = getsockopt(fd, level, optname, optval, optlen);
    
    if (_unlikely((err < 0)))
        throw runtime_error(errstr(fd, "getsockopt"));
}


void Socket::setopt(int level, int optname, const void *optval, socklen_t optlen)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::setopt() called on uninitialized socket");

    xassert(optval != nullptr);
        
    int err = setsockopt(fd, level, optname, optval, optlen);

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "setsockopt"));
}


void Socket::set_reuseaddr()
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::set_reuseaddr() called on uninitialized socket");

    int on = 1;
    int err = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::set_reuseaddr"));
}


void Socket::set_nonblocking()
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::set_nonblocking() called on uninitialized socket");

    int flags = fcntl(this->fd, F_GETFL);
    
    if (_unlikely(flags < 0))
        throw runtime_error(errstr(fd, "Socket::set_nonblocking: F_GETFL fcntl"));

    int err = fcntl(this->fd, F_SETFL, flags | O_NONBLOCK);

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::set_nonblocking: F_SETFL fcntl"));
}


void Socket::set_pacing_rate(double bytes_per_sec)
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::set_pacing_rate() called on uninitialized socket");

    xassert(bytes_per_sec >= 1.0);

    if (_unlikely(bytes_per_sec > 4.0e9)) {
        stringstream ss;
        ss << "Socket::set_pacing_rate(" << bytes_per_sec << "):"
           << " 'bytes_per_sec' values larger than 4e9 (i.e. 36 Gpbs) are not currently supported!"
           << " This is because setsockopt(SOL_SOCKET, SO_MAX_PACING_RATE) takes a uint32 argument."
           << " Suggested workaround: split output across multiple sockets/threads.";
        throw runtime_error(ss.str());
    }

    uint32_t b = uint32_t(bytes_per_sec + 0.5);
    int err = setsockopt(fd, SOL_SOCKET, SO_MAX_PACING_RATE, &b, sizeof(b));

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::set_pacing_rate"));
}
 

void Socket::set_zerocopy()
{
    if (_unlikely(fd < 0))
        throw runtime_error("Socket::set_zerocopy() called on uninitialized socket");

    int on = 1;
    int err = setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &on, sizeof(on));

    if (_unlikely(err < 0))
        throw runtime_error(errstr(fd, "Socket::set_zerocopy"));

    // If the 'zerocopy' flag is set, then MSG_ZEROCOPY will be included in future calls to send().
    this->zerocopy = true;
}


// Move constructor.
Socket::Socket(Socket &&s)
    : fd(s.fd), zerocopy(s.zerocopy), connreset(s.connreset)
{
    s.fd = -1;
    s.zerocopy = false;
    s.connreset = false;
}


// Move assignment-operator.
Socket &Socket::operator=(Socket &&s)
{
    this->close();
    this->fd = s.fd;
    this->zerocopy = s.zerocopy;
    this->connreset = s.connreset;
    
    s.fd = -1;
    s.zerocopy = false;
    s.connreset = false;
    return *this;
}


// -------------------------------------------------------------------------------------------------


Epoll::Epoll(bool init_flag, bool close_on_exec)
{
    if (init_flag)
        this->initialize(close_on_exec);
}


void Epoll::initialize(bool close_on_exec)
{
    if (_unlikely(epfd >= 0))
        throw runtime_error("Epoll::initialize() called on already-initialized Epoll instance");

    int flags = close_on_exec ? EPOLL_CLOEXEC : 0;
    this->epfd = epoll_create1(flags);

    if (_unlikely(epfd < 0))
        throw runtime_error(errstr("epoll_create"));
}


void Epoll::close()
{
    if (epfd < 0)
        return;

    int err = ::close(epfd);
    this->epfd = -1;

    if (_unlikely(err < 0))
        cout << errstr("Epoll::close") << endl;
}


void Epoll::add_fd(int fd, struct epoll_event &ev)
{
    if (_unlikely(epfd < 0))
        throw runtime_error("Epoll::add_fd() called on uninitialized Epoll instance");

    int err = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);

    if (_unlikely(err < 0))
        throw runtime_error(errstr(epfd, "Epoll::add_fd"));

    struct epoll_event ev0;
    memset(&ev0, 0, sizeof(ev0));
    this->events.push_back(ev0);
}


int Epoll::wait(int timeout_ms)
{
    if (_unlikely(epfd < 0))
        throw runtime_error("Epoll::wait() called on uninitialized Epoll instance");

    if (_unlikely(events.size() == 0))
        throw runtime_error("Epoll::wait() was called before Epoll::add_fd()");

    int ret = epoll_wait(epfd, &events[0], events.size(), timeout_ms);

    if (_unlikely(ret < 0))
        throw runtime_error(errstr(epfd, "Epoll::wait"));

    return ret;
}


} // namespace pirate
