# GRPC guidelines

## Creating a grpc server

Checklist for ANY new grpc server, C++ or python (all three existing servers --
FrbServer's RPC service, FrbGrouper, `pirate_frb run toy_sifter` -- follow it):

1. **Disable SO_REUSEPORT.** grpc's default on Linux is ON, which lets two grpc
   servers bind the SAME port: both binds "succeed", and the kernel silently
   load-balances incoming connections between the two servers. Real-world
   scenario: an orphaned old server survives a restart, the new server starts
   "successfully", and RPCs randomly reach the dying process -- intermittent
   failures with no loud error anywhere. Disabling makes port conflicts fail
   loudly at startup instead.

   ```cpp
   builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);   // C++
   ```
   ```py
   grpc.server(..., options=[('grpc.so_reuseport', 0)])       # python
   ```

2. **Check for bind failure explicitly.** In C++, the 2-arg AddListeningPort
   does NOT report bind failures (BuildAndStart() still returns a non-null
   server that silently isn't listening); use the 3-arg overload and treat
   selected_port == 0 as an error. In python, check add_insecure_port() == 0
   (recent grpcio also raises on failure -- keep the check anyway). Note step 1
   is what makes this check effective: with SO_REUSEPORT on, a conflicting bind
   "succeeds" and the check passes.

## .proto guidelines

- When editing .proto files, rewrite gRPC field numbers in order to keep them
  "canoncial": the first member in a message is 1, the second member is 2, etc.
  
  Don't worry about backwards-compatibility -- we attach a version number to
  each .proto file, and throw an exception if the client/server are using
  different versions.

- When you edit a .proto file (unless it's a comments-only edit), always ask
  explicitly whether to increment the version number.

  Generally speaking, we want to increment the version number when the .proto
  file is modified. However, when implementing a new feature, we often make
  many small edits to a .proto file, while the new feature is on a development
  branch "far from production". In this case, I'd prefer not to increment the
  version number each time. It won't be clear from context which case applies,
  so it's best to ask explicitly.

- Every unary RPC should include the version number (but no need for the
  version number in the response). If an RPC creates a TCP stream, then
  the RPC should contain a version number, but subsequent messages on the
  stream do not need a version number.