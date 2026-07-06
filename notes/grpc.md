# GRPC .proto guidelines

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