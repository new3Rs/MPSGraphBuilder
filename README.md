# MPSGraphBuilder
MPSGraphBuilder builds an MPSGraph instance from a .mlmodel format file.

# WARNING
Many layers are not tested, just implemented.

# How to build
```sh
brew install swift-protobuf
make preprocess
swift build
```

# Knowhow
- If you call "run" method of MPSGraph repeatedly, you need wrap the proper part with autoreleasepool even in Swift.