# MPSGraphBuilder
MPSGraphBuilder builds an MPSGraph instance from a .mlmodel format file.

# How to build
```sh
make preprocess
swift build
```

# Knowhow
- If you call "run" method of MPSGraph repeatedly, you need wrap the proper part with autoreleasepool even in Swift.