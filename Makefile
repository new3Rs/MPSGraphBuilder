FORMAT_DIR=Sources/MPSGraphBuilder/format

preprocess:
	if [ ! -d "coremltools" ] ; then git clone  --depth 1 --branch 5.0 https://github.com/apple/coremltools.git; fi
	mkdir -p ${FORMAT_DIR}
	cd coremltools/mlmodel/format; protoc --swift_out=../../../${FORMAT_DIR} *.proto