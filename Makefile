FORMAT_DIR=Sources/MPSGraphBuilder/format

preprocess:
	if [ ! -d "coremltools" ] ; then git clone https://github.com/apple/coremltools.git; fi
	mkdir -p ${FORMAT_DIR}
	cd coremltools/mlmodel/format; protoc --swift_out=../../../${FORMAT_DIR} *.proto