import Foundation
import Accelerate
import MetalPerformanceShadersGraph

enum ConvertError: Error {
    case notAvailable
    case wrongFormat
}

func convertType(_ type: CoreML_Specification_ArrayFeatureType.ArrayDataType) throws -> MPSDataType {
    switch type {
    case .invalidArrayDataType:
        return .invalid
    case .float32, .double: // converts double to float32 since MPSGraph does not support double.
        return .float32
    case .int32:
        return .int32
    default:
        throw ConvertError.notAvailable
    }
}

class MPSGraphBuilder {
    let graph = MPSGraph()
    let dataType: MPSDataType
    var tensors = [String:MPSGraphTensor]()
    var variableBatches = false

    init(dataType: MPSDataType) {
        self.dataType = dataType
    }

    private func convert(weights: CoreML_Specification_WeightParams, shape: [Int], doTranspose: Bool = false) throws -> MPSGraphTensor {
        let _shape = doTranspose ? [shape[1], shape[0]] : shape
        if weights.hasQuantization {
            throw ConvertError.notAvailable
        } else if !weights.floatValue.isEmpty {
            switch dataType {
                case .float32:
                let data: Data
                if doTranspose && shape.count == 2 {
                    var transposed = [Float](repeating: 0.0, count: weights.floatValue.count)
                    for x in 0..<shape[1] {
                        for y in 0..<shape[0] {
                            transposed[x * shape[0] + y] = weights.floatValue[y * shape[1] + x]
                        }
                    }
                    data = transposed.withUnsafeBufferPointer { Data(buffer: $0) }
                } else {
                    data = weights.floatValue.withUnsafeBufferPointer { Data(buffer: $0) }
                }
                return graph.constant(data, shape: _shape.map { NSNumber(value: $0) }, dataType: dataType)
                case .float16:
                let data: Data
                var converted = [Float16](repeating: 0.0, count: weights.floatValue.count)
                if doTranspose && shape.count == 2 {
                    for x in 0..<shape[1] {
                        for y in 0..<shape[0] {
                            converted[x * shape[0] + y] = Float16(weights.floatValue[y * shape[1] + x])
                        }
                    }
                } else {
                    for i in 0..<weights.floatValue.count {
                        converted[i] = Float16(weights.floatValue[i])
                    }
                }
                data = converted.withUnsafeBufferPointer { Data(buffer: $0) }
                return graph.constant(data, shape: _shape.map { NSNumber(value: $0) }, dataType: dataType)
                default:
                fatalError("not supported yet")
            }
        } else if !weights.float16Value.isEmpty {
            assert(dataType == .float16)
            if doTranspose && shape.count == 2 {
                let size = weights.float16Value.count / MemoryLayout<Float16>.stride
                let transposed = [Float16](unsafeUninitializedCapacity: size) { ptr, initializedSize in
                    weights.float16Value.withUnsafeBytes { (dataPtr: UnsafeRawBufferPointer) in
                        let dataSize = MemoryLayout<Float16>.stride / MemoryLayout<UInt8>.stride
                        for x in 0..<shape[1] {
                            for y in 0..<shape[0] {
                                ptr[x * shape[0] + y] = dataPtr.load(fromByteOffset: dataSize * (y * shape[1] + x), as: Float16.self)
                            }
                        }
                        initializedSize = size
                    }
                }
                let data = transposed.withUnsafeBufferPointer { Data(buffer: $0) }
                return graph.constant(data, shape: _shape.map { NSNumber(value: $0) }, dataType: dataType)
            } else {
                return graph.constant(weights.float16Value, shape: _shape.map { NSNumber(value: $0) }, dataType: dataType)
            }
        } else {
            throw ConvertError.wrongFormat
        }
    }

    private func convert(weights: CoreML_Specification_WeightParams, shape: [UInt64], doTranspose: Bool = false) throws -> MPSGraphTensor {
        return try convert(weights: weights, shape: shape.map { Int($0) }, doTranspose: doTranspose)
    }

    private func addConvolution(_ name: String, _ inputs: [String], _ params: CoreML_Specification_ConvolutionLayerParams) throws -> MPSGraphTensor {
        func getDescriptor(from params: CoreML_Specification_ConvolutionLayerParams) -> MPSGraphConvolution2DOpDescriptor {
            func convert(style: CoreML_Specification_ConvolutionLayerParams.OneOf_ConvolutionPaddingType) -> MPSGraphPaddingStyle {
                switch style {
                case .valid(_):
                    return .TF_VALID
                case .same(_):
                    return .TF_SAME
                }
            }
            return MPSGraphConvolution2DOpDescriptor(
                strideInX: Int(params.stride[1]),
                strideInY: Int(params.stride[0]),
                dilationRateInX: Int(params.dilationFactor[1]),
                dilationRateInY: Int(params.dilationFactor[0]),
                groups: Int(params.nGroups),
                paddingStyle: convert(style: params.convolutionPaddingType!),
                dataLayout: .NCHW, 
                weightsLayout: .OIHW
            )!
        }

        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }

        guard params.isDeconvolution == false else {
            throw ConvertError.notAvailable
        }
        let weights = try self.convert(weights: params.weights, shape: [params.outputChannels, params.kernelChannels] + params.kernelSize)
        let descriptor = getDescriptor(from: params)
        if params.hasBias_p && params.hasBias {
            let conv = graph.convolution2D(tensors[input0]!, weights: weights, descriptor: descriptor, name: nil)
            return graph.addition(conv, try self.convert(weights: params.bias, shape: [1, params.outputChannels, 1, 1]), name: name)
        } else {
            return graph.convolution2D(tensors[input0]!, weights: weights, descriptor: descriptor, name: name)
        }
    }

    private func addPooling(_ name: String, _ inputs: [String], _ params: CoreML_Specification_PoolingLayerParams) throws -> MPSGraphTensor {
        func getDescriptor(from params: CoreML_Specification_PoolingLayerParams, shape: [Int]) throws -> MPSGraphPooling2DOpDescriptor {
            let paddingStyle: MPSGraphPaddingStyle
            switch params.poolingPaddingType {
            case .valid(_):
                paddingStyle = .TF_VALID
            case .same(_):
                paddingStyle = .TF_SAME
            default:
                throw ConvertError.notAvailable
            }
            return MPSGraphPooling2DOpDescriptor(
                kernelWidth: params.globalPooling ? shape[3] : Int(params.kernelSize[1]),
                kernelHeight: params.globalPooling ? shape[2] : Int(params.kernelSize[0]),
                strideInX: params.globalPooling ? shape[3] : Int(params.stride[1]),
                strideInY: params.globalPooling ? shape[2] : Int(params.stride[0]),
                dilationRateInX: 1,
                dilationRateInY: 1,
                paddingLeft: 0, 
                paddingRight: 0, 
                paddingTop: 0, 
                paddingBottom: 0, 
                paddingStyle: paddingStyle, 
                dataLayout: .NCHW
            )!
        }

        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }

        let descriptor = try getDescriptor(from: params, shape: tensors[input0]!.shape!.map { $0.intValue })
        switch params.type {
            case .max:
                return graph.maxPooling2D(withSourceTensor: tensors[input0]!, descriptor: descriptor, name: name)
            case .average:
                return graph.avgPooling2D(withSourceTensor: tensors[input0]!, descriptor: descriptor, name: name)
            case .l2:
                throw ConvertError.notAvailable
            default:
                throw ConvertError.notAvailable
        }
    }

    private func addActivation(_ name: String, _ inputs: [String], _ params: CoreML_Specification_ActivationParams) throws -> MPSGraphTensor {
        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }

        switch params.nonlinearityType {
        case .linear(let p):
            return graph.addition(graph.multiplication(tensors[input0]!, graph.constant(Double(p.alpha), dataType: dataType), name: nil), graph.constant(Double(p.beta), dataType: dataType), name: name)
        case .reLu(let p):
            return graph.reLU(with: tensors[input0]!, name: name)
        case .leakyReLu(let p):
            if #available(macOS 12.0, iOS 15.0, macCatalyst 15.0, *) {
                return graph.leakyReLU(with: tensors[input0]!, alpha: Double(p.alpha), name: name)
            } else {
                fatalError("not available")
            }
        case .thresholdedReLu(let p):
            fatalError("not implemented yet")
        case .preLu(let p):
            fatalError("not implemented yet")
        case .tanh(let p):
            return graph.tanh(with: tensors[input0]!, name: name)
        case .scaledTanh(let p):
            let alpha = graph.constant(Double(p.alpha), dataType: dataType)
            let beta = graph.constant(Double(p.beta), dataType: dataType)
            return graph.multiplication(
                graph.tanh(with: graph.multiplication(tensors[input0]!, beta, name: nil), name: nil),
                alpha, name: name)
        case .sigmoid(let p):
            return graph.sigmoid(with: tensors[input0]!, name: name)
        case .sigmoidHard(let p):
            let linear = graph.addition(graph.multiplication(tensors[input0]!, graph.constant(Double(p.alpha), dataType: dataType), name: nil), graph.constant(Double(p.beta), dataType: dataType), name: nil)
            return graph.minimum(graph.maximum(linear, graph.constant(0.0, dataType: dataType), name: nil), graph.constant(1.0, dataType: dataType), name: name)
        case .elu(let p):
            fatalError("not implemented yet")
        case .softsign(let p):
            return graph.division(tensors[input0]!, graph.addition(graph.absolute(with: tensors[input0]!, name: nil), graph.constant(1.0, dataType: dataType), name: nil), name: name)
        case .softplus(let p):
            return graph.logarithm(with: graph.addition(graph.exponent(with: tensors[input0]!, name: nil), graph.constant(1.0, dataType: dataType), name: nil), name: name)
        case .parametricSoftplus(let p):
            fatalError("not implemented yet")
        case .none:
            fatalError("not implemented yet")
        }
    }

    private func addInnerProduct(_ name: String, _ inputs: [String], _ params: CoreML_Specification_InnerProductLayerParams) throws -> MPSGraphTensor {
        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }

        func innerProduct(name: String?) throws -> MPSGraphTensor {
            var shape = tensors[input0]!.shape!
            if variableBatches {
                shape[0] = -1
            }
            let weightShape = [params.outputChannels, params.inputChannels]
            let secondary = try convert(weights: params.weights, shape: weightShape, doTranspose: true)
            let primary: MPSGraphTensor
            switch shape.count {
            case 1, 2:
                primary = tensors[input0]!
                return graph.matrixMultiplication(primary: primary, secondary: secondary, name: name)
            case 4:
                primary = graph.reshape(tensors[input0]!, shape: [shape[0], NSNumber(value: shape.dropFirst(1).reduce(1) { $0 * $1.intValue })], name: nil)
                let multiply = graph.matrixMultiplication(primary: primary, secondary: secondary, name: nil)
                return graph.reshape(multiply, shape: [shape[0], NSNumber(value: params.outputChannels), 1, 1], name: name)
            case 3, 5:
                primary = graph.reshape(tensors[input0]!, shape: [NSNumber(value: shape[0].intValue * shape[1].intValue), NSNumber(value: shape.dropFirst(1).reduce(2) { $0 * $1.intValue })], name: nil)
                let multiply = graph.matrixMultiplication(primary: primary, secondary: secondary, name: nil)
                return graph.reshape(multiply, shape: [shape[0], shape[1], NSNumber(value: params.outputChannels), 1, 1], name: name)
            default:
                throw ConvertError.notAvailable
            }
        }

        if params.hasBias {
            var biasShape = [Int](repeating: 1, count: tensors[input0]!.shape!.count)
            switch biasShape.count {
            case 1:
                biasShape[0] = Int(params.outputChannels)
            case 2, 4:
                biasShape[1] = Int(params.outputChannels)
            case 3, 5:
                biasShape[2] = Int(params.outputChannels)
            default:
                throw ConvertError.notAvailable
            }
            return graph.addition(try innerProduct(name: nil), try convert(weights: params.bias, shape: biasShape), name: name)
        } else {
            return try innerProduct(name: name)
        }
    }

    private func addUnary(_ name: String, _ inputs: [String], _ params: CoreML_Specification_UnaryFunctionLayerParams) throws -> MPSGraphTensor {
        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }
        switch params.type {
        case .abs:
            return graph.absolute(with: tensors[input0]!, name: name)
        case .sqrt:
            return graph.squareRoot(with: tensors[input0]!, name: name)
        case .rsqrt:
            return graph.reverseSquareRoot(with: tensors[input0]!, name: name)
        case .inverse:
            return graph.division(graph.constant(1.0, dataType: dataType), tensors[input0]!, name: name)
        case .power:
            return graph.power(tensors[input0]!, graph.constant(Double(params.alpha), dataType: dataType), name: name)
        case .exp:
            return graph.exponent(with: tensors[input0]!, name: name)
        case .log:
            return graph.logarithm(with: tensors[input0]!, name: name)
        case .threshold:
            fatalError("not implemented yet")
        default:
            fatalError("not implemented yet")
        }
    }

    func build(from source: URL) throws -> ([String:String], [String:MPSGraphTensor], [String:MPSGraphTensor], MPSGraph) {
        let data = try Data(contentsOf: source)
        let model = try CoreML_Specification_Model(serializedData: data)
        var inputs = [String:MPSGraphTensor]()
        var outputs = [String:MPSGraphTensor]()
        switch model.specificationVersion {
        case 1, 2:
            for input in model.description_p.input {
                var shape = input.type.multiArrayType.shape.map { NSNumber(value: $0) }
                assert(shape.count == 3)
                shape.insert(NSNumber(value: -1), at: 0)
                variableBatches = true
                let placeholder = graph.placeholder(shape: shape, dataType: dataType, name: input.name)
                inputs[input.name] = placeholder
                tensors[input.name] = placeholder
            }
        default:
            for input in model.description_p.input {
                var shape = input.type.multiArrayType.shape.map { NSNumber(value: $0) }
                if input.type.multiArrayType.shapeRange.sizeRanges.first?.upperBound == -1 { // バッチが可変なら
                    shape[0] = -1
                    variableBatches = true
                }
                let placeholder = graph.placeholder(shape: shape, dataType: dataType, name: input.name)
                inputs[input.name] = placeholder
                tensors[input.name] = placeholder
            }
        }
        
        for (number, layer) in model.neuralNetwork.layers.enumerated() {
            if case .loadConstant(let params) = layer.layer {
                tensors[layer.output[0]] = try convert(weights: params.data, shape: params.shape)
                continue
            } else if case .loadConstantNd(let params) = layer.layer {
                tensors[layer.output[0]] = try convert(weights: params.data, shape: params.shape)
                continue
            }

            guard let input0Name = layer.input.first, let input0 = tensors[input0Name] else {
                fatalError("should not reach here")
            }
            let output: MPSGraphTensor;
            switch layer.layer {
            case .loadConstant(let params):
                fatalError("should not reach here")
            case .loadConstantNd(let params):
                fatalError("should not reach here")
            /// Start at 100 here
            case .convolution(let params):
                output = try addConvolution(layer.name, layer.input, params)
            case .pooling(let params):
                output = try addPooling(layer.name, layer.input, params)
            case .activation(let params):
                output = try addActivation(layer.name, layer.input, params)
            case .innerProduct(let params):
                output = try addInnerProduct(layer.name, layer.input, params)
            case .embedding(let params):
                fatalError("not implemented yet")
            /// Normalization-related Layers
            case .batchnorm(let params):
                guard let inputShape = input0.shape, inputShape.count >= 3 else {
                    throw ConvertError.wrongFormat
                }
                var weightShape = [Int](repeating: 1, count: inputShape.count)
                weightShape[weightShape.count - 3] = Int(params.channels)
                output = graph.normalize(
                    input0,
                    mean: try convert(weights: params.mean, shape: weightShape),
                    variance: try convert(weights: params.variance, shape: weightShape),
                    gamma: try convert(weights: params.gamma, shape: weightShape),
                    beta: try convert(weights: params.beta, shape: weightShape),
                    epsilon: params.epsilon,
                    name: layer.name)
            case .mvn(let params):
                fatalError("not implemented yet")
            case .l2Normalize(let params):
                fatalError("not implemented yet")
            case .softmax(let params):
                output = graph.softMax(with: input0, axis: 1, name: layer.name)
            case .lrn(let params):
                fatalError("not implemented yet")
            case .crop(let params):
                fatalError("not implemented yet")
            case .padding(let params):
                fatalError("not implemented yet")
            case .upsample(let params):
                fatalError("not implemented yet")
            case .resizeBilinear(let params):
                fatalError("not implemented yet")
            case .cropResize(let params):
                fatalError("not implemented yet")
            case .unary(let params):
                output = try addUnary(layer.name, layer.input, params)
            /// Element-wise Operations
            case .add(let params):
                if layer.input.count == 1 {
                    output = graph.addition(input0, graph.constant(Double(params.alpha), dataType: dataType), name: layer.name)
                } else if layer.input.count == 2 {
                    output = graph.addition(input0, tensors[layer.input[1]]!, name: layer.name)
                } else {
                    fatalError("not implemented yet")
                }
            case .multiply(let params):
                if layer.input.count == 1 {
                    output = graph.multiplication(input0, graph.constant(Double(params.alpha), dataType: dataType), name: layer.name)
                } else {
                    output = graph.multiplication(input0, tensors[layer.input[1]]!, name: layer.name)
                }
            case .average(let params):
                fatalError("not implemented yet")
            case .scale(let params):
                fatalError("not implemented yet")
            case .bias(let params):
                output = graph.addition(input0, try convert(weights: params.bias, shape: params.shape), name: layer.name)
            case .max(let params):
                output = graph.maximum(input0, tensors[layer.input[1]]!, name: layer.name)
            case .min(let params):
                output = graph.minimum(input0, tensors[layer.input[1]]!, name: layer.name)
            case .dot(let params):
                fatalError("not implemented yet")
            case .reduce(let params):
                fatalError("not implemented yet")
            /// Data Reorganization
            case .reshape(let params):
                switch params.mode {
                case .channelFirst:
                    var shape = params.targetShape.map { NSNumber(value: $0) }
                    if variableBatches {
                        shape[0] = -1
                    }
                    output = graph.reshape(input0, shape: shape, name: layer.name)
                case .channelLast:
                    let input = input0
                    var shape = input.shape! // [[S,]B,C,H,W]
                    shape[shape.endIndex - 2] = NSNumber(value: shape[shape.endIndex - 2].intValue * shape[shape.endIndex - 1].intValue)
                    shape[shape.endIndex - 1] = 1
                    let flattenHW = graph.reshape(input, shape: shape, name: nil) // [[S,]B,C,HW,1]
                    let transposed = graph.transposeTensor(flattenHW, dimension: shape.endIndex - 2, withDimension: shape.endIndex - 3, name: nil) // [[S,]B,HW,C,1]
                    var targetShape = params.targetShape.map { NSNumber(value: Int($0)) } // [[S,]B,C',H',W']
                    if variableBatches {
                        targetShape[0] = -1
                    }
                    var targetShape2 = targetShape
                    targetShape2[shape.endIndex - 1] = 1
                    targetShape2[shape.endIndex - 2] = NSNumber(value: Int(targetShape[targetShape.endIndex - 3]))
                    targetShape2[shape.endIndex - 3] = NSNumber(value: Int(targetShape[targetShape.endIndex - 2]) * Int(targetShape[targetShape.endIndex - 1]))
                    let flattenHW2 = graph.reshape(transposed, shape: targetShape2, name: nil) // [[S,]B,H'W',C',1]
                    let transposed2 = graph.transposeTensor(flattenHW2, dimension: targetShape2.endIndex - 2, withDimension: targetShape2.endIndex - 3, name: nil) // [[S,]B,C',H'W',1]

                    output = graph.reshape(transposed2, shape: targetShape, name: layer.name) // [[S,]B,C',H',W']
                default:
                    fatalError("unrecognized mode")
                }
            case .flatten(let params):
                //tensors[input0] = graph.flatten2D(input0, axis: params.axis, name: layer.name)
                fatalError("not implemented yet")
            case .permute(let params):
                fatalError("not implemented yet")
            case .concat(let params):
                output = graph.concatTensors(layer.input.map { tensors[$0]! }, dimension: params.sequenceConcat ? -5 : -3, name: layer.name)
            case .split(let params):
                fatalError("not implemented yet")
            case .sequenceRepeat(let params):
                fatalError("not implemented yet")
            case .reorganizeData(let params):
                fatalError("not implemented yet")
            case .slice(let params):
                output = graph.sliceTensor(input0, dimension: params.axis.rawValue, start: Int(params.startIndex), length: Int(params.endIndex - params.startIndex), name: layer.name)
            /// Recurrent Layers
            case .simpleRecurrent(let params):
                fatalError("not implemented yet")
            case .gru(let params):
                fatalError("not implemented yet")
            case .uniDirectionalLstm(let params):
                fatalError("not implemented yet")
            case .biDirectionalLstm(let params):
                fatalError("not implemented yet")
            /// Custom (user-implemented) Layer
            case .custom(let params):
                fatalError("not implemented yet")
            /// Control Flow related Layers
            case .copy(let params):
                fatalError("not implemented yet")
            case .branch(let params):
                fatalError("not implemented yet")
            case .loop(let params):
                fatalError("not implemented yet")
            case .loopBreak(let params):
                fatalError("not implemented yet")
            case .loopContinue(let params):
                fatalError("not implemented yet")
            case .rangeStatic(let params):
                fatalError("not implemented yet")
            case .rangeDynamic(let params):
                fatalError("not implemented yet")
            /// Element-wise Unary Layers
            case .clip(let params):
                fatalError("not implemented yet")
            case .ceil(let params):
                output = graph.ceil(with: input0, name: layer.name)
            case .floor(let params):
                output = graph.floor(with: input0, name: layer.name)
            case .sign(let params):
                fatalError("not implemented yet")
            case .round(let params):
                output = graph.round(with: input0, name: layer.name)
            case .exp2(let params):
                output = graph.exponentBase2(with: input0, name: layer.name)
            case .sin(let params):
                output = graph.sin(with: input0, name: layer.name)
            case .cos(let params):
                output = graph.cos(with: input0, name: layer.name)
            case .tan(let params):
                output = graph.tan(with: input0, name: layer.name)
            case .asin(let params):
                output = graph.asin(with: input0, name: layer.name)
            case .acos(let params):
                output = graph.acos(with: input0, name: layer.name)
            case .atan(let params):
                output = graph.atan(with: input0, name: layer.name)
            case .sinh(let params):
                output = graph.sinh(with: input0, name: layer.name)
            case .cosh(let params):
                output = graph.cosh(with: input0, name: layer.name)
            case .tanh(let params):
                output = graph.tanh(with: input0, name: layer.name)
            case .asinh(let params):
                output = graph.asinh(with: input0, name: layer.name)
            case .acosh(let params):
                output = graph.acosh(with: input0, name: layer.name)
            case .atanh(let params):
                output = graph.atanh(with: input0, name: layer.name)
            case .erf(let params):
                fatalError("not implemented yet")
            case .gelu(let params):
                fatalError("not implemented yet")
            /// Element-wise Binary with Broadcasting Support
            case .equal(let params):
                fatalError("not implemented yet")
            case .notEqual(let params):
                fatalError("not implemented yet")
            case .lessThan(let params):
                fatalError("not implemented yet")
            case .lessEqual(let params):
                fatalError("not implemented yet")
            case .greaterThan(let params):
                fatalError("not implemented yet")
            case .greaterEqual(let params):
                fatalError("not implemented yet")
            case .logicalOr(let params):
                fatalError("not implemented yet")
            case .logicalXor(let params):
                fatalError("not implemented yet")
            case .logicalNot(let params):
                fatalError("not implemented yet")
            case .logicalAnd(let params):
                fatalError("not implemented yet")
            case .modBroadcastable(let params):
                fatalError("not implemented yet")
            case .minBroadcastable(let params):
                fatalError("not implemented yet")
            case .maxBroadcastable(let params):
                fatalError("not implemented yet")
            case .addBroadcastable(let params):
                output = graph.addition(input0, tensors[layer.input[1]]!, name: layer.name)
            case .powBroadcastable(let params):
                fatalError("not implemented yet")
            case .divideBroadcastable(let params):
                fatalError("not implemented yet")
            case .floorDivBroadcastable(let params):
                fatalError("not implemented yet")
            case .multiplyBroadcastable(let params):
                fatalError("not implemented yet")
            case .subtractBroadcastable(let params):
                fatalError("not implemented yet")
            /// Tensor Manipulations
            case .tile(let params):
                fatalError("not implemented yet")
            case .stack(let params):
                fatalError("not implemented yet")
            case .gather(let params):
                fatalError("not implemented yet")
            case .scatter(let params):
                fatalError("not implemented yet")
            case .gatherNd(let params):
                fatalError("not implemented yet")
            case .scatterNd(let params):
                fatalError("not implemented yet")
            case .softmaxNd(let params):
                fatalError("not implemented yet")
            case .gatherAlongAxis(let params):
                fatalError("not implemented yet")
            case .scatterAlongAxis(let params):
                fatalError("not implemented yet")
            case .reverse(let params):
                fatalError("not implemented yet")
            case .reverseSeq(let params):
                fatalError("not implemented yet")
            case .splitNd(let params):
                fatalError("not implemented yet")
            case .concatNd(let params):
                fatalError("not implemented yet")
            case .transpose(let params):
                fatalError("not implemented yet")
            case .sliceStatic(let params):
                fatalError("not implemented yet")
            case .sliceDynamic(let params):
                fatalError("not implemented yet")
            case .slidingWindows(let params):
                fatalError("not implemented yet")
            case .topK(let params):
                fatalError("not implemented yet")
            case .argMin(let params):
                fatalError("not implemented yet")
            case .argMax(let params):
                fatalError("not implemented yet")
            case .embeddingNd(let params):
                fatalError("not implemented yet")
            case .batchedMatmul(let params):
                output = graph.matrixMultiplication(primary: tensors[layer.input[0]]!, secondary: tensors[layer.input[1]]!, name: layer.name)
            /// Tensor Allocation / Reshape-related Operations
            case .getShape(let params):
                fatalError("not implemented yet")
            case .fillLike(let params):
                fatalError("not implemented yet")
            case .fillStatic(let params):
                fatalError("not implemented yet")
            case .fillDynamic(let params):
                fatalError("not implemented yet")
            case .broadcastToLike(let params):
                fatalError("not implemented yet")
            case .broadcastToStatic(let params):
                fatalError("not implemented yet")
            case .broadcastToDynamic(let params):
                fatalError("not implemented yet")
            case .squeeze(let params):
                output = graph.squeeze(input0, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
            case .expandDims(let params):
                output = graph.expandDims(input0, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
            case .flattenTo2D(let params):
                fatalError("not implemented yet")
            case .reshapeLike(let params):
                fatalError("not implemented yet")
            case .reshapeStatic(let params):
                output = graph.reshape(input0, shape: params.targetShape.map { NSNumber(value: $0) }, name: layer.name)
            case .reshapeDynamic(let params):
                fatalError("not implemented yet")
            case .rankPreservingReshape(let params):
                fatalError("not implemented yet")
            case .constantPad(let params):
                fatalError("not implemented yet")
            /// Random Distributions
            case .randomNormalLike(let params):
                fatalError("not implemented yet")
            case .randomNormalStatic(let params):
                fatalError("not implemented yet")
            case .randomNormalDynamic(let params):
                fatalError("not implemented yet")
            case .randomUniformLike(let params):
                fatalError("not implemented yet")
            case .randomUniformStatic(let params):
                fatalError("not implemented yet")
            case .randomUniformDynamic(let params):
                fatalError("not implemented yet")
            case .randomBernoulliLike(let params):
                fatalError("not implemented yet")
            case .randomBernoulliStatic(let params):
                fatalError("not implemented yet")
            case .randomBernoulliDynamic(let params):
                fatalError("not implemented yet")
            case .categoricalDistribution(let params):
                fatalError("not implemented yet")
            /// Reduction-related Layers:
            case .reduceL1(let params):
                fatalError("not implemented yet")
            case .reduceL2(let params):
                fatalError("not implemented yet")
            case .reduceMax(let params):
                output = graph.reductionMaximum(with: input0, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
            case .reduceMin(let params):
                fatalError("not implemented yet")
            case .reduceSum(let params):
                fatalError("not implemented yet")
            case .reduceProd(let params):
                fatalError("not implemented yet")
            case .reduceMean(let params):
                fatalError("not implemented yet")
            case .reduceLogSum(let params):
                fatalError("not implemented yet")
            case .reduceSumSquare(let params):
                fatalError("not implemented yet")
            case .reduceLogSumExp(let params):
                fatalError("not implemented yet")
            /// Masking / Selection Layers
            case .whereNonZero(let params):
                fatalError("not implemented yet")
            case .matrixBandPart(let params):
                fatalError("not implemented yet")
            case .lowerTriangular(let params):
                fatalError("not implemented yet")
            case .upperTriangular(let params):
                fatalError("not implemented yet")
            case .whereBroadcastable(let params):
                fatalError("not implemented yet")
            /// Normalization Layers
            case .layerNormalization(let params):
                fatalError("not implemented yet")
            case .nonMaximumSuppression(let params):
                fatalError("not implemented yet")
            /// Following layers are available only after Core ML Specification
            /// version >= 5 (iOS >= 14, macOS >= 11.0)
            case .oneHot(let params):
                fatalError("not implemented yet")
            case .cumSum(let params):
                fatalError("not implemented yet")
            case .clampedReLu(let params):
                fatalError("not implemented yet")
            case .argSort(let params):
                fatalError("not implemented yet")
            case .pooling3D(let params):
                fatalError("not implemented yet")
            case .globalPooling3D(let params):
                fatalError("not implemented yet")
            case .sliceBySize(let params):
                fatalError("not implemented yet")
            case .convolution3D(let params):
                fatalError("not implemented yet")
            case .none:
                fatalError("not implemented yet")
            }
            tensors[layer.output[0]] = output;
        }
        for output in model.description_p.output {
            if let tensor = tensors[output.name] {
                outputs[output.name] = tensor
            } else {
                print(model.description_p.output)
                print(outputs.keys)
                fatalError("no such output")
            }
        }
        return (model.description_p.metadata.userDefined, inputs, outputs, graph)
    }
}

public func mlmodelToMPSGraph(from source: URL, dataType: MPSDataType) throws -> ([String:String], [String:MPSGraphTensor], [String:MPSGraphTensor], MPSGraph)  {
    let builder = MPSGraphBuilder(dataType: dataType)
    return try builder.build(from: source)
}
