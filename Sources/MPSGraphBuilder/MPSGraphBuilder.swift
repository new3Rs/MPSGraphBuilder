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
        print("convertType", type)
        throw ConvertError.notAvailable
    }
}

class MPSGraphBuilder {
    let graph = MPSGraph()
    var dataType = MPSDataType.float32
    var tensors = [String:MPSGraphTensor]()
    var variableBatches = false

    private func convert(weights: CoreML_Specification_WeightParams, shape: [Int], doTranspose: Bool = false) throws -> MPSGraphTensor {
        let _shape = doTranspose ? [shape[1], shape[0]] : shape
        if weights.hasQuantization {
            print("convert(weights:shape:doTranspose:)")
            throw ConvertError.notAvailable
        } else if !weights.floatValue.isEmpty {
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
            return graph.constant(data, shape: _shape.map { NSNumber(value: $0) }, dataType: .float32)
        } else if !weights.float16Value.isEmpty {
            let size = weights.float16Value.count / MemoryLayout<UInt16>.stride
            var uint16 = [UInt16](unsafeUninitializedCapacity: size) { buffer, initializedCount in
                let _ = weights.float16Value.copyBytes(to: buffer)
                initializedCount = size
            }
            if doTranspose && shape.count == 2 {
                var transposed = [UInt16](repeating: 0, count: size)
                for x in 0..<shape[1] {
                    for y in 0..<shape[0] {
                        transposed[x * shape[0] + y] = uint16[y * shape[1] + x]
                    }
                }
                uint16 = transposed
            }
            var float32 = [Float](repeating: 0.0, count: size)
            var sourceBuffer = vImage_Buffer(data: &uint16, height: 1, width: UInt(uint16.count), rowBytes: MemoryLayout<UInt16>.stride)
            var destinationBuffer = vImage_Buffer(data: &float32, height: 1, width: UInt(uint16.count), rowBytes: MemoryLayout<Float>.stride)
            if vImageConvert_Planar16FtoPlanarF(&sourceBuffer, &destinationBuffer, 0) != kvImageNoError {
                throw ConvertError.wrongFormat
            }
            let data = float32.withUnsafeBufferPointer { Data(buffer: $0) }
            return graph.constant(data, shape: _shape.map { NSNumber(value: $0) }, dataType: .float32)
        } else {
            throw ConvertError.wrongFormat
        }
    }

    private func convert(weights: CoreML_Specification_WeightParams, shape: [UInt64], doTranspose: Bool = false) throws -> MPSGraphTensor {
        return try convert(weights: weights, shape: shape.map { Int($0) }, doTranspose: doTranspose)
    }

    private func addConvolution(_ name: String, _ inputs: [String], _ output: String, _ params: CoreML_Specification_ConvolutionLayerParams) throws {
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
            tensors[output] = graph.addition(conv, try self.convert(weights: params.bias, shape: [1, params.outputChannels, 1, 1]), name: name)
        } else {
            tensors[output] = graph.convolution2D(tensors[input0]!, weights: weights, descriptor: descriptor, name: name)
        }
    }

    private func addPooling(_ name: String, _ inputs: [String], _ output: String, _ params: CoreML_Specification_PoolingLayerParams) throws {
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
                tensors[output] = graph.maxPooling2D(withSourceTensor: tensors[input0]!, descriptor: descriptor, name: name)
            case .average:
                tensors[output] = graph.avgPooling2D(withSourceTensor: tensors[input0]!, descriptor: descriptor, name: name)
            case .l2:
                throw ConvertError.notAvailable
            default:
                throw ConvertError.notAvailable
        }
    }

    private func addActivation(_ name: String, _ inputs: [String], _ output: String, _ params: CoreML_Specification_ActivationParams) throws {
        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }

        switch params.nonlinearityType {
        case .linear(let p):
            tensors[output] = graph.addition(graph.multiplication(tensors[input0]!, graph.constant(Double(p.alpha), dataType: dataType), name: nil), graph.constant(Double(p.beta), dataType: dataType), name: name)
        case .reLu(let p):
            tensors[output] = graph.reLU(with: tensors[input0]!, name: name)
        case .leakyReLu(let p):
            if #available(macOS 12.0, iOS 15.0, macCatalyst 15.0, *) {
                tensors[output] = graph.leakyReLU(with: tensors[input0]!, alpha: Double(p.alpha), name: name)
            } else {
                fatalError("not available")
            }
        case .thresholdedReLu(let p):
            fatalError("not implemented yet")
        case .preLu(let p):
            fatalError("not implemented yet")
        case .tanh(let p):
            tensors[output] = graph.tanh(with: tensors[input0]!, name: name)
        case .scaledTanh(let p):
            fatalError("not implemented yet")
        case .sigmoid(let p):
            tensors[output] = graph.sigmoid(with: tensors[input0]!, name: name)
        case .sigmoidHard(let p):
            let linear = graph.addition(graph.multiplication(tensors[input0]!, graph.constant(Double(p.alpha), dataType: dataType), name: nil), graph.constant(Double(p.beta), dataType: dataType), name: nil)
            tensors[output] = graph.minimum(graph.maximum(linear, graph.constant(0.0, dataType: dataType), name: nil), graph.constant(1.0, dataType: dataType), name: name)
        case .elu(let p):
            fatalError("not implemented yet")
        case .softsign(let p):
            tensors[output] = graph.division(tensors[input0]!, graph.addition(graph.absolute(with: tensors[input0]!, name: nil), graph.constant(1.0, dataType: dataType), name: nil), name: name)
        case .softplus(let p):
            tensors[output] = graph.logarithm(with: graph.addition(graph.exponent(with: tensors[input0]!, name: nil), graph.constant(1.0, dataType: dataType), name: nil), name: name)
        case .parametricSoftplus(let p):
            fatalError("not implemented yet")
        case .none:
            fatalError("not implemented yet")
        }
    }

    private func addInnerProduct(_ name: String, _ inputs: [String], _ output: String, _ params: CoreML_Specification_InnerProductLayerParams) throws {
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
            var biasShape = tensors[input0]!.shape!.map { $0.intValue }
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
            tensors[output] = graph.addition(try innerProduct(name: nil), try convert(weights: params.bias, shape: biasShape), name: name)
        } else {
            tensors[output] = try innerProduct(name: name)
        }
    }

    private func addUnary(_ name: String, _ inputs: [String], _ output: String, _ params: CoreML_Specification_UnaryFunctionLayerParams) throws {
        guard let input0 = inputs.first else {
            throw ConvertError.wrongFormat
        }
        switch params.type {
        case .abs:
            tensors[output] = graph.absolute(with: tensors[input0]!, name: name)
        case .sqrt:
            tensors[output] = graph.squareRoot(with: tensors[input0]!, name: name)
        case .rsqrt:
            tensors[output] = graph.reverseSquareRoot(with: tensors[input0]!, name: name)
        case .inverse:
            tensors[output] = graph.division(graph.constant(1.0, dataType: dataType), tensors[input0]!, name: name)
        case .power:
            tensors[output] = graph.power(tensors[input0]!, graph.constant(Double(params.alpha), dataType: dataType), name: name)
        case .exp:
            tensors[output] = graph.exponent(with: tensors[input0]!, name: name)
        case .log:
            tensors[output] = graph.logarithm(with: tensors[input0]!, name: name)
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
                let placeholder = graph.placeholder(shape: shape, dataType: try convertType(input.type.multiArrayType.dataType), name: input.name)
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
                let placeholder = graph.placeholder(shape: shape, dataType: try convertType(input.type.multiArrayType.dataType), name: input.name)
                inputs[input.name] = placeholder
                tensors[input.name] = placeholder
            }
        }
        
        for (number, layer) in model.neuralNetwork.layers.enumerated() {
            print(number, layer.name, String(describing: layer.layer).split(whereSeparator: \.isNewline)[0])
            fflush(stdout)
            if case .loadConstant(let params) = layer.layer {
                tensors[layer.output[0]] = try convert(weights: params.data, shape: params.shape)
                continue
            } else if case .loadConstantNd(let params) = layer.layer {
                tensors[layer.output[0]] = try convert(weights: params.data, shape: params.shape)
                continue
            }
            guard let input0 = layer.input.first else {
                fatalError("should not reach here")
            }
            switch layer.layer {
            case .loadConstant(let params):
                fatalError("should not reach here")
            case .loadConstantNd(let params):
                fatalError("should not reach here")
            /// Start at 100 here
            case .convolution(let params):
                try addConvolution(layer.name, layer.input, layer.output[0], params)
            case .pooling(let params):
                try addPooling(layer.name, layer.input, layer.output[0], params)
            case .activation(let params):
                try addActivation(layer.name, layer.input, layer.output[0], params)
            case .innerProduct(let params):
                try addInnerProduct(layer.name, layer.input, layer.output[0], params)
            case .embedding(let params):
                fatalError("not implemented yet")
            /// Normalization-related Layers
            case .batchnorm(let params):
                print(tensors[input0]!.shape)
                fflush(stdout)
                guard let inputShape = tensors[input0]?.shape, inputShape.count >= 3 else {
                    throw ConvertError.wrongFormat
                }
                var weightShape = [Int](repeating: 1, count: inputShape.count)
                weightShape[weightShape.count - 3] = Int(params.channels)
                tensors[layer.output[0]] = graph.normalize(
                    tensors[input0]!,
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
                tensors[layer.output[0]] = graph.softMax(with: tensors[input0]!, axis: 1, name: layer.name)
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
                try addUnary(layer.name, layer.input, layer.output[0], params)
            /// Element-wise Operations
            case .add(let params):
                if layer.input.count == 1 {
                    tensors[layer.output[0]] = graph.addition(tensors[input0]!, graph.constant(Double(params.alpha), dataType: dataType), name: layer.name)
                } else {
                    tensors[layer.output[0]] = graph.addition(tensors[input0]!, tensors[layer.input[1]]!, name: layer.name)
                }
            case .multiply(let params):
                if layer.input.count == 1 {
                    tensors[layer.output[0]] = graph.multiplication(tensors[input0]!, graph.constant(Double(params.alpha), dataType: dataType), name: layer.name)
                } else {
                    tensors[layer.output[0]] = graph.multiplication(tensors[input0]!, tensors[layer.input[1]]!, name: layer.name)
                }
            case .average(let params):
                fatalError("not implemented yet")
            case .scale(let params):
                fatalError("not implemented yet")
            case .bias(let params):
                tensors[layer.output[0]] = graph.addition(tensors[input0]!, try convert(weights: params.bias, shape: params.shape), name: layer.name)
            case .max(let params):
                tensors[layer.output[0]] = graph.maximum(tensors[input0]!, tensors[layer.input[1]]!, name: layer.name)
            case .min(let params):
                tensors[layer.output[0]] = graph.minimum(tensors[input0]!, tensors[layer.input[1]]!, name: layer.name)
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
                    tensors[layer.output[0]] = graph.reshape(tensors[input0]!, shape: shape, name: layer.name)
                case .channelLast:
                    let input = tensors[input0]!
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

                    tensors[layer.output[0]] = graph.reshape(transposed2, shape: targetShape, name: layer.name) // [[S,]B,C',H',W']
                default:
                    fatalError("unrecognized mode")
                }
            case .flatten(let params):
                //tensors[input0] = graph.flatten2D(tensors[input0]!, axis: params.axis, name: layer.name)
                fatalError("not implemented yet")
            case .permute(let params):
                fatalError("not implemented yet")
            case .concat(let params):
                tensors[layer.output[0]] = graph.concatTensors(layer.input.map { tensors[$0]! }, dimension: params.sequenceConcat ? -5 : -3, name: layer.name)
            case .split(let params):
                fatalError("not implemented yet")
            case .sequenceRepeat(let params):
                fatalError("not implemented yet")
            case .reorganizeData(let params):
                fatalError("not implemented yet")
            case .slice(let params):
                tensors[layer.output[0]] = graph.sliceTensor(tensors[input0]!, dimension: params.axis.rawValue, start: Int(params.startIndex), length: Int(params.endIndex - params.startIndex), name: layer.name)
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
                tensors[layer.output[0]] = graph.ceil(with: tensors[input0]!, name: layer.name)
            case .floor(let params):
                tensors[layer.output[0]] = graph.floor(with: tensors[input0]!, name: layer.name)
            case .sign(let params):
                fatalError("not implemented yet")
            case .round(let params):
                tensors[layer.output[0]] = graph.round(with: tensors[input0]!, name: layer.name)
            case .exp2(let params):
                tensors[layer.output[0]] = graph.exponentBase2(with: tensors[input0]!, name: layer.name)
            case .sin(let params):
                tensors[layer.output[0]] = graph.sin(with: tensors[input0]!, name: layer.name)
            case .cos(let params):
                tensors[layer.output[0]] = graph.cos(with: tensors[input0]!, name: layer.name)
            case .tan(let params):
                tensors[layer.output[0]] = graph.tan(with: tensors[input0]!, name: layer.name)
            case .asin(let params):
                tensors[layer.output[0]] = graph.asin(with: tensors[input0]!, name: layer.name)
            case .acos(let params):
                tensors[layer.output[0]] = graph.acos(with: tensors[input0]!, name: layer.name)
            case .atan(let params):
                tensors[layer.output[0]] = graph.atan(with: tensors[input0]!, name: layer.name)
            case .sinh(let params):
                tensors[layer.output[0]] = graph.sinh(with: tensors[input0]!, name: layer.name)
            case .cosh(let params):
                tensors[layer.output[0]] = graph.cosh(with: tensors[input0]!, name: layer.name)
            case .tanh(let params):
                tensors[layer.output[0]] = graph.tanh(with: tensors[input0]!, name: layer.name)
            case .asinh(let params):
                tensors[layer.output[0]] = graph.asinh(with: tensors[input0]!, name: layer.name)
            case .acosh(let params):
                tensors[layer.output[0]] = graph.acosh(with: tensors[input0]!, name: layer.name)
            case .atanh(let params):
                tensors[layer.output[0]] = graph.atanh(with: tensors[input0]!, name: layer.name)
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
                fatalError("not implemented yet")
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
                fatalError("not implemented yet")
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
                tensors[layer.output[0]] = graph.squeeze(tensors[input0]!, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
            case .expandDims(let params):
                tensors[layer.output[0]] = graph.expandDims(tensors[input0]!, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
            case .flattenTo2D(let params):
                fatalError("not implemented yet")
            case .reshapeLike(let params):
                fatalError("not implemented yet")
            case .reshapeStatic(let params):
                tensors[layer.output[0]] = graph.reshape(tensors[input0]!, shape: params.targetShape.map { NSNumber(value: $0) }, name: layer.name)
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
                tensors[layer.output[0]] = graph.reductionMaximum(with: tensors[input0]!, axes: params.axes.map { NSNumber(value: $0) }, name: layer.name)
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

public func mlmodelToMPSGraph(from source: URL) throws -> ([String:String], [String:MPSGraphTensor], [String:MPSGraphTensor], MPSGraph)  {
    let builder = MPSGraphBuilder()
    return try builder.build(from: source)
}
