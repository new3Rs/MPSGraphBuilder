import MetalPerformanceShadersGraph

extension MPSDataType {
    static func from(_ type: CoreML_Specification_ArrayFeatureType.ArrayDataType) throws -> MPSDataType {
        switch type {
            case .invalidArrayDataType:
            return .invalid
            case .float32:
            return .float32
            case .int32:
            return .int32
            default: // MPSGraph does not support .double 
            throw ConvertError.notAvailable
        }
    }
}

extension MPSGraphPaddingStyle {
    static func from(_ style: CoreML_Specification_ConvolutionLayerParams.OneOf_ConvolutionPaddingType) -> MPSGraphPaddingStyle {
        switch style {
        case .valid(_):
            return .TF_VALID
        case .same(_):
            return .TF_SAME
        }
    }
}

extension MPSGraphConvolution2DOpDescriptor {
    static func from(_ params: CoreML_Specification_ConvolutionLayerParams) -> MPSGraphConvolution2DOpDescriptor {
        return MPSGraphConvolution2DOpDescriptor(
            strideInX: Int(params.stride[1]),
            strideInY: Int(params.stride[0]),
            dilationRateInX: Int(params.dilationFactor[1]),
            dilationRateInY: Int(params.dilationFactor[0]),
            groups: Int(params.nGroups),
            paddingStyle: MPSGraphPaddingStyle.from(params.convolutionPaddingType!),
            dataLayout: .NCHW, 
            weightsLayout: .OIHW
        )!
    }
}

extension MPSGraphPooling2DOpDescriptor {
    static func from(_ params: CoreML_Specification_PoolingLayerParams, shape: [Int]) throws -> MPSGraphPooling2DOpDescriptor {
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


}