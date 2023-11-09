import XCTest
import MetalPerformanceShadersGraph
@testable import MPSGraphBuilder

final class MPSGraphBuilderTests: XCTestCase {
    func testSAI() throws {
        XCTAssertNoThrow(try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/SAI_773_fp16", withExtension: "mlmodel")!, dataType: .float16))
    }

    func testKataGo() throws {
        XCTAssertNoThrow(try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/KataGoKata1_b18c384nbts658", withExtension: "mlmodel")!, dataType: .float32))
    }

    func testKataGoFp16() throws {
        XCTAssertNoThrow(try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/KataGoKata1_b18c384nbts658", withExtension: "mlmodel")!, dataType: .float16))
    }

    func testPerformanceKataGo() throws {
        let dataType = MPSDataType.float16
        let batch = NSNumber(value: 1)
        guard let device = MTLCreateSystemDefaultDevice() else { 
           fatalError( "Failed to get the system's default Metal device." ) 
        }
        let (userDefined, inputs, outputs, graph) = try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/KataGoKata1_b18c384nbts658", withExtension: "mlmodel")!, dataType: dataType)
        let feeds = [
            inputs["bin_inputs"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: dataType, shape: [batch,22,19,19]))),
            inputs["global_inputs"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: dataType, shape: [batch,19]))),
            inputs["mask_sum_hw_sqrt_offset_10"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: dataType, shape: [1,1,1,1])))
        ]
        measure {
            let _ = graph.run(
                feeds: feeds,
                targetTensors: outputs.map { k, v in v },
                targetOperations: nil
            )
        }
    }
}
