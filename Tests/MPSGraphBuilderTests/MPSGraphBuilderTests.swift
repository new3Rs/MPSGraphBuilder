import XCTest
import MetalPerformanceShadersGraph
@testable import MPSGraphBuilder

final class MPSGraphBuilderTests: XCTestCase {
    func testSAI() throws {
        XCTAssertNoThrow(try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/SAI_773_fp16", withExtension: "mlmodel")!))
    }

    func testKataGo() throws {
        XCTAssertNoThrow(try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/KataGoKata1_b18c384nbts658", withExtension: "mlmodel")!))
    }

    func testPerformanceKataGo() throws {
        let batch = NSNumber(value: 16)
        guard let device = MTLCreateSystemDefaultDevice() else { 
           fatalError( "Failed to get the system's default Metal device." ) 
        }
        let (userDefined, inputs, outputs, graph) = try mlmodelToMPSGraph(from: Bundle.module.url(forResource: "MLModels/KataGoKata1_b18c384nbts658", withExtension: "mlmodel")!)
        let feeds = [
            inputs["bin_inputs"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: [batch,22,19,19]))),
            inputs["global_inputs"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: [batch,19]))),
            inputs["mask_sum_hw_sqrt_offset_10"]!: MPSGraphTensorData(MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: [1,1,1,1])))
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
