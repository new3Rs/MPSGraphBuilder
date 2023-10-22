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
}
