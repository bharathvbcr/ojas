import Foundation

public struct Arch02StepMetrics: Sendable {
    public let completedStep: UInt64
    public let loss: Float
    public let gradientNorm: Float
    public let clipFactor: Float
    public let learningRateMultiplier: Float
    public let dispatches: UInt64
}

private struct CStepMetrics {
    var completed_step: UInt64 = 0
    var loss: Float = 0
    var grad_norm: Float = 0
    var clip_factor: Float = 0
    var lr_multiplier: Float = 0
    var dispatches: UInt64 = 0
}

@_silgen_name("arch02_engine_create")
private func cCreate(_ json: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>) -> Int32
@_silgen_name("arch02_engine_load")
private func cLoad(_ path: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<OpaquePointer?>) -> Int32
@_silgen_name("arch02_engine_expected_tokens")
private func cExpectedTokens(_ engine: OpaquePointer) -> Int
@_silgen_name("arch02_engine_train")
private func cTrain(_ engine: OpaquePointer, _ input: UnsafePointer<Int32>,
                    _ target: UnsafePointer<Int32>, _ count: Int,
                    _ metrics: UnsafeMutablePointer<CStepMetrics>) -> Int32
@_silgen_name("arch02_engine_evaluate")
private func cEvaluate(_ engine: OpaquePointer, _ input: UnsafePointer<Int32>,
                       _ target: UnsafePointer<Int32>, _ count: Int,
                       _ loss: UnsafeMutablePointer<Float>) -> Int32
@_silgen_name("arch02_engine_save")
private func cSave(_ engine: OpaquePointer, _ path: UnsafePointer<CChar>) -> Int32
@_silgen_name("arch02_engine_destroy")
private func cDestroy(_ engine: OpaquePointer?)
@_silgen_name("arch02_last_error")
private func cLastError(_ buffer: UnsafeMutablePointer<CChar>?, _ capacity: Int) -> Int

public enum Arch02EngineError: Error, CustomStringConvertible {
    case native(String)
    case tokenCount(expected: Int, inputs: Int, targets: Int)

    public var description: String {
        switch self {
        case .native(let message): return message
        case .tokenCount(let expected, let inputs, let targets):
            return "expected \(expected) tokens, got inputs=\(inputs), targets=\(targets)"
        }
    }
}

/// Move-safe at the Swift boundary: the owning pointer is held by one final
/// reference type, while step metrics are immutable value types.
public final class Arch02Engine {
    private let handle: OpaquePointer

    public convenience init() throws { try self.init(configurationJSON: nil) }

    public init(configurationJSON: String?) throws {
        var result: OpaquePointer?
        let status: Int32
        if let configurationJSON {
            status = configurationJSON.withCString { cCreate($0, &result) }
        } else {
            status = cCreate(nil, &result)
        }
        try Self.check(status)
        guard let result else { throw Arch02EngineError.native("native create returned no engine") }
        handle = result
    }

    public init(checkpointPath: String) throws {
        var result: OpaquePointer?
        let status = checkpointPath.withCString { cLoad($0, &result) }
        try Self.check(status)
        guard let result else { throw Arch02EngineError.native("native load returned no engine") }
        handle = result
    }

    deinit { cDestroy(handle) }

    public var expectedTokenCount: Int { cExpectedTokens(handle) }

    public func train(inputs: [Int32], targets: [Int32]) throws -> Arch02StepMetrics {
        try validate(inputs, targets)
        var value = CStepMetrics()
        let status = inputs.withUnsafeBufferPointer { input in
            targets.withUnsafeBufferPointer { target in
                cTrain(handle, input.baseAddress!, target.baseAddress!, input.count, &value)
            }
        }
        try Self.check(status)
        return Arch02StepMetrics(completedStep: value.completed_step, loss: value.loss,
            gradientNorm: value.grad_norm, clipFactor: value.clip_factor,
            learningRateMultiplier: value.lr_multiplier, dispatches: value.dispatches)
    }

    public func evaluate(inputs: [Int32], targets: [Int32]) throws -> Float {
        try validate(inputs, targets)
        var loss: Float = 0
        let status = inputs.withUnsafeBufferPointer { input in
            targets.withUnsafeBufferPointer { target in
                cEvaluate(handle, input.baseAddress!, target.baseAddress!, input.count, &loss)
            }
        }
        try Self.check(status)
        return loss
    }

    public func save(to checkpointPath: String) throws {
        try Self.check(checkpointPath.withCString { cSave(handle, $0) })
    }

    private func validate(_ inputs: [Int32], _ targets: [Int32]) throws {
        let expected = expectedTokenCount
        guard inputs.count == expected, targets.count == expected else {
            throw Arch02EngineError.tokenCount(expected: expected,
                inputs: inputs.count, targets: targets.count)
        }
    }

    private static func check(_ status: Int32) throws {
        guard status == 0 else {
            let n = cLastError(nil, 0)
            var buffer = [CChar](repeating: 0, count: max(n, 1))
            _ = cLastError(&buffer, buffer.count)
            throw Arch02EngineError.native(String(cString: buffer))
        }
    }
}
