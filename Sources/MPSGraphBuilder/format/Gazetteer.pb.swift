// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: Gazetteer.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2019, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// A model which uses an efficient probabilistic representation
/// for assigning labels to a set of strings.
struct CoreML_Specification_CoreMLModels_Gazetteer {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///
  /// Stores the revision number for the model, revision 2 is available on
  /// iOS, tvOS 13.0+, macOS 10.15+
  var revision: UInt32 = 0

  ///
  /// Stores the language of the model, as specified in BCP-47 format,
  /// e.g. "en-US". See https://tools.ietf.org/html/bcp47
  var language: String = String()

  ///
  /// Natural Lanaguge framework's efficient representation of a gazetter.
  var modelParameterData: Data = Data()

  ///
  /// Stores the set of output class labels
  var classLabels: CoreML_Specification_CoreMLModels_Gazetteer.OneOf_ClassLabels? = nil

  var stringClassLabels: CoreML_Specification_StringVector {
    get {
      if case .stringClassLabels(let v)? = classLabels {return v}
      return CoreML_Specification_StringVector()
    }
    set {classLabels = .stringClassLabels(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  ///
  /// Stores the set of output class labels
  enum OneOf_ClassLabels: Equatable {
    case stringClassLabels(CoreML_Specification_StringVector)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_CoreMLModels_Gazetteer.OneOf_ClassLabels, rhs: CoreML_Specification_CoreMLModels_Gazetteer.OneOf_ClassLabels) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.stringClassLabels, .stringClassLabels): return {
        guard case .stringClassLabels(let l) = lhs, case .stringClassLabels(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      }
    }
  #endif
  }

  init() {}
}

#if swift(>=5.5) && canImport(_Concurrency)
extension CoreML_Specification_CoreMLModels_Gazetteer: @unchecked Sendable {}
extension CoreML_Specification_CoreMLModels_Gazetteer.OneOf_ClassLabels: @unchecked Sendable {}
#endif  // swift(>=5.5) && canImport(_Concurrency)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification.CoreMLModels"

extension CoreML_Specification_CoreMLModels_Gazetteer: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Gazetteer"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "revision"),
    10: .same(proto: "language"),
    100: .same(proto: "modelParameterData"),
    200: .same(proto: "stringClassLabels"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularUInt32Field(value: &self.revision) }()
      case 10: try { try decoder.decodeSingularStringField(value: &self.language) }()
      case 100: try { try decoder.decodeSingularBytesField(value: &self.modelParameterData) }()
      case 200: try {
        var v: CoreML_Specification_StringVector?
        var hadOneofValue = false
        if let current = self.classLabels {
          hadOneofValue = true
          if case .stringClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.classLabels = .stringClassLabels(v)
        }
      }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    if self.revision != 0 {
      try visitor.visitSingularUInt32Field(value: self.revision, fieldNumber: 1)
    }
    if !self.language.isEmpty {
      try visitor.visitSingularStringField(value: self.language, fieldNumber: 10)
    }
    if !self.modelParameterData.isEmpty {
      try visitor.visitSingularBytesField(value: self.modelParameterData, fieldNumber: 100)
    }
    try { if case .stringClassLabels(let v)? = self.classLabels {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_CoreMLModels_Gazetteer, rhs: CoreML_Specification_CoreMLModels_Gazetteer) -> Bool {
    if lhs.revision != rhs.revision {return false}
    if lhs.language != rhs.language {return false}
    if lhs.modelParameterData != rhs.modelParameterData {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}