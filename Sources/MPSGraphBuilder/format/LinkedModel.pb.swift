// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: LinkedModel.proto
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
/// A model which wraps another (compiled) model external to this one
struct CoreML_Specification_LinkedModel {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var linkType: CoreML_Specification_LinkedModel.OneOf_LinkType? = nil

  /// A model located via a file system path
  var linkedModelFile: CoreML_Specification_LinkedModelFile {
    get {
      if case .linkedModelFile(let v)? = linkType {return v}
      return CoreML_Specification_LinkedModelFile()
    }
    set {linkType = .linkedModelFile(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  enum OneOf_LinkType: Equatable {
    /// A model located via a file system path
    case linkedModelFile(CoreML_Specification_LinkedModelFile)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_LinkedModel.OneOf_LinkType, rhs: CoreML_Specification_LinkedModel.OneOf_LinkType) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.linkedModelFile, .linkedModelFile): return {
        guard case .linkedModelFile(let l) = lhs, case .linkedModelFile(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      }
    }
  #endif
  }

  init() {}
}

/// Model is referenced by a model file name and search path
struct CoreML_Specification_LinkedModelFile {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// Model file name: e.g. "MyFetureExtractor.mlmodelc"
  var linkedModelFileName: CoreML_Specification_StringParameter {
    get {return _linkedModelFileName ?? CoreML_Specification_StringParameter()}
    set {_linkedModelFileName = newValue}
  }
  /// Returns true if `linkedModelFileName` has been explicitly set.
  var hasLinkedModelFileName: Bool {return self._linkedModelFileName != nil}
  /// Clears the value of `linkedModelFileName`. Subsequent reads from it will return its default value.
  mutating func clearLinkedModelFileName() {self._linkedModelFileName = nil}

  /// Search path to find the linked model file
  /// Multiple paths can be searched using the unix-style path separator ":"
  /// Each path can be relative (to this model) or absolute
  ///
  /// An empty string is the same as teh relative search path "."
  /// which searches in the same location as this model file
  ///
  /// There are some special paths which start with $
  /// - $BUNDLE_MAIN - Indicates to look in the main bundle
  /// - $BUNDLE_IDENTIFIER(identifier) - Looks in Bunde with given identifer
  var linkedModelSearchPath: CoreML_Specification_StringParameter {
    get {return _linkedModelSearchPath ?? CoreML_Specification_StringParameter()}
    set {_linkedModelSearchPath = newValue}
  }
  /// Returns true if `linkedModelSearchPath` has been explicitly set.
  var hasLinkedModelSearchPath: Bool {return self._linkedModelSearchPath != nil}
  /// Clears the value of `linkedModelSearchPath`. Subsequent reads from it will return its default value.
  mutating func clearLinkedModelSearchPath() {self._linkedModelSearchPath = nil}

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}

  fileprivate var _linkedModelFileName: CoreML_Specification_StringParameter? = nil
  fileprivate var _linkedModelSearchPath: CoreML_Specification_StringParameter? = nil
}

#if swift(>=5.5) && canImport(_Concurrency)
extension CoreML_Specification_LinkedModel: @unchecked Sendable {}
extension CoreML_Specification_LinkedModel.OneOf_LinkType: @unchecked Sendable {}
extension CoreML_Specification_LinkedModelFile: @unchecked Sendable {}
#endif  // swift(>=5.5) && canImport(_Concurrency)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_LinkedModel: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".LinkedModel"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "linkedModelFile"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try {
        var v: CoreML_Specification_LinkedModelFile?
        var hadOneofValue = false
        if let current = self.linkType {
          hadOneofValue = true
          if case .linkedModelFile(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.linkType = .linkedModelFile(v)
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
    try { if case .linkedModelFile(let v)? = self.linkType {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_LinkedModel, rhs: CoreML_Specification_LinkedModel) -> Bool {
    if lhs.linkType != rhs.linkType {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_LinkedModelFile: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".LinkedModelFile"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "linkedModelFileName"),
    2: .same(proto: "linkedModelSearchPath"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._linkedModelFileName) }()
      case 2: try { try decoder.decodeSingularMessageField(value: &self._linkedModelSearchPath) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._linkedModelFileName {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    try { if let v = self._linkedModelSearchPath {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_LinkedModelFile, rhs: CoreML_Specification_LinkedModelFile) -> Bool {
    if lhs._linkedModelFileName != rhs._linkedModelFileName {return false}
    if lhs._linkedModelSearchPath != rhs._linkedModelSearchPath {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
