package skalch.cuda

import skalch.AngelicSketch
import skalch.cuda.annotations._
import sketch.util.DebugOut

/**
 * Base class for a deterministic CUDA kernel
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required,
 *          if you make changes, please consider contributing back!
 */

/// would contain deref, but need to avoid specialization
abstract class Ptr

@scPointerObject("int") case class IntPtr(val deref : Int) extends Ptr {
    @scSpecialFcn def atomicAdd (
        amnt : Int @scCTypeNameOverride("int")) = 0
    @scSpecialFcn def atomicExch(
        amnt : Int @scCTypeNameOverride("int")) = 0
    @scSpecialFcn def atomicCAS (
        amnt : Int @scCTypeNameOverride("int")) = 0
    @scSpecialFcn def atomicAnd (
        amnt : Int @scCTypeNameOverride("int")) = 0
    @scSpecialFcn def atomicXor (
        amnt : Int @scCTypeNameOverride("int")) = 0
    @scSpecialFcn def atomicOr  (
        amnt : Int @scCTypeNameOverride("int")) = 0
}
@scPointerObject("unsigned long long") case class LongPtr(val deref : Long) extends Ptr {
    @scSpecialFcn def atomicAdd (
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
    @scSpecialFcn def atomicExch(
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
    @scSpecialFcn def atomicCAS (
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
    @scSpecialFcn def atomicAnd (
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
    @scSpecialFcn def atomicXor (
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
    @scSpecialFcn def atomicOr  (
        amnt : Long @scCTypeNameOverride("unsigned long long")) = 0
}
