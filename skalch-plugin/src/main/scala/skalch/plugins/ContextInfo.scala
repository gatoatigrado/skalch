package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import sketch.compiler.ast.{base, core, scala => scast}

/**
 * support early creation styles, where the sketch node is created
 * before its children are traversed. used for classes, which are
 * directly lowered into global functions.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
class ContextInfo(val old : ContextInfo) {
    var curr_clazz : scast.typs.ScalaClass =
        if (old != null) old.curr_clazz else null
    var ident : String = if (old == null) "" else ("    " + old.ident)
}
