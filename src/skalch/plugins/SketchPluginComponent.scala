package skalch.plugins

import scala.collection.mutable.ListBuffer
import scala.tools.nsc
import java.io.{File, FileInputStream, FileOutputStream}
import nsc._
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import nsc.util
import scala.tools.nsc.io.{AbstractFile, PlainFile}
import scala.tools.nsc.util.{FakePos, OffsetPosition, RangePosition}

/**
 * fixme -- I'm not sure if I'm using these abstract classes correctly... i don't fully
 * understand the import global._ as of now.
 * @author gatoatigrado (nicholas tung) [ntung at ntung]
 */
abstract class SketchPluginComponent(val global : Global) extends PluginComponent {
    import global._

    abstract class SketchTransformer extends Transformer {
        object ConstructType extends Enumeration {
            val Hole = Value("hole")
            val Oracle = Value("oracle")
        }

        abstract class CallType(val cons_type : ConstructType.Value)
        // FIXME - override val is bad.
        case class NewConstruct(override val cons_type : ConstructType.Value)
            extends CallType(cons_type)
        case class AssignedConstruct(override val cons_type : ConstructType.Value,
            val param_type : String) extends CallType(cons_type)

        def fcnName(tree : Tree) : String = {
            val fcn_node = tree match {
                case TypeApply(fun, args) => fun
                case _ => tree
            }
            if (fcn_node.symbol.exists) {
                fcn_node.symbol.name.toString()
            } else {
                fcn_node.toString()
            }
        }

        def applySketchType(fcn: Tree, args : List[Tree]): CallType = {
            val cons_type = fcnName(fcn) match {
                case "$qmark$qmark" => ConstructType.Hole
                case "$bang$bang" => ConstructType.Oracle
                case _ => return null
            }
            if (args.length == 1) {
                NewConstruct(cons_type)
            } else {
                var annotation : String = null
                for (a <- fcn.symbol.annotations) {
                    a match {
                        case AnnotationInfo(atp, args, assocs)
                            if atp.toString.contains("skalch.Description") =>
                                for ((name, value) <- assocs) {
                                    (name.toString, value) match {
                                        case ("value", LiteralAnnotArg(strv)) => {
                                            annotation = strv.stringValue
                                        }
                                        case _ => ()
                                    }
                                }
                        case _ => ()
                    }
                }
                AssignedConstruct(cons_type, annotation)
            }
        }

        def transformSketchClass(clsdef : ClassDef) : Tree
        def transformSketchCall(tree : Apply, ct : CallType) : Tree

        /** is a type skalch.DynamicSketch or a subtype of it? */
        def is_dynamic_sketch(tp : Type) : Boolean = tp match {
            case ClassInfoType(parents, decls, type_sym) =>
                parents.exists(is_dynamic_sketch(_))

            case TypeRef(pre, sym, args) => (pre match {
                case TypeRef(_, sym, args) => (sym.name.toString == "skalch")
                case _ => false
            }) && (sym.name.toString == "DynamicSketch")

            case _ => false
        }

        var transformCallDepth = 0
        def transformIdent() : String = {
            var result = ""
            for (i <- 0 until transformCallDepth) {
                result += "    "
            }
            result
        }

        override def transform(tree : Tree) = {
            transformCallDepth += 1
            var result : Tree = null
            tree match {
                case clsdef : ClassDef =>
                    if (is_dynamic_sketch(clsdef.symbol.info)) {
                        result = transformSketchClass(clsdef)
                    }
                case apply @ Apply(fcn, args) =>
                    val call_type = applySketchType(fcn, args)
                    if (call_type != null) {
                        result = transformSketchCall(apply, call_type)
                    }
                case _ => ()
            }
            result = (if (result == null) { super.transform(tree) } else { result })
            transformCallDepth -= 1
            result
        }
    }
}
