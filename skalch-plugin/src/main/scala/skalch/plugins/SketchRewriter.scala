package skalch.plugins

import java.lang.Integer
import java.io.{File, FileInputStream, FileOutputStream}

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import scala.tools.nsc
import nsc._
import nsc.plugins.{Plugin, PluginComponent}
import nsc.io.{AbstractFile, PlainFile}
import nsc.util.{Position, NoPosition, FakePos, OffsetPosition, RangePosition}

import ScalaDebugOut._
import sketch.util.DebugOut
import streamit.frontend.nodes
import streamit.frontend.nodes.scala._

/*
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.cli
*/

/**
 * Scala plugin for sketching support. See "FIRST TASK", "SECOND TASK", etc. below
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
class SketchRewriter(val global: Global) extends Plugin {
    val name = "sketchrewriter"
    val fname_extension = ".hints.xml"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](ConstructRewriter,
        FileCopyComponent
        , SketchGeneratorComponent
        )
    var scalaFileMap = Map[Object, XmlDoc]()
    val fake_pos = FakePos("Inserted literal for call to sketch construct")

    case class ConstructFcn(val type_ : String, val uid : Int,
            val entire_pos : Object, val arg_pos : Object)
    {
        var parameter_type : String = "undefined"
    }
    case class XmlDoc(var cons_fcn_arr : List[ConstructFcn])

    object SketchXMLFormatter extends XmlFormatter {
        def formatObjectInner(x : Object) = x match {
            case XmlDoc(cons_fcn_arr) => ("document", List(),
                (for (construct <- cons_fcn_arr)
                    yield ("construct_" + construct.uid.toString, construct)).toList)
            case construct_fcn : ConstructFcn => (construct_fcn.type_,
                List(("uid", construct_fcn.uid.toString),
                    ("param_type", construct_fcn.parameter_type.toString)),
                List(("entire_pos", construct_fcn.entire_pos),
                    ("arg_pos", construct_fcn.arg_pos)))
            case rangepos : RangePosition => ("rangepos", List(),
                List(("start", rangepos.focusStart), ("end", rangepos.focusEnd)))
            case offsetpos : OffsetPosition => ("position", List(
                ("line", offsetpos.line.toString),
                ("column", offsetpos.column.toString)), List())
            case _ => ("unknown", List(("stringrep", x.toString)), List())
        }
    }

    /**
     * FIRST TASK - BEFORE THE TYPE SYSTEM.
     * Rewrite ??(arglist) to ??(uid, arglist)
     */
    private object ConstructRewriter extends SketchPluginComponent(global) {
        import global._
        val runsAfter = List[String]("parser");
        override val runsBefore = List[String]("namer")
        val phaseName = SketchRewriter.this.name
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def name = SketchRewriter.this.name
            var hintsSink: List[ConstructFcn] = null

            def apply(comp_unit: CompilationUnit) {
                hintsSink = List()
                comp_unit.body = CallTransformer.transform(comp_unit.body)
                if (!hintsSink.isEmpty) {
                    scalaFileMap += (comp_unit -> XmlDoc(hintsSink))
                }
                hintsSink = null
            }

            // Rewrite calls to ?? to include a call site specific uid
            object CallTransformer extends SketchTransformer {
                def zeroLenPosition(pos : Position) : RangePosition = {
                    val v = pos match {
                        case rp : RangePosition => pos.end - 1
                        case _ => pos.point - 1
                    }
                    new RangePosition(pos.source, v, v, v)
                }

                def transformSketchClass(clsdef : ClassDef) = null
                def transformSketchCall(tree : Apply, ct : CallType) = {
                    val uid = hintsSink.length
                    val uidLit = Literal(uid)
                    uidLit.setPos(fake_pos)
                    uidLit.setType(ConstantType(Constant(uid)))
                    val type_ = ct.cons_type match {
                        case ConstructType.Hole => "holeapply"
                        case ConstructType.Oracle => "oracleapply"
                    }
                    assert(hintsSink != null, "internal err - CallTransformer - hintsSink null")
                    // make a fake 0-length position for the argument position
                    val arg_pos = (if (tree.args.length == 0) { zeroLenPosition(tree.pos) }
                        else { tree.args(0).pos })
                    hintsSink = hintsSink ::: List(new ConstructFcn(
                        type_, uid, tree.pos, arg_pos))
                    treeCopy.Apply(tree, tree.fun, uidLit :: transformTrees(tree.args))
                }
            }
        }
    }

    /**
     * SECOND TASK - AFTER JVM.
     * Write out XML hints files -- info about constructs and source files.
     */
    private object FileCopyComponent extends SketchPluginComponent(global) {
        val runsAfter = List("jvm")
        val phaseName = "sketch_copy_src_desc"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            var processed = List[String]()
            var processed_no_holes = List[String]()

            def prefixLen(x0 : String, x1 : String) =
                (x0 zip x1).prefixLength((x : (Char, Char)) => x._1 == x._2)

            def stripPrefix(arr : List[String], other : List[String])
                : List[String] =
            {
                arr match {
                    case head :: next :: tail =>
                        val longestPrefix = (for (v <- (arr ::: other))
                            yield prefixLen(v, head)).min
                        (for (v <- arr) yield v.substring(longestPrefix)).toList
                    case _ => arr
                }
            }

            def join_str(sep : String, arr : List[String]) = arr match {
                case head :: tail => (head /: tail)(_ + sep + _)
                case _ => ""
            }

            override def run() {
                currentRun.units foreach applyPhase
                val processed_noprefix = stripPrefix(processed, processed_no_holes)
                val processed_no_holes_noprefix = stripPrefix(processed_no_holes,
                    processed)
                println("sketchrewriter processed: " + join_str(", ", processed_noprefix))
                println("sketchrewriter processed, no holes: " +
                    join_str(", ", processed_no_holes_noprefix))
            }

            def apply(comp_unit : CompilationUnit) {
                if (!scalaFileMap.keySet.contains(comp_unit)) {
                    processed_no_holes ::= comp_unit.source.file.path
                    return
                } else {
                    processed ::= comp_unit.source.file.path
                }

                val xmldoc = scalaFileMap(comp_unit)
                val out_dir = global.getFile(comp_unit.body.symbol, "")
                val out_dir_path = out_dir.getCanonicalPath.replaceAll("<empty>$", "")
                val copy_name = comp_unit.source.file.name + fname_extension
                val out_file = (new File(out_dir_path +
                    File.separator + copy_name)).getCanonicalPath

                var sketchClasses = ListBuffer[Symbol]()
                (new SketchDetector(sketchClasses, xmldoc)).transform(comp_unit.body)
                for (cls_sym <- sketchClasses) {
                    val cls_out_file = global.getFile(cls_sym, "") + ".info"
                    (new FileOutputStream(cls_out_file)).write("%s\n%s".format(
                        out_file, comp_unit.source.file.path).getBytes)
                }

                // NOTE - the tranformer now also adds info about which construct
                // was used.
                val xml : String = SketchXMLFormatter.formatXml(xmldoc)
                (new FileOutputStream(out_file)).write(xml.getBytes())
            }

            class SketchDetector(val sketchClasses : ListBuffer[Symbol],
                    val xmldoc : XmlDoc) extends SketchTransformer
            {
                def transformSketchClass(clsdef : ClassDef) = {
                    sketchClasses += clsdef.symbol
                    null
                }

                def transformSketchCall(tree : Apply, ct : CallType) = {
                    ct match {
                        case AssignedConstruct(cons_type, param_type) => {
                            val uid = tree.args(0) match {
                                case Literal(Constant(v : Int)) => v
                                case _ => -1
                            }
                            assert(param_type != null,
                                "please set annotations for call " + tree.toString())
                            xmldoc.cons_fcn_arr(uid).parameter_type = param_type
                        }
                        case _ => assert(false, "INTERNAL ERROR - NewConstruct after jvm")
                    }
                    null
                }
            }
        }
    }



    /**
     * THIRD TASK - ALSO AFTER JVM.
     * Generate the SKETCH AST and dump it via xstream.
     */
    private object SketchGeneratorComponent extends SketchPluginComponent(global) {
        import global._
        val runsAfter = List("jvm")
        val phaseName = "sketch_static_ast_gen"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def run {
                try {
                    scalaPrimitives.init
                    super.run
                } catch {
                    case e : java.lang.Exception => DebugOut.print_exception(
                        "skipping generating SKETCH AST's due to exception", e)
                }
            }

            def apply(comp_unit : CompilationUnit) {
                val ast_gen = new SketchAstGenerator(comp_unit)
                ast_gen.transform(comp_unit.body)
                (new CheckVisited(ast_gen.visited)).transform(comp_unit.body)
            }

            class SketchAstGenerator(comp_unit : CompilationUnit) extends Transformer {
                val visited = new HashSet[Tree]()
                var root : Object = null
                val name_string_factory = new SketchNames.NameStringFactory(false)

                val connectors = ListBuffer[AutoNodeConnector[Symbol]]()
                def AutoNodeConnector(x : String) = {
                    val result = new AutoNodeConnector[Symbol](x)
                    connectors += result
                    result
                }
                val _goto_connect = AutoNodeConnector("scala_goto_label")
                val _class_connect = AutoNodeConnector("scala_class")
                val _class_fcn_connect = AutoNodeConnector("scala_class_fcn")

                /**
                 * The main recursive call to create SKETCH nodes.
                 */
                def getSketchAST(tree : Tree, info : ContextInfo) : nodes.base.FEAnyNode = {
                    // TODO - fill in source values...
                    val start = tree.pos.focusStart match {
                        case NoPosition => (0, 0)
                        case t : Position => (t.line, t.column)
                    }
                    val end = tree.pos.focusEnd match {
                        case NoPosition => (0, 0)
                        case t : Position => (t.line, t.column)
                    }
                    val the_ctx : misc.ScalaFENode = new misc.ScalaFENode(
                        comp_unit.source.file.path,
                        start._1, start._2, end._1, end._2)



                    // move code out of this class, it's already 300 lines! ick. --ntung
                    val _glbl : SketchGeneratorComponent.this.global.type = global

                    object sketch_types extends {
                        val global : SketchGeneratorComponent.this.global.type = _glbl
                        val ctx = the_ctx
                    } with SketchTypes

                    object sketch_node_map extends {
                        val global : SketchGeneratorComponent.this.global.type = _glbl
                        val types = sketch_types
                        val ctx = the_ctx

                        val goto_connect = _goto_connect
                        val class_connect = _class_connect
                        val class_fcn_connect = _class_fcn_connect
                    } with ScalaSketchNodeMap {
                        def getname(elt : Object, sym : Symbol) : String = {
                            name_string_factory.get_name(elt match {
                                case name : Name =>
                                    def alternatives(idx : Int, verbosity : Int) = verbosity match {
                                        case 0 => name.toString
                                        case 1 => sym.owner.simpleName + "." + sym.simpleName
                                        case 2 => sym.fullNameString
                                        case 3 => name.toString + "_" + idx.toString
                                    }
                                    SketchNames.LogicalName(name.toString, sym.fullNameString,
                                        if (name.isTypeName) "type" else "term",
                                        _ + "_t", alternatives)
                                case t : Tree => subtree(t) match {
                                    case subtree =>
                                        DebugOut.not_implemented("getname(tree)", tree, elt, subtree)
                                        null
                                }
                            })
                        }
                        def getname(elt : Object) : String = getname(elt, tree.symbol)
                        def gettype(tpe : Type) : nodes.Type = sketch_types.gettype(tpe)
                        def gettype(tree : Tree) : nodes.Type = gettype(tree.tpe)

                        def subtree(tree : Tree, next_info : ContextInfo = null)
                            : nodes.base.FEAnyNode =
                        {
                            val rv = getSketchAST(tree,
                                if (next_info == null) (new ContextInfo(info)) else next_info)
                            DebugOut.print(info.ident + ".")
                            rv
                        }
                        def subarr(arr : List[Tree]) =
                            (for (elt <- arr) yield subtree(elt)).toArray
                    }



                    if (visited != EmptyTree && visited.contains(tree)) {
                        DebugOut.print("warning: already visited tree", tree)
                    }
                    visited.add(tree)

                    val tree_str = tree.toString.replace("\n", " ")
                    DebugOut.print(info.ident +
                        "SKETCH AST translation for Scala tree", tree.getClass)
                    DebugOut.print(info.ident + tree_str.substring(
                        0, Math.min(tree_str.length, 60)))

                    sketch_node_map.execute(tree, info)
                }

                override def transform(tree : Tree) : Tree = {
                    root = getSketchAST(tree, new ContextInfo(null))
                    for (connector <- connectors) {
                        connector.checkDone()
                    }
                    tree
                }
            }

            /**
             * Make sure we're handling all of the nodes in the Scala AST.
             */
            class CheckVisited(transformed : HashSet[Tree]) extends Transformer {
                override def transform(tree : Tree) : Tree = {
                    if (!transformed.contains(tree)) {
                        println("WARNING - didn't transform node\n    " + tree.toString())
                    }
                    super.transform(tree)
                }
            }
        }
    }
}
