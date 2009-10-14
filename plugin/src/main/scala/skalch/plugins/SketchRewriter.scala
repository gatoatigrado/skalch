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
import sketch.compiler.ast.{base, core, scala => scast}
import net.sourceforge.gxl._

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
    val gxl_extension = ".ast.gxl"
    val description = "de-sugars sketchy constructs"
    val components : List[PluginComponent] = List[PluginComponent](
        ConstructRewriter, FileCopyComponent, GxlGeneratorComponent)
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
        val phaseName = "sketch_construct_rewriter"
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
     * SECOND TASK - AFTER icode.
     * Write out XML hints files -- info about constructs and source files.
     */
    private object FileCopyComponent extends SketchPluginComponent(global) {
        val runsAfter = List("icode")
        val phaseName = "sketch_copy_src_desc"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            var processed = List[String]()
            var processed_no_holes = List[String]()

            def prefixLen(x0 : String, x1 : String) : Int = {
                for (i <- 0 until Math.min(x0.length, x1.length)) {
                    if (x0.charAt(i) != x1.charAt(i)) {
                        return i
                    }
                }
                return 0
            }
            // broken by Scala trunk
//                 (x0 zip x1).prefixLength((x : (Char, Char)) => x._1 == x._2)

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
                val out_file = new File(comp_unit.source.file.path + fname_extension)
                val gxlout = new File(comp_unit.source.file.path + gxl_extension)

                var sketchClasses = ListBuffer[Symbol]()
                (new SketchDetector(sketchClasses, xmldoc)).transform(comp_unit.body)
                for (cls_sym <- sketchClasses) {
                    val cls_out_file = global.getFile(cls_sym, "") + ".info"
                    (new FileOutputStream(cls_out_file)).write("%s\n%s\n%s".format(
                        out_file, comp_unit.source.file.path, gxlout).getBytes)
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
                        case _ => assert(false, "INTERNAL ERROR - NewConstruct after icode")
                    }
                    null
                }
            }
        }
    }



    /**
     * THIRD TASK - ALSO AFTER icode.
     * Generate the SKETCH AST and dump it via xstream.
     */
    private object GxlGeneratorComponent extends SketchPluginComponent(global) {
        val _glbl : global.type = global
        import global._

        val runsAfter = List("icode")
        val phaseName = "sketch_static_ast_gen"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        object gxl_node_map extends {
            val _glbl : global.type = global
        } with ScalaGxlNodeMap

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def run {
                scalaPrimitives.init
                super.run
                for (v <- gxl_node_map.nf.new_node_names) {
                    println("node class " + v + ";")
                }
                for (v <- gxl_node_map.nf.new_edge_names) {
                    println("edge class " + v + ";")
                }
            }

            def apply(comp_unit : CompilationUnit) {
                val gxlout = new File(comp_unit.source.file.path + gxl_extension)
                val ast_gen = new GxlAstGenerator(comp_unit, gxlout)
                gxl_node_map.nf.sourceFile = comp_unit.source.file.path
                ast_gen.transform(comp_unit.body)
                println("WARNING -- please re-enable visitor checks sometime")
//                 (new CheckVisited(gxl_node_map.visited)).transform(comp_unit.body)
            }

            class GxlAstGenerator(comp_unit : CompilationUnit, gxlout : File)
                extends Transformer
            {
                gxl_node_map.visited = new HashSet[Tree]()
                var root : Object = null

                override def transform(tree : Tree) : Tree = {
                    import GxlViews.{xmlwrapper, arrwrapper, nodewrapper}
                    val rcurl = getClass().getResource("/skalch/plugins/type_graph.gxl")
                    assert(rcurl != null, "resource type_graph.gxl not in jar package!")
                    val gxldoc = new GXLDocument(rcurl)
                    val gxlroot = gxldoc.getDocumentElement()
                    val graphs : Array[GXLGraph] = (for (i <- 0 until gxlroot.getGraphCount)
                        yield gxlroot.getGraphAt(i)).toArray
                    val typ_graph = graphs.get_by_attr("id" -> "SCE_ScalaAstModel")
                    val def_graph = graphs.get_by_attr("id" -> "DefaultGraph")

                    // gxl type graph edges are really nodes
                    // kinda ugly but not hard to fix
                    val node_types = typ_graph.getnodes()
                    val real_nodes = node_types.filtertype(
                        "http://www.gupro.de/GXL/gxl-1.0.gxl#NodeClass")
                    val edges = node_types.filtertype(
                        "http://www.gupro.de/GXL/gxl-1.0.gxl#EdgeClass")

                    gxl_node_map.nf.node_ids = real_nodes map (_.getAttribute("id"))
                    gxl_node_map.nf.edge_ids = edges map (_.getAttribute("id"))

                    root = gxl_node_map.getGxlAST(tree)

                    println("writing " + gxlout)
                    gxldoc.write(gxlout)
                    tree
                }
            }

            /**
             * Make sure we're handling all of the nodes in the Scala AST.
             */
            class CheckVisited(transformed : HashSet[Tree]) extends Transformer {
                var parentCls : String = null
                override def transform(tree : Tree) : Tree = tree match {
                    case New(tpt) => tree
                    case TypeTree() => tree
                    case _ =>
                        if (!transformed.contains(tree)) {
                            println("WARNING - didn't transform node    " + tree.getClass() +
                                "; parent class " + parentCls)
                            println(tree.toString + "\n")
                        }
                        val prev_parent : String = parentCls
                        parentCls = tree.getClass().toString
                        val rv = super.transform(tree)
                        parentCls = prev_parent
                        rv
                }
            }
        }
    }
}
