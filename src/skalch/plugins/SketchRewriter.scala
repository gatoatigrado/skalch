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

class SketchRewriter(val global: Global) extends Plugin {

    val name = "sketchrewriter"
    val fname_extension = ".hints.xml"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](ConstructRewriter,
        FileCopyComponent)
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
                ("line", offsetpos.line.get.toString),
                ("column", offsetpos.column.get.toString)), List())
            case _ => ("unknown", List(("stringrep", x.toString)), List())
        }
    }

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
                    hintsSink = hintsSink ::: List(new ConstructFcn(type_, uid, tree.pos, tree.args(0).pos))
                    treeCopy.Apply(tree, tree.fun, uidLit :: transformTrees(tree.args))
                }
            }
        }
    }

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

            def stripPrefix(arr : List[String], other : List[String]) : List[String] = arr match {
                case head :: next :: tail =>
                    val longestPrefix = (for (v <- (arr ::: other)) yield prefixLen(v, head)).min
                    (for (v <- arr) yield v.substring(longestPrefix)).toList
                case _ => arr
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
                println("sketchrewriter processed, no holes: " + join_str(", ", processed_no_holes_noprefix))
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
                        case _ => println("INTERNAL ERROR - NewConstruct after jvm")
                    }
                    null
                }
            }
        }
    }
}
