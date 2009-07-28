package skalch.plugins

import java.lang.Integer
import java.io.{File, FileInputStream, FileOutputStream}

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import scala.tools.nsc
import nsc._
import nsc.plugins.{Plugin, PluginComponent}
import nsc.io.{AbstractFile, PlainFile}
import nsc.util.{FakePos, OffsetPosition, RangePosition}

import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.cli

class SketchRewriter(val global: Global) extends Plugin {

    val name = "sketchrewriter"
    val fname_extension = ".hints.xml"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](ConstructRewriter,
        FileCopyComponent, SketchGeneratorComponent)
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
                def zeroLenPosition(pos : Object) : RangePosition = pos match {
                    case rp : RangePosition => new RangePosition(
                        rp.source0, rp.end - 1, rp.end - 1, rp.end - 1)
                    case OffsetPosition(src, off) => {
                        println("note - range positions are not being used.")
                        new RangePosition( src, off - 1, off - 1, off - 1)
                    }
                    case _ => assert(false, "please enable range positions"); null
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
                    assert(hintsSink != null, "internal err - CallTransformer - hintsSink null");
                    // make a fake 0-length position
                    val arg_pos = (if (tree.args.length == 0) { zeroLenPosition(tree.pos) }
                        else { tree.args(0).pos })
                    hintsSink = hintsSink ::: List(new ConstructFcn(type_, uid, tree.pos, arg_pos))
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
                        case _ => assert(false, "INTERNAL ERROR - NewConstruct after jvm")
                    }
                    null
                }
            }
        }
    }



    /**
     * Generate the SKETCH AST and dump it via xstream.
     * Currently using Skalch to determine which fields are used.
     */
    private object SketchGeneratorComponent extends SketchPluginComponent(global) {
        val runsAfter = List("jvm")
        val phaseName = "sketch_static_ast_gen"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            var query_table : (Int => Boolean) = null

            // this is persistent across backtracking invocations.
            var names_map = new HashMap[(String, String), Int]()
            // these are not
            var forbidden_objects = new HashSet[Int]()
            var visit_queue = new HashMap[Int, (String, String)]()
            var failed : Boolean = false

            def field_uid(cls_name : String, field_name : String) : Int = {
                val key = (cls_name, field_name)
                val result = names_map.get(key)
                if (!result.isDefined) {
                    val result = names_map.size
                    names_map.put(key, result)
                    result
                } else {
                    result.get
                }
            }

            def solution_string() : String = {
                var result = ""
                for ( ((clsname, fldname), uid) <- names_map ) {
                    result += clsname + "." + fldname + " = " + query_table(uid) + "\n"
                }
                result
            }

            class OrderSketch() extends DynamicSketch {
                def dysketch_main() : Boolean = {
                    failed = false
                    forbidden_objects.clear
                    visit_queue.clear
                    query_table = (??(_, 2) == 1)
                    currentRun.units foreach applyPhase
                    skdprint(solution_string())
                    skdprint("visit queue: " + visit_queue.toString)
                    println("sketch done")
                    !failed
                }

                val test_generator = NullTestGenerator
            }

            override def run() {
                val args = Array("--sy_num_threads", "1", "--sy_num_solutions", "1", "--ui_no_gui")
                val cmdopts = new cli.CliParser(args)
                BackendOptions.add_opts(cmdopts)
                skalch.synthesize(() => new OrderSketch())
            }

            def apply(comp_unit : CompilationUnit) {
                (new SketchAstGenerator()).transform(comp_unit.body)
            }

            class SketchAstGenerator() extends SketchTransformer {
                def transformSketchClass(clsdef : ClassDef) = null
                def transformSketchCall(tree : Apply, ct : CallType) = null

                override def transform(tree : Tree) : Tree = {
                    val tree_ptr = System.identityHashCode(tree)
                    visit_queue remove tree_ptr
                    if (forbidden_objects contains tree_ptr) {
                        println("failed...")
                        failed = true
                        tree
                    } else if (failed) {
                        tree
                    } else {
                        val clazz = tree.getClass()
                        for (fld <- clazz.getDeclaredFields()) {
                            fld.setAccessible(true)
                            val uid = field_uid(clazz.getCanonicalName, fld.getName)
                            val fld_obj = fld.get(tree)
                            val field_ptr = System.identityHashCode(fld_obj)
                            if (!query_table(uid)) {
                                // println("adding forbidden field " + clazz.getCanonicalName + ", " + fld.getName)
                                forbidden_objects add field_ptr
                            } else {
                                visit_queue.put(field_ptr, (clazz.getCanonicalName, fld.getName))
                            }
                        }
                        SketchNodes.get_sketch_class(tree.getClass, tree)
                        super.transform(tree)
                    }
                }

            }
        }
    }
}
