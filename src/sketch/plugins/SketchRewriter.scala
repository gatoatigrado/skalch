package plugins

import scala.tools.nsc
import java.io.{File, FileInputStream, FileOutputStream}
import nsc._
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import nsc.util
import scala.tools.nsc.io.{AbstractFile, PlainFile}
import scala.tools.nsc.util.{OffsetPosition, RangePosition}

class SketchRewriter(val global: Global) extends Plugin {

    val name = "sketchrewriter"
    val fname_extension = ".hints.xml"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](SketchRewriterComponent,
        FileCopyComponent)
    var scalaFileMap = Map[Object, String]()

    private object SketchRewriterComponent extends PluginComponent {
        val global = SketchRewriter.this.global
        val runsAfter = List[String]("parser");
        override val runsBefore = List[String]("namer")
        val phaseName = SketchRewriter.this.name
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            override def name = SketchRewriter.this.name

            var uid: Int = 0

            var hintsSink: java.io.StringWriter = null

            class SketchFcnApply(val uid : Int, val entire_pos : Object, val arg_pos : Object)

            def toXmlTag(name : String, x : Object, ident: String) : String = {
                ident + (x match {
                    case fcnapply : SketchFcnApply =>
                        "<sketchconstruct type=\"%s\" uid=\"%d\">\n".format(name, fcnapply.uid) +
                        toXmlTag("entire_position", fcnapply.entire_pos, ident + "    ") +
                        toXmlTag("argument_position", fcnapply.arg_pos, ident + "    ") +
                        ident + "</sketchconstruct>\n"
                    case rangepos : RangePosition => "<rangepos name=\"%s\">\n".format(name) +
                        toXmlTag("start", rangepos.focusStart, ident + "    ") +
                        toXmlTag("end", rangepos.focusEnd, ident + "    ") + ident + "</rangepos>\n"
                    case pos : OffsetPosition =>
                        "<position name=\"%s\" line=\"%d\" column=\"%d\" />\n".format(
                            name, pos.line.get, pos.column.get)
                    case _ => "<unknown toString=\"" + x.toString + "\" />\n"
                })
            }

            def apply(comp_unit: CompilationUnit) {
                Console.println("applying " + name + " to " + comp_unit.source.file.path)
                hintsSink = new java.io.StringWriter()
                comp_unit.body = CallTransformer.transform(comp_unit.body)
                val hints = hintsSink.toString()
                if(hints.length != 0) {
                    val fname = comp_unit.source.file.path + fname_extension
                    val fw = new java.io.FileWriter(fname)
                    scalaFileMap += (comp_unit -> comp_unit.source.file)
                    fw.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<document>\n")
                    fw.write(hints)
                    fw.write("</document>")
                    fw.close()
                }
            }

            def fcnName(tree : Tree) : String = tree match {
                case TypeApply(fun, args) => fun.toString()
                case _ => tree.toString()
            }

            // Rewrite calls to ?? to include a call site specific uid
            object CallTransformer extends Transformer {
                def isSketchConstruct(tree: Tree, args : List[Tree]): Boolean = {
                    val sketchConstructs = List[String]("$qmark$qmark", "$bang$bang")
                    (args.length == 1) && sketchConstructs.contains(fcnName(tree))
                }

                import scala.tools.nsc.util.FakePos

                override def transform(tree: Tree) = tree match {
                    case Apply(select, args) if isSketchConstruct(select, args) =>
                        val uidLit = Literal(uid)
                        uidLit.setPos(FakePos("Inserted literal for call to ??"))
                        uidLit.setType(ConstantType(Constant(uid)))
                        val newTree = treeCopy.Apply(tree, select, uidLit :: transformTrees(args))

                        hintsSink.write(toXmlTag(fcnName(select),
                            new SketchFcnApply(uid, tree.pos, args(0).pos), ""))
                        // print(global.genJVM.codeGenerator.

                        uid += 1
                        newTree
                    case _ => 
                        super.transform(tree)
                }
            }
        }
    }

    private object FileCopyComponent extends PluginComponent {
        val global = SketchRewriter.this.global
        val runsAfter = List("jvm")
        val phaseName = "sketch_copy_src_desc"
        def newPhase(prev: Phase) = new SketchRewriterPhase(prev)

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            import global._

            def apply(comp_unit : CompilationUnit) {
                if (!scalaFileMap.keySet.contains(comp_unit)) {
                    return
                }
                val copy_from = scalaFileMap(comp_unit)

                // copy to calculation; comp_unit.body is of type "package".
                // I'm not sure how great this is.
                val copy_to = settings.outputDirs.outputDirFor(copy_from)
                val out_dir = global.getFile(comp_unit.body.symbol, "")
                val copy_name = copy_from.name + fname_extension
                val out_file = out_dir.getAbsolutePath + File.separator + copy_name
                val in_file = copy_from.path + fname_extension
                SketchDetector.transform(comp_unit.body)

                val instream = new FileInputStream(in_file)
                val outstream = new FileOutputStream(out_file)

                val inbuf = new Array[Byte](instream.available)
                instream.read(inbuf)
                outstream.write(inbuf)
            }

            def print(x : Object*) : Unit = {
                for (v <- x) { Console.print(String.format("%20s ", "\"" + v.toString + "\"")) }
                Console.println()
            }

            object SketchDetector extends Transformer {
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

                override def transform(tree: Tree) = {
                    tree match {
                        case clsdef : ClassDef =>
                            if (is_dynamic_sketch(clsdef.symbol.info)) {
                            }
                        case _ => ()
                    }
                    super.transform(tree)
                }
            }
        }
    }
}
