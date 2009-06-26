package plugins

import scala.tools.nsc
import nsc._
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import nsc.util
import scala.tools.nsc.util.{OffsetPosition, RangePosition}

class SketchRewriter(val global: Global) extends Plugin {

    val name = "sketchrewriter"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](SketchRewriterComponent)

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
                    case fcnapply : SketchFcnApply => "<sketchconstruct type=\"%s\" uid=\"%d\">\n".format(
                        name, fcnapply.uid) +
                        toXmlTag("entire_position", fcnapply.entire_pos, ident + "    ") +
                        toXmlTag("argument_position", fcnapply.arg_pos, ident + "    ") +
                        ident + "</sketchconstruct>\n"
                    case rangepos : RangePosition => "<rangepos name=\"%s\">\n".format(name) +
                        toXmlTag("start", rangepos.focusStart, ident + "    ") +
                        toXmlTag("end", rangepos.focusEnd, ident + "    ") + ident + "</rangepos>\n"
                    case pos : OffsetPosition => "<position name=\"%s\" line=\"%d\" column=\"%d\" />\n".format(
                        name, pos.line.get, pos.column.get)
                    case _ => "<unknown toString=\"" + x.toString + "\" />\n"
                })
            }

            def apply(unit: CompilationUnit) {
                Console.println("applying " + name + " to " + unit.source.file.path)
                hintsSink = new java.io.StringWriter()
                hintsSink.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<document>\n")
                unit.body = CallTransformer.transform(unit.body)
                hintsSink.write("</document>")
                val hints = hintsSink.toString()
                if(hints.length != 0) {
                    val fw = new java.io.FileWriter(unit.source.file.path + ".hints.xml")
                    fw.write(hints)
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

                def print(x : Object*) : Unit = {
                    for (v <- x) { Console.print(String.format("%40s ", v.toString)) }
                    Console.println()
                }

                override def transform(tree: Tree) = tree match {
                    case Apply(select, args) if isSketchConstruct(select, args) =>
                        val uidLit = Literal(uid)
                        uidLit.setPos(FakePos("Inserted literal for call to ??"))
                        uidLit.setType(ConstantType(Constant(uid)))
                        val newTree = treeCopy.Apply(tree, select, uidLit :: transformTrees(args))

                        hintsSink.write(toXmlTag(fcnName(select),
                            new SketchFcnApply(uid, tree.pos, args(0).pos), ""))

                        uid += 1
                        newTree
                    case _ => 
                        super.transform(tree)
                }

            }
        }
    }
}
