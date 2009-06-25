package plugins

import scala.tools.nsc
import nsc._
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import nsc.util

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

            def apply(unit: CompilationUnit) {
                hintsSink = new java.io.StringWriter()
                unit.body = CallTransformer.transform(unit.body)
                val hints = hintsSink.toString()
                if(hints.length != 0) {
                    val fw = new java.io.FileWriter(unit.source.file.path + ".hints")
                    fw.write(hints)
                    fw.close()
                }
            }

            // Rewrite calls to ?? to include a call site specific uid
            object CallTransformer extends Transformer {
                
                def isSketchConstruct(tree: Tree): Boolean = {
                    val sketchConstructs = List[String]("$qmark$qmark", "$bang$bang")

                    return tree.toString.endsWith("$qmark$qmark")

                    for(constructName <- sketchConstructs) {
                        if(tree.toString.endsWith(constructName)) {
                            return true
                        }
                    }

                    return false
                }

                import scala.tools.nsc.util.FakePos

                override def transform(tree: Tree) = tree match {
                    case Apply(select, args) if isSketchConstruct(select) && args.length == 1 =>
                        val uidLit = Literal(uid)
                        uidLit.setPos(FakePos("Inserted literal for call to ??"))
                        uidLit.setType(ConstantType(Constant(uid)))
                        val newTree = treeCopy.Apply(tree, select, uidLit :: transformTrees(args))

                        hintsSink.write(select.toString + " " + uid + " " + tree.pos.line.get + " " + tree.pos.column.get + "\n")
                        uid += 1
                        newTree
                    case _ => 
                        super.transform(tree)
                }

            }
        }
    }
}
