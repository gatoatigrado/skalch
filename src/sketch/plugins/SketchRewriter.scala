package plugins

import scala.tools.nsc
import nsc._
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent

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

            def apply(unit: CompilationUnit) {
                unit.body = CallTransformer.transform(unit.body)
            }

            // Rewrite calls to ?? to include a call site specific uid
            object CallTransformer extends Transformer {

                import scala.tools.nsc.util.FakePos

                override def transform(tree: Tree) = tree match {
                    case Apply(select, args) if select.toString.endsWith("$qmark$qmark") && args.length == 1 =>
                        val uidLit = Literal(uid)
                        uidLit.setPos(FakePos("Inserted literal for call to ??"))
                        uidLit.setType(ConstantType(Constant(uid)))
                        uid += 1
                        val newTree = treeCopy.Apply(tree, select, uidLit :: transformTrees(args))
                        Console.println("Rewrite: " + tree + " => " + newTree)
                        newTree
                    case _ => 
                        super.transform(tree)
                }

            }
        }
    }
}
