package plugins

import scala.tools.nsc
import nsc._
import nsc.Global
import nsc.Phase
import nsc.plugins.Plugin
import nsc.plugins.PluginComponent
import nsc.transform._
import nsc.symtab.Flags._


class SketchRewriter(val global: Global) extends Plugin {
    import global._

    val name = "sketchrewriter"
    val description = "de-sugars sketchy constructs"
    val components = List[PluginComponent](SketchRewriterComponent)

    private object SketchRewriterComponent extends PluginComponent {
        val global: SketchRewriter.this.global.type = SketchRewriter.this.global
        val runsAfter = List[String]("namer");
        val phaseName = SketchRewriter.this.name
        def newPhase(_prev: Phase) = new SketchRewriterPhase(_prev)    

        class SketchRewriterPhase(prev: Phase) extends StdPhase(prev) {
            override def name = SketchRewriter.this.name

            var uid: Int = 0

            def apply(unit: CompilationUnit) {
                Console.println("I'm applying!")
                unit.body = CallTransformer.transform(unit.body)
            }

            object CallTransformer extends Transformer {

                import scala.tools.nsc.util.FakePos

                override def transform(tree: Tree) = tree match {
                    case a @ Apply(select, args) if select.toString.endsWith("??") && args.length == 1 =>
                        val newArg = Literal(uid)
                        newArg.setPos(FakePos("Inserted literal for call to ??"))
                        newArg.setType(ConstantType(Constant(uid)))
                        uid += 1
                        /*
                        // The following attempts to set the correct method
                        // type for the new call to ??, adding the extra argument
                        // however, for reasons currently beyond my understanding
                        // it makes everything blow up
                        // for now, the code *almost works*, but since the type is wrong,
                        // the second argument to ?? is ignored
                        val newMethodType = MethodType(List[Symbol](definitions.IntClass, definitions.IntClass), definitions.IntClass.tpe)
                        select.symbol.setInfo(newMethodType)
                        select.setType(newMethodType)
                        Console.println(newMethodType)
                        Console.println(select.symbol.infosString)
                        */
                        var newTree = a.copy(args = newArg :: transformTrees(args))
                        newTree.setType(a.tpe)
                        Console.println("Rewrite: " + a + " => " + newTree)
                        newTree
                    case _ => 
                        super.transform(tree)
                }

            }
        }
    }
}
