package test.rx

import ec.util.ThreadLocalMT
import scala.util.matching.Regex

/** generate regular expressions */
object RegexGen {
    val mt = new ThreadLocalMT()
    val literals = Array('a', 'b', 'c')

    // doesn't have to be fast...
    abstract class RegexNode(val cons_type : String) {
        def getPattern(f : (RegexNode => String)) : String
        def getPattern() : String = getPattern(_.getPattern())
        override def toString() = " " + getPattern(_.toString()) + " "
    }

    // for now, these are integers "0", "1", and "2"
    case class LazyStarNode(val inner : RegexNode) extends RegexNode("node") {
        def getPattern(f : (RegexNode => String)) = "(" + f(inner) + "*?)"
    }
    case class StarNode(val inner : RegexNode) extends RegexNode("node") {
        def getPattern(f : (RegexNode => String)) = "(" + f(inner) + "*)"
    }
    case class MaybeNode(val inner : RegexNode) extends RegexNode("node") {
        def getPattern(f : (RegexNode => String)) = "(" + f(inner) + "?)"
    }

    // these classes are integers "3" and "4"
    case class ParenNode(val seq : Array[RegexNode]) extends RegexNode("list") {
        def getPattern(f : (RegexNode => String)) =
            "(" + ("" /: seq)((x, y) => x + f(y)) + ")"
    }
    case class SwitchNode(val options : Array[RegexNode]) extends RegexNode("list") {
        def getPattern(f : (RegexNode => String)) =
            "(" + ("" /: options)((x, y) => (if (x.isEmpty) "" else x + "|") + f(y)) + ")"
    }

    // this is created when there is no budget.
    case class LiteralNode(val lit : Char) extends RegexNode("literal") {
        def getPattern(f : (RegexNode => String)) = lit.toString
    }

    def generate_rx(budget : Int) : RegexNode = {
        if (budget <= 0) {
            // must create a literal
            LiteralNode(literals(mt.get().nextInt(literals.length)))
        } else {
            val nodeType : Int = mt.get.nextInt(4)
            val listLength = (if (nodeType <= 1) 1 else mt.get.nextInt(3) + 2)
            val list = (for (i <- 0 until listLength)
                yield generate_rx(mt.get.nextInt(budget))).toArray
            nodeType match {
                case 0 => LazyStarNode(list(0))
                case 1 => StarNode(list(0))
                case 2 => ParenNode(list)
                case 3 => SwitchNode(list)
            }
        }
    }

    def generate_tests(node : RegexNode, ntests : Int) {
        //val regex = new Regex(node.getPattern())
    }

    def main(args : Array[String]) {
        val re = generate_rx(2)
        println("random regex")
        println("toString formatted: " + re.toString())
        println("pattern: " + re.getPattern())
    }
}
