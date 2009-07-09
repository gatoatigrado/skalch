package test.rx

import ec.util.ThreadLocalMT
import scala.util.matching.Regex

/** generate regular expressions */
object RegexGen {
    val mt = new ThreadLocalMT()
    val literals = Array('a', 'b', 'c')

    // doesn't have to be fast...
    abstract class RegexNode(val cons_type : String) {
        var could_be_zero = true
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
        for (v <- seq if !v.could_be_zero) { could_be_zero = false }
    }
    case class SwitchNode(val options : Array[RegexNode]) extends RegexNode("list") {
        def getPattern(f : (RegexNode => String)) =
            "(" + ("" /: options)((x, y) => (if (x.isEmpty) "" else x + "|") + f(y)) + ")"
        could_be_zero = false
        for (v <- options if v.could_be_zero) { could_be_zero = true }
    }

    // this is created when there is no budget.
    case class LiteralNode(val lit : Char) extends RegexNode("literal") {
        def getPattern(f : (RegexNode => String)) = lit.toString
        could_be_zero = false
    }

    def generate_rx(budget : Int, must_have_nonzero : Boolean) : RegexNode = {
        if (budget <= 0) {
            // must create a literal
            LiteralNode(literals(mt.get().nextInt(literals.length)))
        } else {
            val nodeType : Int = if (must_have_nonzero) { mt.get().nextInt(2) + 2 } else { mt.get().nextInt(4) }
            val listLength = (if (nodeType <= 1) 1 else mt.get.nextInt(3) + 2)
            val next_budget = mt.get.nextInt(budget)

            while (true) {
                val list = (for (i <- 0 until listLength)
                    yield generate_rx(next_budget, nodeType < 2)).toArray
                val rv = nodeType match {
                    case 0 => MaybeNode(list(0))
                    case 1 => StarNode(list(0))
                    case 2 => ParenNode(list)
                    case 3 => SwitchNode(list)
                }
                if (!rv.could_be_zero || !must_have_nonzero) {
                    return rv
                }
            }
            null
        }
    }

    def generate_tests(pattern : String, ntests : Int) {
        println()
        println("pattern: " + pattern)
        Console.flush()
        val regex = new Regex(pattern)
        var re : gnu.regexp.RE = null
        try {
            re = new gnu.regexp.RE(pattern, 0,
                gnu.regexp.RESyntax.RE_SYNTAX_POSIX_EXTENDED)
        } catch {
            case e => {
                println("invalid regex for gnu: " + pattern)
                println("error: " + e)
                return
            }
        }
        for (test_idx <- 0 until ntests) {
            var random_input_string = ""
            for (i <- 0 until 30) {
                random_input_string += literals(mt.get().nextInt(literals.length))
            }
            println("input: " + random_input_string)
            Console.flush()
            /*
            val re_m = re.getMatch(random_input_string)
            val re_mstr = if (re_m == null || re_m.getStartIndex() != 0) { null } else { re_m.toString() }
            println("re matching: " + re_mstr)
            */
            Console.flush()
            try {
                val rx_m = regex.findPrefixOf(random_input_string)
                val rx_mstr = if (rx_m.isDefined) { rx_m.get } else { null }
                println("rx matching: " + rx_mstr)
            } catch {
                case e => println(e)
            }
        }
    }

    def main(args : Array[String]) {
        for (i <- 0 until 100) {
            val re = generate_rx(4, false)
            generate_tests(re.getPattern(), 1)
        }
        // println("random regex")
        // println("pattern: " + re.getPattern())
    }
}
