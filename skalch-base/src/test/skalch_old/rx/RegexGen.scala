package test.skalch_old.rx

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

    def get_random_input(length : Int) : String = {
        var random_input_string = ""
        for (i <- 0 until length) {
            random_input_string += literals(mt.get().nextInt(literals.length))
        }
        random_input_string
    }

    def test_re(re : gnu.regexp.RE, input : String) : String = {
        Console.flush()
        val re_m = re.getMatch(input)
        if (re_m == null || re_m.getStartIndex() != 0) { null } else { re_m.toString() }
    }

    def test_rx(regex : Regex, input : String) : String = {
        Console.flush()
        val rx_m = regex.findPrefixOf(input)
        if (rx_m.isDefined) { rx_m.get } else { null }
    }

    def generate_tests(pattern : String, input_length : Int,
            ntests : Int) : Array[RegexInput] =
    {
        // println()
        // println("pattern: " + pattern)
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
                return null
            }
        }

        def next_input() = {
            var random_input_string = get_random_input(input_length)
            // println("input: " + random_input_string)
            val rx_match_result = test_rx(regex, random_input_string)
            val matchlen = if (rx_match_result == null) { -1 } else { rx_match_result.length }
            // println("rx matching: " + rx_match_result)
            new RegexInput(random_input_string, matchlen)
        }

        (for (test_idx <- 0 until ntests) yield next_input()).toArray
    }

    def main(args : Array[String]) {
        for (i <- 0 until 10) {
            val re = generate_rx(4, false)
            println("generated re: " + re.getPattern())
            generate_tests(re.getPattern(), 30, 1000)
        }
    }
}
