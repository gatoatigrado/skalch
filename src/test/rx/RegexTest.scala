package test.rx

import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class RegexSketch(input_rx : String, input_length : Int, num_inputs : Int)
    extends DynamicSketch
{
    val in_result = new InputGenerator(untilv=input_length)
    val in_string = new InputGenerator(untilv=3)
    class NFANode {
        // each entry is a bit vector, with every bit representing
        // whether the edge is taken
        val transitions = new HoleArray(3, 1 << 16)
        var is_active = false
        var is_final = new BooleanHole()
    }
    val nodes = (for (i <- 0 until 16) yield new NFANode()).toArray

    def transition(previous : Int, input : Int) = {
        var next_state = 0
        for (node <- nodes if (node.is_active)) {
            next_state |= node.transitions(input)
        }
        next_state
    }

    def any_final() : Boolean = {
        for (node <- nodes if (node.is_active && node.is_final())) {
            return true
        }
        false
    }

    def nfa_reset() {
        for (node <- nodes) { node.is_active = false }
        nodes(0).is_active = true
    }

    def dysketch_main() : Boolean = {
        for (input_idx <- 0 until num_inputs) {
            skdprint_loc("new input")
            nfa_reset()
            var state = 0
            var longest_match = if (any_final()) { 0 } else { -1 }
            for (characters_consumed <- 1 to input_length) {
                state = transition(state, in_string())
                if (any_final()) {
                    skdprint("final at index = " + characters_consumed)
                    longest_match = characters_consumed
                }
            }
            val test_result = in_result()
            skdprint("longest match: " + longest_match)
            skdprint("test result: " + test_result)
            if (longest_match != test_result) {
                return false
            }
        }
        true
    }

    val test_generator = new TestGenerator {
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set() {
            val tests = RegexGen.generate_tests(input_rx, input_length, num_inputs)
            for (test <- tests) {
                put_input(in_result, test.result)
                for (in_char <- test.input) {
                    val input_value = in_char match {
                        case 'a' => 0
                        case 'b' => 1
                        case 'c' => 2
                    }
                    put_input(in_string, input_value)
                }
            }
        }

        def tests() { test_case() }
    }
}

object RegexTest {
    object TestOptions extends CliOptGroup {
        import java.lang.Integer
        add("--input_rx", "", "operations to process")
        add("--input_length", 10 : Integer, "number of tests")
        add("--num_inputs", 1 : Integer,
            "number of inputs to test regex with")
    }

    def main(args: Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new RegexSketch(
            opts.str_("input_rx"),
            opts.long_("input_length").intValue,
            opts.long_("num_inputs").intValue
            ))
    }
}

