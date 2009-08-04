package test.skalch_old.rx

import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class RegexSketch(input_rx : String, input_length : Int, num_inputs : Int)
    extends DynamicSketch
{
    val in_result = new InputGenerator(untilv=input_length)
    val in_string = new InputGenerator(untilv=3)
    class NFANode(val idx : Int) {
        // each entry is a bit vector, with every bit representing
        // whether the edge is taken
        val transitions = new HoleArray(3, 1 << 2)
        val state_mask = (1 << idx)
        def is_active() : Boolean = (state & state_mask) > 0
        var is_final = new BooleanHole()
    }
    val nodes = (for (i <- 0 until 2) yield new NFANode(i)).toArray
    var state = 0

    def transition(input : Int) = {
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

    def dysketch_main() : Boolean = {
        for (input_idx <- 0 until num_inputs) {
            skdprint_loc("new input")
            state = 1
            var longest_match = if (any_final()) { 0 } else { -1 }
            // skdprint("first is_final uid: " + nodes(0).is_final.hole.uid)
            for (characters_consumed <- 1 to input_length) {
                skdprint("state: " + state)
                state = transition(in_string())
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
    object TestOptions extends cli.CliOptionGroup {
        import java.lang.Integer
        add("--input_rx", "", "operations to process")
        add("--input_length", 10 : Integer, "number of tests")
        add("--num_inputs", 1 : Integer,
            "number of inputs to test regex with")
    }

    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new RegexSketch(
            opts.str_("input_rx"),
            opts.long_("input_length").intValue,
            opts.long_("num_inputs").intValue
            ))
    }
}

