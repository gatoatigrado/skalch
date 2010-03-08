package edu.berkeley.cs.stringsearch

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.cli
import sketch.util.DebugOut._ // assertFalse, not_implemented, print()

// Search for a pattern in a text array

class SearchSketch() extends AngelicSketch {
    val tests = Array( () )

    def main() {

        // val text : Array[Char] =  Array('a','b','a','a','b','a','a','b','b','a','a','b')
        val text : Array[Char] =  Array('a','b','a','a','b','a','a','b','b','a','a','b', 'b', 'a', 'a', 'b')
        // val text : Array[Char] =  Array('a','b','a','a','b','a','a','b','b','a','a','a','b', 'b', 'a', 'a', 'b')
 
        // val text : Array [Char] = Array('a','a','a','a','a','a','a','a','a','a','a','b')
        // this one is good for recovering unique tbl with 1024 vars
        
        val N = text.size

        val pattern : Array[Char] = Array('a','a','b', 'b', 'a', 'a', 'b')  // results in table 0, 0, 2, 1, 0, 0, 2
        // val pattern : Array[Char] = Array('a','b', 'b', 'a', 'a', 'b')   // results in table 0, 1, 1, 0, 2, 1
        // val pattern : Array[Char] = Array('a','b', 'b', 'a')  // results in 0, 1, 1, 0
        // val pattern : Array[Char] = Array('b','a', 'b', 'b', 'a')  // results in 0, 1, 0, 2, 1

        val M = pattern.size

        val table = new Array[Int](M)

        val prev = new Array[Int](M)

        // angelically initialize prev
        for (i <- 0 to M - 1) {
	    //            prev(i) = compute_prev(pattern, i)
            // prev(i) = !!(i)
            // synthAssert(check_pre(pattern, i,prev(i)))
	    // skdprint("Prev[" + i + "] = " + prev(i))
        }

        // angelically initialize the table
        for (i <- 0 to M - 1) {
    	    // table(i) = compute_table(pattern, prev, i)
            // table(i) = !!(i+1)
            // skdprint("Table[" + i + "] = " + table(i))
        }

        match_against_text_variations(text, pattern, table, N, M, 1) // last parameter needs to be 1 or above.
    }

    def check_pre(pattern : Array[Char], i: Int, prev : Int) : Boolean = {
        var im = i - 1
        var pm = prev - 1
        while (pm >= 0) {
	    if (pattern(pm) != pattern(im)) return false;
            pm = pm - 1
            im = im - 1
        }
        return true;
    }

    def compute_prev(pattern : Array[Char], i : Int) : Int = {
        print("Compute Prev(" + i + ")");
        if (i <= 1) return 0;
        var y = i - 1
        while (y > 0) {
            val f = compute_prev(pattern, y)
            if (pattern(f+1) == pattern(i))
                return f+1;
            y = f;
		}
        return 0;
    }

    def compute_table(pattern : Array[Char], prev : Array[Int], pf : Int) : Int = {
        skdprint("Compute Table(" + pf + ")")
        if (pf == 0) return 0;
        val p = prev(pf)
        if (pattern(p) != pattern(pf))
	    return p + 1;
        else
            return compute_table(pattern, prev, p)
    }

    def do_match(text: Array[Char], pattern: Array[Char], table : Array[Int], N: Int, M: Int, pos : Int) {
        val found_pos = retracting_pos_find(text, pattern, table, N, M);
        synthAssert(found_pos == pos)
    }


    def do_no_match(text: Array[Char], pattern: Array[Char], table : Array[Int], N: Int, M: Int) {
        val found_pos = retracting_pos_find(text, pattern, table, N, M);
        synthAssert(found_pos < 0)
    }

    def seq_next(text : Array[Char], N : Int, i : Int) {
        // if (i == N) return;
        if (text(i) == 'a') {
	    text(i) = 'b';
        } else {
            text(i) = 'a';
            seq_next(text, N, i + 1 % N);
        }
    }

    def match_against_text_variations(text: Array[Char], pattern: Array[Char], table: Array[Int], N: Int, M: Int, vars : Int) {
        // match pattern against text and its mutations
	for (i <- 0 to vars-1) {
            val pos = iterative_pos_find(text, pattern, N, M)
            if ( pos >= 0 ) {
                skdprint("Pattern occurs in text: " + text.mkString);
                do_match(text, pattern, table, N, M, pos)
	    } else {
                skdprint("Pattern does not occur in text: " + text.mkString);
                do_no_match(text, pattern, table, N, M)
	    }

            seq_next(text, N, 0);

            // mutate a position in text for the next variation
// 	    val j = i % N
// 	    if (text(j) == 'a') { 
//                 text(j) = 'b';
//             } else if (text(j) == 'b') {
//                 text(j) = 'a';
//             }
        }
    }

    def iterative_pos_find(text: Array[Char], pattern: Array[Char], N: Int, M: Int): Int = {
        var pos = 0
        var found = false
        var found_pos = -1;
        while (pos < N - M + 1 && !found) {
            if (match_at_tpos_ppos(text, pattern, pos, 0, M) == M) {
               found = true;
               found_pos = pos;
            }
            pos = pos + 1;
        }
        return found_pos;
    }

    def guess_pos_find(text: Array[Char], pattern: Array[Char], N: Int, M: Int): Boolean = {
        val pos = !!(N-M+1)
        val found = (match_at_tpos_ppos(text, pattern, pos, 0, M) == M)
        return found
    }

    def retracting_pos_find(text: Array[Char], pattern: Array[Char], table : Array[Int], N: Int, M: Int): Int = {
        var ppos = 0 // where to start matching in pattern
        var tpos = 0 // where to start matching in text
        var found = false
        var found_pos = -1;
        while (tpos < N - (M - ppos) + 1 && !found) {
            val pf = match_at_tpos_ppos(text, pattern, tpos, ppos, M) // "frontier" in pattern at first mismatch
            val m = pf - ppos // num chars matched, could be 0
            skdprint("pf = " + pf + ", matched = " + m)
            if (pf == M) {
                found = true;
                found_pos = tpos+m-M;
            } else {
               ppos = retract(text, pattern, table, ppos, tpos, m);
               tpos = tpos + m + 1;
               skdprint("ppos = " + ppos + ", tpos = " + tpos)
            }
        }
        return found_pos;
    }

    def retract(text : Array[Char], pattern : Array[Char], table : Array[Int], ppos : Int, tpos : Int, m : Int) : Int = {

        val pf = ppos + m // index of first mismatch in the pattern

        // pattern(ppos + m) mismatches text(tpos + m), but matches leftwards of it i.e. pattern(0 .. pos+m-1)
        // find x, an index in the pattern in the range 0 <= x <= pf such that 
        //     pattern (0 .. x-1) matches text ( .. tpos+m); when x is 0, no chars are matched
        // x is the index that will be positioned to match up with tpos+m+1 next
        // return largest x that works, or 0 if non such

        val x = refine_x(pattern, table, pf);

        // ensure the value meets the spec
        synthAssert ( x == iter_x(text, pattern, ppos, tpos, m, pf)) ; // tried guess_nomax_x too

        return x    
    }

    // deterministic but slow computation of suitable x, serves as spec
    def iter_x(text : Array[Char], pattern : Array[Char], ppos : Int, tpos : Int, m : Int, pf : Int) : Int = {
        var x = pf;
        while (x > 0) {
           if  (checkprefix(text, pattern, ppos, tpos, x, m))
                return x;
           x = x - 1;
        }
        return 0;
    }

    // attempt to write spec using non determinism, but not complete because of lack of max check
    def guess_nomax_x(text : Array[Char], pattern : Array[Char], ppos : Int, tpos : Int, m : Int, pf : Int) : Int = {
        val x = !!(pf + 1)
        synthAssert(x >= 0 && x <= pf)
        skdprint("x = " + x);
   
        // correctness of x: verify match of pat(0 .. x-1)
        synthAssert ( checkprefix(text, pattern, ppos, tpos, x, m) )

        // note: maximality of x not checked yet, which is a problem

        return x;
    }

    // refinement-based derivation of x. Should depend only on the contents of the pattern.
    def refine_x(pattern : Array[Char], table : Array[Int], pf: Int) : Int = {
        val x = !!(pf + 1);
        // val x = table(pf)
        skdprint("x = " + x);
        return x;
    }

    def checkprefix(text : Array[Char], pattern : Array[Char], ppos : Int, tpos : Int, x : Int, m : Int) : Boolean = {
        var i = x - 1;
        var j = tpos + m;
        while (i >= 0) {
            if (pattern(i) != text(j))
                return false;
            i = i - 1;
            j = j - 1;
        }
        return true;
    }

    // tpos: position in the text array starting which we start comparing
    // ppos: position in the pattern array starting which we look for a match. This generality is needed because
    //       sometimes we may wish to only match a suffix of the pattern, assuming that the characters to the left 
    //         of ppos already match the text to the left of pos.
    // return: We may not be able to match until the end of the pattern array. Therefore the return value is the
    //            position in the pattern array where the first mismatch was found (or M, if success).

    def match_at_tpos_ppos(text: Array[Char], pattern: Array[Char], tpos : Int, ppos : Int, M: Int): Int = {
        var green = true;
        var next = ppos; // position in pattern where to match next
        var tcursor = tpos;
        while (next < M && green) { 
            if (text(tcursor) != pattern(next)) {
                green = false;
            } else {
                tcursor = tcursor+1;
                next = next+1;
            }
        }
        return next;
    }
}

object SearchCliOptions extends cli.CliOptionGroup {
    var result : cli.CliOptionResult = null
}

object Search {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        SearchCliOptions.result = SearchCliOptions.parse(cmdopts)
        skalch.AngelicSketchSynthesize(() => new SearchSketch())
    }
}
