package edu.berkeley.cs;

import skalch.AngelicSketch;

class ListZipReverseSketch() extends AngelicSketch {

    val tests = Array( () );
    
    val bool = List(false,true);

    def main() = {  // 2 solutions: both up and down are legal
        val x = List("a", "b", "c", "d");
        val y = List("4", "3", "2", "1");

        // trace of angels from previous iteration
        var trace = List("d","4","c","3","b","2","a","1");
        // TODO: record the trace programmatically in the previous version

        // checking angels against a previous trace sped it up from 25s (for recur depth limited to 5) to 0s (for no limit to recursion depth)
        def ct(l:List[String]) : String = {
        	// trace better has one more entry
            synthAssertTerminal(trace != Nil);
            val v = !!(l);
            skdprint(v);
            synthAssertTerminal(v == trace.head);
            trace = trace.tail;
            v;
        }

        var r:List[String] = Nil;
        
        // do we need to do "work" (ie consing) on the way up from the recursion or down?
        val up = !!();
        
        def descent() : Unit = {
            if (!!()) {
            	// angels that appear in the previous version (and have not been refined)
            	// are now replaced with ct(), which checks that the value generated is the
            	// same as in the safe trace from previous version
            	if (!up) r = ct(x) + ct(y) :: r;
                descent();
                if (up)  r = ct(x) + ct(y) :: r;
            }
        }
        descent();
        synthAssertTerminal(r == List("a1","b2","c3","d4"));
    }
}

object ListZipReverseMain2 {
    def main(args: Array[String]) =
    	skalch.AngelicSketchSynthesize(() => new ListZipReverseSketch());
}
