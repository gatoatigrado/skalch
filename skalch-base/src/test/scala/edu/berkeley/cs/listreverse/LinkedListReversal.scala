package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/*
 * I wanted to test if list reversal could be done iteratively. This
 * example is quite staightforward. Additionally, I wanted to test using the queues
 * and see if they could help the refinement process.
 * 
 * Constraints:
 *   Can only iterate through loop n times
 *   Must append to beginning of loop
 * Angelic choices:
 *   Can access any element in the loop to append to beginning
 *   
 * Lessens learned:
 *   It might be possible to do the function iteratively in n steps.
 */

class LinkedListReversalSketch extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			var i : Int = list.length
			
			while(!!() && i > 0) {
				val v : T = !!(list)
				skput_check(v.toString())
				skdprint(v.toString())
				revList = v :: revList
				i = i - 1
			}
			return revList;
		}
		
		val l : List[String] = List("1", "2", "3", "4")
		val rev : List[String] = List("4", "3", "2", "1")
		val rev_l = reverse(l)
		synthAssert(rev_l == rev)
	}
}

object ListReverseMain {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch())
		}
	}

