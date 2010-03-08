package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/* Scalability problems due to no changes in state
 * Constraints:
 *   Must only traverse the original list once.
 *   Control flow of program. Must access head, then recurse, then append.
 * Angelic choices:
 *   Base case.
 *   What should be passed to the recursive call.
 *   Whether to append the element to the list.
 *   Can append to beginning of list or end of list.
 *   
 * Lessens learned:
 *   Theoretically can work but their are scalibility problems due to no state change
 *   in recursive call. This overwhelms the backtracking algorithm
 *   and cannot find the solution.
 */

class LinkedListReversalSketch2 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			var value : T = list.head	
			
			if (!!()) {
				revList = reverse(!!(list, list.tail))
			}
			
			if (!!()) {
				revList = value :: revList
			}
			if (!!()) {
				revList = revList ::: List(value)
			}
			
			return revList;
		}
		
		val l : List[String] = List("1", "2", "3", "4")
		val rev : List[String] = List("4", "3", "2", "1")
		val rev_l = reverse(l)
		synthAssert(rev_l == rev)
	}
}

object ListReverseMain2 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.addOpts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch2())
		}
	}