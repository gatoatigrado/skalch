package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/* Failure. This is my first iteration writing the program recursively.
 * Constraints:
 *   Must only traverse the original list once.
 *   Control flow of program. Must access head, then recurse, then append.
 *   Must append to beginning of list.
 * Angelic choices:
 *   Base case.
 *   What should be passed to the recursive call.
 *   Whether to append the element to the list.
 *   
 * Lessens learned:
 *   Too many constraints and not enough freedom for angels. This method
 *   clearly will not work.
 */

class LinkedListReversalSketch1 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			var value : T = list.head
			
			if (!!())
				return revList;
		
			if (!!()) {
				revList = reverse(!!(list, list.tail))
			}
			
			if(!!()) {
				revList = value :: revList
			}
			
			return revList;
		}
		
		val l : List[String] = List("1", "2", "3", "4")
		val rev : List[String] = List("4", "3", "2", "1")
		val rev_l = reverse(l)
		synthAssert(rev_l == rev)
	}
}

object ListReverseMain1 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch1())
		}
	}