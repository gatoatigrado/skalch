package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/* 
 * Constraints:
 *   Must only traverse the original list once.
 *   Control flow of program. Must access head, then recurse, then append.
 *   Must only pass the tail of the list to the next call.
 * Angelic choices:
 *   Base case.
 *   Whether to append the element to the list.
 *   Can append to beginning of list or end of list.
 *   
 * Lessens learned:
 *   Works but shows that the only way this method can work is if the
 *   element is added to the end of the returned list. This method
 *   will take O(n^2) time and is unacceptable. Need more freedom with 
 *   structure of program.
 */
class LinkedListReversalSketch3 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T], i : Int) : List[T] = {
			var revList : List[T] = Nil
			var value : T = list.head
			
			if (!!() && list.tail != null && i < 5) {
				revList = reverse((list.tail), i+1)
				skdprint(revList.toString())
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
		val rev_l = reverse(l, 0)
		synthAssert(rev_l == rev)
	}
}

object ListReverseMain3 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.addOpts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch3())
		}
	}