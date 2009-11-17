package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/* 
 * Constraints:
 *   Must only traverse the original list once.
 *   Must only pass the tail of the list to the next call.
 *   Can only append to the beginning of the list.
 *   Have a global list which is returned. Global list is initially nil.
 *   Base case.
 *   Tail recursion.
 * Angelic choices:
 *   
 * Lessens learned:
 *   Deterministic tail recursion. Now need to get rid of global variable.
 */
class LinkedListReversalSketch5 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			
			def reverseRec(list : List[T]) : Unit = {
				var value : T = list.head
				
				revList = value :: revList
				
				if (!list.tail.isEmpty) {
					reverseRec(list.tail)
				}		
			}
			reverseRec(list)
			return revList;
		}
		
		val l : List[String] = List("1", "2", "3", "4")
		val rev : List[String] = List("4", "3", "2", "1")
		val rev_l = reverse(l)
		synthAssert(rev_l == rev)
	}
}

object ListReverseMain5 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch5())
	}
}