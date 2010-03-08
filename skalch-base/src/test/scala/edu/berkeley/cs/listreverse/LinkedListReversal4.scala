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
 * Angelic choices:
 *   Base case.
 *   Whether and when to append the element to the return list.
 *   Whether to do work when going up or down.
 *   
 * Lessens learned:
 *   Works and shows that you can do tail recursion.
 */
class LinkedListReversalSketch4 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			val up : Boolean = !!()
			
			def reverseRec(list : List[T]) : Unit = {
				var value : T = list.head
			    skput_check(value)
                
				if (up && !!())
					revList = value :: revList
				
				if (!!() && !list.tail.isEmpty) {
					reverseRec(list.tail)
					skdprint(revList.toString())
				}
				
				if (!up && !!())
					revList = value :: revList
					
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

object ListReverseMain4 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.addOpts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch4())
	}
}