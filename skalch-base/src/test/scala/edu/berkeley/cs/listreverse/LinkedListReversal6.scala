package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

// fail, only works at appending to end, need external list
class LinkedListReversalSketch6 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T]) : List[T] = {
			var revList : List[T] = Nil
			def reverseRec() : Unit = {
				if (!!())
					return;
				
				var value : T = !!(list)	
				revList = value :: revList
				
				reverseRec()
			}
			reverseRec()
			return revList;
		}
		
		val l : List[String] = List("1", "2", "3", "4")
		val rev : List[String] = List("4", "3", "2", "1")
		val rev_l = reverse(l)
		synthAssertTerminal(rev_l == rev)
	}
}

object ListReverseMain6 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch6())
	}
}