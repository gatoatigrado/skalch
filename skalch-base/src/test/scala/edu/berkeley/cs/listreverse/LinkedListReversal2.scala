package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

// scalability problems due to no change in state
class LinkedListReversalSketch2 extends AngelicSketch {
	val tests = Array( () )
	
	def main() {
		def reverse[T](list : List[T], i : Int) : List[T] = {
			var revList : List[T] = Nil
			var value : T = list.head
			
			
			
			if (!!() && i < 5) {
				revList = reverse(!!(list, list.tail), i+1)
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
		synthAssertTerminal(rev_l == rev)
	}
}

object ListReverseMain2 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch2())
		}
	}