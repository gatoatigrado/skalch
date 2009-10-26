package edu.berkeley.cs.listreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

// fail, only works at appending to end, need external list
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
		synthAssertTerminal(rev_l == rev)
	}
}

object ListReverseMain3 {
	def main(args: Array[String]) = {
		for (arg <- args)
			Console.println(arg)
	    val cmdopts = new cli.CliParser(args)
	    BackendOptions.add_opts(cmdopts)
		skalch.AngelicSketchSynthesize(() => 
			new LinkedListReversalSketch3())
		}
	}