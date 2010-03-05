package edu.berkeley.cs.listreversejoel

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class Node(v : Int) extends AngelicSketch{
  var value = v
  var next : Node = null;
  
  def add(v : Int) {
    if(next != null) {
      next.add(v)
    } else {
      next = new Node(v)
    }
  }
  
  def getList() : List[Node] = {
    if (next != null) {
      return this :: next.getList
    } else {
      return List(this)
    }
  }
  
  def toList() : List[Int] = {
    if (next != null) {
      return value :: next.toList
    } else {
      return List(value)
    }
  }

  override def toString() : String = {
    return "" + value;
  }
}

class LinkedList extends AngelicSketch {
    var head : Node = null
    var size : Int = 0
    
    def add(v : Int) {
      size = size + 1
      if (head == null) {
        head = new Node(v)
      } else {
        head.add(v)
      }
    }
    
    def check() {
      var node : Node = head
      skdprint("" + size)
      for (i <- 0 to size - 1) {
        skdprint("" + node.value)
        synthAssert(node != null)
        node = node.next
      }
      synthAssert(node == null)
    }
    
     def getList() : List[Node] = {
       this.check()
       return head.getList()
      }
      
      def toList() : List[Int] = {
       this.check()
       return head.toList()
      }
      
      def length() : Int = {
        this.check()
        return size
     }
      
     override def toString() : String = {
       return getList.toString
     }
}
