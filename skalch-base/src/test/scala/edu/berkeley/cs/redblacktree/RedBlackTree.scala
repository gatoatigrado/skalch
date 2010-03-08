package edu.berkeley.cs.redblacktree;

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class RedBlackTreeSketch extends AngelicSketch {
  val tests = Array( () )
  
  def main() = {
	class RedBlackTree { 
      class RedBlackTreeNode(v : Int, l : RedBlackTreeNode,
  		  r : RedBlackTreeNode, c : Boolean) {
   	    var value : Int = v
  	    var left : RedBlackTreeNode = r
  	    var right : RedBlackTreeNode = l
	    var isBlack : Boolean = c
	    
	    override def toString() : String = {
	      var returnString : String = value.toString()
	      if (isBlack) {
	        returnString += "[black]"
	      } else {
	    	returnString += "[red]"
	      }
	      return returnString
	    }
      }
  
      var root : RedBlackTreeNode = null
     
      def insert(v : Int) = {
	    skput("insert " + v)
	    if (root == null) {
          root = new RedBlackTreeNode(v, null, null, true)
        } else {
        	
          def insert(v : Int, curNode : RedBlackTreeNode,
        		  parent : RedBlackTreeNode) : Unit= {
        	val putOnLeft : Boolean = v < curNode.value
        	if (putOnLeft) {
              if (curNode.left == null) {
                val isBlack : Boolean = !!(true, false)
                curNode.left = new RedBlackTreeNode(v, null, null, isBlack)
              } else {
                insert(v, curNode.left, curNode)
              }
            } else {	  
              if (curNode.right == null) {
                val isBlack : Boolean = !!(true, false)
                curNode.right = new RedBlackTreeNode(v, null, null, isBlack)
              } else {
            	insert(v, curNode.right, curNode)
              }
            }

  	        if (!checkRedBlack()) {
    	      if (!!()) {
    	    	if (!!()) {
    	    	  rotateClockwise(curNode, parent, putOnLeft)
    	    	  skput("clock")
    	    	} else { 
    	          rotateCounterClockwise(curNode, parent, putOnLeft)
    	          skput("cclock")
    	    	}
    	      } else {
    	    	  skput("none")
    	      }
    	    }  
          }
	      
          insert(v, root, null)
          
        }
	    synthAssert(checkRedBlack())
	    synthAssert(checkOrder())
	    skdprint(toString())
      }
	  
	  def rotateClockwise(node : RedBlackTreeNode, parent : RedBlackTreeNode, 
			  isLeftChild : Boolean) = {
	    if (node.left != null) {
	      if (!!()) {
	    	node.isBlack = !node.isBlack
	      }
	      if (node.left != null && !!()) {
	    	node.left.isBlack = !node.left.isBlack
	      }
	      if (node.left.right != null && !!()) {
	    	node.left.right.isBlack = !node.left.right.isBlack
	      }
	      
	      // set pointer of parent to correct node
	      if (parent != null) {
	    	if (isLeftChild) {
	    	  parent.left = node.left
	        } else {
	    	  parent.right = node.left
	        }
	      } else {
	    	// if parent is null, then the node is the root
	    	root = node.left
	      }
	      
	      var leftRightChild : RedBlackTreeNode = node.left.right
	      node.left.right = node
	      node.left = leftRightChild
	    }
	  }
	    
	  def rotateCounterClockwise(node : RedBlackTreeNode, parent : RedBlackTreeNode, 
				  isLeftChild : Boolean) = {
	    if (node.right != null) {
	      if (!!()) {
	    	node.isBlack = !node.isBlack
	      }
		  if (node.right != null && !!()) {
			node.right.isBlack = !node.right.isBlack
		  }
		  if (node.right.left != null && !!()) {
			node.right.left.isBlack = !node.right.left.isBlack    
		  }
	    	
	      // set pointer of parent to correct node
          if (parent != null) {
            if (isLeftChild) {
              parent.left = node.right
            } else {
              parent.right = node.right
            }
          } else {
            // if parent is null, then the node is the root
            root = node.right
          }
		      
          var rightLeftChild : RedBlackTreeNode = node.right.left
          node.right.left = node
          node.right = rightLeftChild
	    }
      }
   
      override def toString() : String = {
        if (root == null) {
          return ""
        }
        def toString(node : RedBlackTreeNode, depth : Int,
        		isLeft : Boolean) : String = {
          var returnString : String = ""

          for (i <- 0 until depth) {
            returnString += "  "
          }
          
          if (node != root) {
            if (isLeft) {
        	  returnString += "[L]"
            } else {
        	  returnString += "[R]"
            }
          }
        
		  returnString += node.toString()
		  
		  if (node.left != null) {
		    returnString += "\n" + toString(node.left, depth+1, true)
		  }
		  
		  if (node.right != null) {
		    returnString += "\n" + toString(node.right, depth+1, false)
		  }
		  
		  return returnString
	    }
	    return toString(root, 0, true)
      }
 
      def checkRedBlack() : Boolean = {
	    if (root == null) {
	      return true;
	    }
	  
	    // root must be black
	    if (root.isBlack != true ||
			  !checkBothChildrenOfRedAreBlack(root) ||
			  !checkBlackNodeCount(root)) {
          return false
	    }	  
	    return true
      }
  
      def checkOrder() : Boolean = {
	    if (root == null) {
		  return true;
        }
		  
        def checkOrder(node : RedBlackTreeNode) : Boolean = {
	      if (node.left != null) {
		    if (node.left.value > node.value ||
		         !checkOrder(node.left)) {
              return false
            }
          }
		  if (node.right != null) {
		    if (node.right.value < node.value ||
		  	    !checkOrder(node.right)) {
		      return false
		    }
		  }
		  return true
        }
	    checkOrder(root)
      }
  
      private def checkBothChildrenOfRedAreBlack(node : RedBlackTreeNode) : Boolean = {
	    if (node == null) {
	      return true;
	    }
	  
	    if (!node.isBlack) {
	      if ((node.left != null && !node.left.isBlack) ||
			    (node.right != null && !node.right.isBlack)) {
		    return false
		  }
	    }
	  
	    return checkBothChildrenOfRedAreBlack(node.left) &&
	        checkBothChildrenOfRedAreBlack(node.right)
      }
  
      private def checkBlackNodeCount(node : RedBlackTreeNode) : Boolean = {
	    def blackNodeCount(node : RedBlackTreeNode) : Int = {
		  if (node == null) {
		    return 0
		  } 
			  
		  var leftCount :Int = blackNodeCount(node.left)
		  var rightCount : Int = blackNodeCount(node.right)
		      
		  if (leftCount == -1 || leftCount != rightCount) {
		    return -1
		  }
		  
		  if (node.isBlack) {
		    leftCount += 1
		  }
		  return leftCount
	    }
	  
	    return blackNodeCount(node) != -1
      } 
	}
	
	val tree : RedBlackTree = new RedBlackTree()
 /*   if (!!()) { tree.insert(1)
    if (!!()) {	tree.insert(2)
    if (!!()) { tree.insert(3)
    if (!!()) { tree.insert(4)
    }}}}
*/  tree.insert(1)
	tree.insert(2)
	tree.insert(3)
	tree.insert(4)
	
    synthAssert(tree.checkRedBlack())
	synthAssert(tree.checkOrder())
	skdprint(tree.toString())
  }
}  


	/*
	private def test_() : RedBlackTree = {
			return new RedBlackTree()
	}
	
	private def test1() : RedBlackTree = {
			var tree : RedBlackTree = new RedBlackTree()
			tree.insert(1)
			return tree
	}
	
	private def test1_10() : RedBlackTree = {
			var tree : RedBlackTree = new RedBlackTree()
			tree.insert(1)
			tree.insert(2)
			tree.insert(3)
			tree.insert(4)
			tree.insert(5)
			tree.insert(6)
			tree.insert(7)
			tree.insert(8)
			tree.insert(9)
			tree.insert(10)
			
			return tree
	}
	
}
	*/

object RedBlackTreeMain {
  def main(args: Array[String]) = {
	  for (arg <- args)
		  Console.println(arg)
      val cmdopts = new cli.CliParser(args)
      BackendOptions.addOpts(cmdopts)
	  skalch.AngelicSketchSynthesize(() => new RedBlackTreeSketch())
  }
}

