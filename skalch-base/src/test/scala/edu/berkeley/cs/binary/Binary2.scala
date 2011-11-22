
package edu.berkeley.cs.binary

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class Binary2Sketch extends AngelicSketch {
  val tests = Array(())

  def main() {
//      var bitString = List[List[Int]](List(0), List(1));
//      for (i <- 0 to 6) {
//          bitString = bitString ::: bitString.map((oldVal => 1 :: oldVal)) ::: bitString.map((oldVal => 0 :: oldVal)) 
//      }
      
      class BinaryTreeNode(val value : String) {
          var leftChild : BinaryTreeNode = null;
          var rightChild : BinaryTreeNode = null;
          
          override def toString() : String = {
              var str = "[" + value;
              str += ":";
              if (leftChild != null) {
                  str += leftChild.value; 
              }
              str += ",";
              if (rightChild != null) {
                  str += rightChild.value;
              }
              str += "]";
              return str;
          }
      }
      
      class BinaryTree {
          var root : BinaryTreeNode = new BinaryTreeNode("_");
          var nodes : List[BinaryTreeNode] = List(root);
          
          def add(value : String) {
              val newNode = new BinaryTreeNode(value)
              nodes = newNode :: nodes;
              
              var curNode = root;
              var lastNode : BinaryTreeNode = null;
              
              var i = 0;
              while (curNode != null) {
                  synthAssert(i < 10);
                  i = i + 1;
                  
                  lastNode = curNode;
                  if (value < curNode.value) {
                      curNode = curNode.leftChild;
                  } else if (value > curNode.value) {
                      curNode = curNode.rightChild;
                  } else {
                      return;
                  }
              }
              
              if (value < lastNode.value) {
                  lastNode.leftChild = newNode;
              } else if (value > lastNode.value) {
                  lastNode.rightChild = newNode;
              }
          }
          
          def remove(value : String) {
              nodes = nodes.remove(node => node.value == value)
              
              skdprint("Removing:" + value)
              var curNode = root;
              var lastNode : BinaryTreeNode = null;
              var i = 0;
              while (curNode != null && curNode.value != value) {
                  synthAssert(i < 10);
                  i = i + 1;
                  
                  lastNode = curNode;
                  skdprint("Current Node:" + curNode);
              
                  if (value < curNode.value) {
                      curNode = curNode.leftChild;
                  } else if (value > curNode.value) {
                      curNode = curNode.rightChild;
                  }
              }
              
              skdprint("Node:" + curNode);
              if (curNode != null) {
                  val leftChild = curNode.leftChild
                  val rightChild = curNode.rightChild
                  val nullNode : BinaryTreeNode = null;
                  
                  var replacement : BinaryTreeNode = null;
                  var other : BinaryTreeNode = null;
                  
                  val nodeList = List(leftChild, rightChild, curNode, nullNode) 
                  val replacementOracle : Int = !!(nodeList.length);
                  sktrace(replacementOracle);
                  replacement = nodeList(replacementOracle);
                  
                  val otherOracle : Int = !!(nodeList.length);
                  sktrace(otherOracle);
                  other = nodeList(otherOracle);
                  /*
                  if (leftChild == null && rightChild != null) {
                      replacement = rightChild;
                  } else if (leftChild != null && rightChild == null) {
                      replacement = leftChild
                  } else {
                      replacement = leftChild;
                      other = rightChild;
                  }
                  */
                  
                  skdprint("Replacement:" + replacement)
                  val booleanList = List(true, false)
                  var whichChildOracle : Int = !!(booleanList.length);
                  //sktrace(whichChildOracle)
                  
                  if (curNode == lastNode.leftChild) {
                      lastNode.leftChild = replacement;
                  } else {
                      lastNode.rightChild = replacement;
                  }
                  /*
                  if(lastNode.leftChild == curNode)
                      lastNode.leftChild = replacement;
                  else
                      lastNode.rightChild = replacement;
                  */
                  
                  if (other != null) {
                      skdprint("New config:" + lastNode)
                      
                      var nodeToFix = replacement;
                      
                      /*
                      while(nodeToFix.rightChild != null) {
                          skdprint("Fixing:" + nodeToFix)
                          nodeToFix = nodeToFix.rightChild;
                      }
                      */
                      
                      var traceVal : Int = 1;
                      
                      i = 0;
                      while (!!(true, false)) {
                          synthAssert(i < 10);
                          i = i + 1;
                  
                          traceVal = traceVal << 1;
                          
                          if (!!(true, false)) {
                              nodeToFix = nodeToFix.leftChild;
                              traceVal = traceVal + 1;
                          } else {
                              nodeToFix = nodeToFix.rightChild;
                          }
                      }
                      sktrace(traceVal);
                      
                      skdprint("Other:" + other)
                      /*
                      nodeToFix.rightChild = other;
                      */
                      
                      whichChildOracle = !!(booleanList.length);
                      sktrace(whichChildOracle)
                  
                      if (booleanList(whichChildOracle)) {
                          nodeToFix.leftChild = other;
                      } else {
                          nodeToFix.rightChild = other;
                      }
                      skdprint("Fixed:" + nodeToFix)
                  } else {
                      sktrace(0);
                      sktrace(0);
                  }
              }
          }
          
          def isPresent(value : String) : Boolean = {
              var curNode = root;
              var i = 0;
              while (curNode != null) {
                  synthAssert(i < 10);
                  i = i + 1;
                  
                  if (value == curNode.value) {
                      return true;
                  } else if (value < curNode.value) {
                      curNode = curNode.leftChild;
                  } else if (value > curNode.value) {
                      curNode = curNode.rightChild;
                  }
              }
              return false;
          }
          
          override def toString() : String = {
              return nodes.foldLeft("")((str, node) => str + node.toString());
          }
      }
      
      val tree = new BinaryTree();
//      Example 1
//      tree.add("cat");
//      tree.add("bye");
//      tree.add("amigo");
//      tree.add("down")
//      skdprint("Tree: " + tree.toString());
//      
//      synthAssert(tree.isPresent("amigo"));
//      synthAssert(tree.isPresent("bye"));
//      synthAssert(tree.isPresent("cat"));
//      synthAssert(tree.isPresent("down"));
//      
//      tree.remove("cat");
//      skdprint("Tree: " + tree.toString());
//      
//      tree.remove("bye");
//      skdprint("Tree: " + tree.toString());
//      
//      synthAssert(tree.isPresent("amigo"));
//      synthAssert(!tree.isPresent("bye"));
//      synthAssert(!tree.isPresent("cat"));
//      synthAssert(tree.isPresent("down"));    
      
//      Example 2
      tree.add("down")
      tree.add("bye");
      tree.add("cat");
      tree.add("amigo");
      tree.add("fat");
//      tree.add("egg");
//      tree.add("gut");
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(tree.isPresent("cat"));
      synthAssert(tree.isPresent("down"));
      synthAssert(tree.isPresent("fat"));
//      synthAssert(tree.isPresent("egg"));
//      synthAssert(tree.isPresent("gut"));
      
      tree.remove("down");
      skdprint("Tree: " + tree.toString());
      
      tree.remove("bye");
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(!tree.isPresent("bye"));
      synthAssert(tree.isPresent("cat"));
      synthAssert(!tree.isPresent("down"));
      synthAssert(tree.isPresent("fat"));
//      synthAssert(tree.isPresent("egg"));
//      synthAssert(tree.isPresent("gut"));
      
  }
}

object Binary2 {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new Binary2Sketch())
  }
}
