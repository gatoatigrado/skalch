
package edu.berkeley.cs.binary

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class BinarySketch extends AngelicSketch {
  val tests = Array(())

  def main() {
      class BinaryTreeNode(val value : String) {
          var leftChild : BinaryTreeNode = null;
          var rightChild : BinaryTreeNode = null;
          
          override def toString() : String = {
              var ret = "(";
              if (leftChild != null) {
                  ret += leftChild.toString();
              }
              ret += value;
              if (rightChild != null) {
                  ret += rightChild.toString();
              }
              ret += ")";
              return ret;
          }
      }
      
      class BinaryTree {
          var root : BinaryTreeNode = new BinaryTreeNode("_");
          
          def add(value : String) {
              var curNode = root;
              var lastNode : BinaryTreeNode = null;
              
              while (curNode != null) {
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
                  lastNode.leftChild = new BinaryTreeNode(value);
              } else if (value > lastNode.value) {
                  lastNode.rightChild = new BinaryTreeNode(value);
              }
          }
          
          def remove(value : String) {
              skdprint("Removing:" + value)
              var curNode = root;
              var lastNode : BinaryTreeNode = null; 
              while (curNode != null && curNode.value != value) {
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
                  
                  var replacement : BinaryTreeNode = null;
                  var other : BinaryTreeNode = null;
                  if (leftChild == null && rightChild != null) {
                      replacement = rightChild;
                  } else if (leftChild != null && rightChild == null) {
                      replacement = leftChild
                  } else {
                      replacement = leftChild;
                      other = rightChild;
                  }
                  
                  skdprint("Replacement:" + replacement)
                  
                  if(lastNode.leftChild == curNode)
                      lastNode.leftChild = replacement;
                  else
                      lastNode.rightChild = replacement;
                  
                  if (other != null) {
                      skdprint("New config:" + lastNode)
                      
                      var nodeToFix = replacement;
                      
                      while(nodeToFix.rightChild != null) {
                          skdprint("Fixing:" + nodeToFix)
                          nodeToFix = nodeToFix.rightChild;
                      }
                      
                      replacement = other;
                      skdprint("Replacement:" + replacement)
                      nodeToFix.rightChild = replacement;
                      skdprint("Fixed:" + nodeToFix)
                  }
              }
          }
          
          def isPresent(value : String) : Boolean = {
              var curNode = root;
              while (curNode != null) {
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
              return root.toString();
          }
      }
      
      val tree = new BinaryTree();
      tree.add("amigo");
      tree.add("cat");
      tree.add("bye");
      tree.add("downtube")
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(!tree.isPresent("down"));
      synthAssert(!tree.isPresent("tube"));
      
      tree.remove("amigo");
      
      synthAssert(!tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(!tree.isPresent("down"));
      synthAssert(!tree.isPresent("tube"));
   }
}

object Binary {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new BinarySketch())
  }
}
