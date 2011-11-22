
package edu.berkeley.cs.binary

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class Binary1Sketch extends AngelicSketch {
  val tests = Array(())

  def main() {
      class BinaryTreeNode(val value : String) {
          var leftChild : BinaryTreeNode = null;
          var rightChild : BinaryTreeNode = null;
          
          override def toString() : String = {
              return toString(0);
          }
          
          def toString(depth : Int) : String = {
              if (depth > 10) {
                  return "";
              }
              
              var ret = "(";
              if (leftChild != null) {
                  ret += leftChild.toString(depth + 1);
              }
              ret += value;
              if (rightChild != null) {
                  ret += rightChild.toString(depth + 1);
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
                  lastNode.leftChild = new BinaryTreeNode(value);
              } else if (value > lastNode.value) {
                  lastNode.rightChild = new BinaryTreeNode(value);
              }
          }
          
          def remove(value : String) {
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
                  
                  var replacement : BinaryTreeNode = null;
                  var other : BinaryTreeNode = null;
                  
                  replacement = !!(leftChild, rightChild, curNode);
                  other = !!(leftChild, rightChild, curNode);
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
                  
                  if (!!(true, false)) {
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
                      
                      i = 0;
                      while (!!(true, false)) {
                          synthAssert(i < 10);
                          i = i + 1;
                  
                          if (!!(true, false)) {
                              nodeToFix = nodeToFix.leftChild;
                          } else {
                              nodeToFix = nodeToFix.rightChild;
                          }
                      }
                      
                      skdprint("Other:" + other)
                      /*
                      nodeToFix.rightChild = other;
                      */
                      if (!!(true, false)) {
                          nodeToFix.leftChild = other;
                      } else {
                          nodeToFix.rightChild = other;
                      }
                      skdprint("Fixed:" + nodeToFix)
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
              return root.toString();
          }
      }
      
      val tree = new BinaryTree();
      tree.add("down")
      tree.add("amigo");
      tree.add("cat");
      tree.add("bye");
      tree.add("elope");
      tree.add("grass");
      tree.add("frugal");
      tree.add("tube");
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(tree.isPresent("cat"));
      synthAssert(tree.isPresent("down"));
      synthAssert(tree.isPresent("elope"));
      synthAssert(tree.isPresent("grass"));
      synthAssert(tree.isPresent("frugal"));
      synthAssert(tree.isPresent("tube"));
      
      tree.remove("elope");
      tree.remove("grass");
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(tree.isPresent("cat"));
      synthAssert(tree.isPresent("down"));
      synthAssert(!tree.isPresent("elope"));
      synthAssert(!tree.isPresent("grass"));
      synthAssert(tree.isPresent("frugal"));
      synthAssert(tree.isPresent("tube"));
  }
}

object Binary1 {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new Binary1Sketch())
  }
}
