
package edu.berkeley.cs.binary

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class Binary3Sketch extends AngelicSketch {
  val tests = Array(())

  def main() {
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
          var nodes : List[BinaryTreeNode] = List();
          
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
      tree.add("cat");
      tree.add("bye");
      tree.add("amigo");
      tree.add("down")
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(tree.isPresent("bye"));
      synthAssert(tree.isPresent("cat"));
      synthAssert(tree.isPresent("down"));
      
      tree.remove("cat");
      skdprint("Tree: " + tree.toString());
      
      tree.remove("bye");
      skdprint("Tree: " + tree.toString());
      
      synthAssert(tree.isPresent("amigo"));
      synthAssert(!tree.isPresent("bye"));
      synthAssert(!tree.isPresent("cat"));
      synthAssert(tree.isPresent("down"));
  }
}

object Binary3 {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new Binary3Sketch())
  }
}
