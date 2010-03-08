package edu.berkeley.cs.incremental

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class ShortestPathSketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
      
      import scala.collection.mutable.HashMap
      import scala.collection.mutable.HashSet
        
      class Vertex(name : String) {
      }
      
      class Graph {
        
        val edges = new HashMap[Tuple2[Vertex,Vertex], Int]()
        val vertices = new HashMap[String, Vertex]()
        
        def addEdge(start:String, end:String, distance:Int) {
          assert(distance > 0)
          val s : Vertex = vertices(start)
          val  e : Vertex = vertices(end)
          assert(s != null && e != null)
          edges((s,e)) = distance
        }
        
        def addVertex(name:String) {
          val v = new Vertex(name)
          vertices(name) = v
        }
        
        def shortestPath(vertex:String) : HashMap[String,List[String]] = {
          val v = vertices(vertex)
          val curPaths = HashMap[Vertex, List[Vertex]]()
          val curDistance = HashMap[Vertex, Int]()
        //  val verticesLeft = vertices.clone.keySet() - v
          return null
        }
        
        def getEdges(start:Vertex) : HashMap[Vertex, Int] = {
          val edgesFromStart = new HashMap[Vertex, Int];
          edges.foreach { tuple:((Vertex,Vertex),Int) =>
            val ((s,e),d) = tuple
            if (start == s) {
              edgesFromStart(e) = d
            }
          }
          return edgesFromStart
        }
      }
    }
}

object ShortestPath {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ShortestPathSketch())
        }
    }

