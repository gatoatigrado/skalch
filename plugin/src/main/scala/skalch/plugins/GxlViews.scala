package skalch.plugins

import scala.collection.mutable.{ListBuffer, HashMap, HashSet}

import ScalaDebugOut._
import scala.tools.nsc
import nsc._

import net.sourceforge.gxl._

class GxlArrWrapper[T <: GXLElement](arr : Array[T]) {
    def get_by_attr(key : Tuple2[String, String]) : GXLElement = {
        arr.filter(_.getAttribute(key._1) == key._2)(0)
    }
}

class XmlWrapper[T <: GXLElement : ClassManifest](elt : T) {
    def subelts() = (for (i <- 0 until elt.getChildCount()) yield elt.getChildAt(i)).toArray
    def filter(f : (GXLElement => Boolean)) = subelts.filter(f)
    def getnodes() = {
        val result = ListBuffer[GXLNode]()
        for (subelt <- subelts) subelt match {
            case node : GXLNode => result.append(node)
            case _ => ()
        }
        result.toArray
    }
    def getedges() = {
        val result = ListBuffer[GXLEdge]()
        for (subelt <- subelts) subelt match {
            case node : GXLEdge => result.append(node)
            case _ => ()
        }
        result.toArray
    }
}

class GxlNodeWrapper[T <: GXLNode](arr : Array[T]) {
    def filtertype(typ : String) = arr.filter(_.getType().getURI().toString() == typ)
}

object GxlViews {
    implicit def xmlwrapper[T <: GXLElement : ClassManifest](elt : T) = new XmlWrapper(elt)
    implicit def arrwrapper[T <: GXLElement](arr : Array[T]) = new GxlArrWrapper(arr)
    implicit def nodewrapper[T <: GXLNode](arr : Array[T]) = new GxlNodeWrapper(arr)
}
