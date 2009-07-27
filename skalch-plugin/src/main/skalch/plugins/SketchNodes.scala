package skalch.plugins

import streamit.frontend.scala.ScReflectionUtils

object SketchNodes {
    val all_classes = ScReflectionUtils.singleton.get_node_classes()

    def get_sketch_class[T](clazz : Class[T]) = {
        ScReflectionUtils.singleton.get_sketch_class(clazz)
    }
}
