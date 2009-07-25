package skalch.plugins

import streamit.frontend.scala.ScReflectionUtils

object SketchNodeInfo {
    val all_classes = ScReflectionUtils.singleton.get_node_classes()
}

class SketchNode {
}
