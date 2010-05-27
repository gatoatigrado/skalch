repositories.remote << 'http://www.ibiblio.org/maven2'
repositories.remote << 'http://cobol.cs.berkeley.edu/mvn'
repositories.remote << 'http://scala-tools.org/repo-snapshots'

require 'buildr/scala'
Buildr::Scala::DEFAULT_VERSION = '2.8.0'

SCALA_LIB = 'org.scala-lang:scala-library:2.8.0-SNAPSHOT'
SCALA_COMPILER = 'org.scala-lang:scala-compiler:2.8.0-SNAPSHOT'
SKETCH = 'edu.berkeley.cs.sketch:SKETCH:jar:0.01-SNAPSHOT'
SKETCH_UTIL = 'edu.berkeley.cs.sketch:sketch-util:jar:0.01'

define 'skalch' do
    project.group = 'edu.berkeley.cs.sketch'
    project.version = '0.03-SNAPSHOT'
    compile.options.target = '1.5'
    compile.options.lint = 'all'

    define 'skalch-plugin' do
        compile.with(transitive(SKETCH))
        package :jar
    end

    define 'skalch-base' do
        package :jar
        resources.from("src/main")
    end
end
