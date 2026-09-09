import java.io.File

import sbt._

/**
 * Source selection used when an agent-supplied CFU design overlays the tracked
 * AgentCfu scaffolds.
 *
 * An agent run compiles a per-run workspace copy of AgentCfu.scala /
 * AgentCfuFiber.scala. Because Scala cannot hold two definitions of the same
 * class in one package, the repository copies must be dropped from the source
 * list; the workspace copies then provide the class. Filtering the resolved
 * source list is deterministic and needs no shell metacharacters in the sbt
 * command line, which keeps the agent command policy check meaningful.
 */
object AgentCfuSourceFilter {
  private val scaffoldSuffixes = Seq(
    "src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala",
    "src/main/scala/vexiiriscv/soc/mico/AgentCfuFiber.scala"
  )

  private val sourceSuffixes = Seq(".scala", ".java")

  /** True for the tracked repository scaffolds that an overlay replaces. */
  def isRepoScaffold(file: File): Boolean = {
    val path = file.getAbsolutePath.replace(File.separatorChar, '/')
    scaffoldSuffixes.exists(path.endsWith)
  }

  /** Expand source directories into a filtered source list. */
  def sources(dirs: Seq[File]): Seq[File] = {
    def expand(file: File): Seq[File] = {
      if (file.isFile) Seq(file)
      else if (file.isDirectory) Option(file.listFiles).toSeq.flatten.sortBy(_.getName).flatMap(expand)
      else Seq.empty
    }
    dirs.flatMap(expand).filter(f => sourceSuffixes.exists(f.getName.endsWith)).filterNot(isRepoScaffold)
  }
}
