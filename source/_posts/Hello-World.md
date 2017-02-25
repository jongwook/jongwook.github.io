title: Hello World
tags: ['Hexo', 'LaTeX', 'KaTeX']
categories: ['Blog']
---

I have just set up a new blog with [Hexo](https://hexo.io/), and made a slightly modified version of [hexo-tag-katex](https://github.com/jongwook/hexo-tag-katex) to compile LaTeX expressions inside dollar signs, instead of using Hexo tags. [KaTeX](https://khan.github.io/KaTeX/) is a wonderful library built by [Khan Academy](https://www.khanacademy.org/), which renders LaTeX equations a lot faster and more beautifully than MathJax.

\[
H(T) \left | \psi(t) \right \rangle = i \hbar \frac{\partial}{\partial t} \left | \psi(t) \right \rangle
\]

It is also very easy to include code snippets like:

```scala
import com.kakao.cuesheet

object MySparkJob extends CueSheet {{
  spark.sql("show databases").show();
}}
```

I like this hackability of Hexo, in part because I am less familiar with Ruby. I am going to make this my personal website, migrating from Tumbler which started to put a video ad on their page.
