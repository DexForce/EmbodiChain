# Policy Models

| Model | Use it when | Inputs and outputs |
| --- | --- | --- |
| [Neural Motion Generator](nmg.md) | An arm needs to visit a sequence of joint or Cartesian goals | Joint state, tool pose and encoded goals → one action per controlled joint |
| [Action Chunking Transformer (ACT)](../../api_reference/visual_act.rst) | A policy trained from visual demonstrations should drive the robot | Camera images and robot observations → action chunks processed into physical commands |

```{toctree}
:hidden:

nmg
ACT <../../api_reference/visual_act>
```
