# Training trajectories

Selected from Together AI's
[CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview).
Each file contains the original trajectory records; only Parquet packaging
and compression have changed.

| File | Selection | Trajectories | ID manifest |
| --- | --- | ---: | --- |
| [`top.parquet`](top.parquet) | Top-TQS | 1,024 | [`top.txt`](../training/splits/top.txt) |
| [`bottom.parquet`](bottom.parquet) | Bottom-TQS | 1,024 | [`bottom.txt`](../training/splits/bottom.txt) |
| [`random.parquet`](random.parquet) | Random | 1,024 | [`random.txt`](../training/splits/random.txt) |

Columns: `trajectory_id`, `finish_reason`, `image`, `messages`, `reward`,
`tools`, and `license`. `messages` and `tools` are JSON strings; `image` is
the upstream container-image reference. Every trajectory has `reward = 1.0`.
See the [training guide](../training/README.md) for conversion and training.

## Source and licensing

Credit: Alpay Ariyak and the CoderForge-Preview collaborators at Together AI;
see the [dataset release](https://www.together.ai/blog/coderforge-preview).
Original IDs, repository references, messages, tool definitions, and per-row
`license` values are preserved. Upstream data and repository content retain
their own licenses; the TQS code's MIT license does not relicense these records.
