---
license: apache-2.0
task_categories:
- question-answering
tags:
- gui
- multimodal
- cross-platform
- critic-model
- os-oracle
pretty_name: OS-Critic-Bench
size_categories:
- n<1K
configs:
- config_name: OS-Critic-Bench
  data_files:
  - split: mobile
    path: mobile.jsonl
  - split: desktop
    path: desktop.jsonl
  - split: web
    path: web.jsonl
---
