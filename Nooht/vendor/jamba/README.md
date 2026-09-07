# vendor/jamba —— vendored Jamba 文件

来源：https://github.com/huggingface/transformers/tree/main/src/transformers/models/jamba
（钉定版本：transformers==4.46.3）

## 文件状态
| 文件 | 状态 |
|---|---|
| __init__.py | 官方原版，未改动 |
| configuration_jamba.py | 官方原版，未改动 |
| modular_jamba.py | 插入 Nooht hook（唯一事实源） |
| modeling_jamba.py | 含 Nooht hook 的同步产物（tests/test_vendor_sync.py + CI 再生 job 双重防漂移） |

## 部署
    python scripts/deploy_vendor.py          # 复制到已安装的 transformers/models/jamba/
    python scripts/deploy_vendor.py --check  # 校验部署版本与 vendor 一致

## 同步规则
- 任何 hook 改动只进 modular_jamba.py；
- 改动后同步 modeling_jamba.py（重新生成或手工同步），保持两者 AST 一致；
- 升级上游 transformers 时重新执行「替换 → 插入 hook → 同步 → 重跑 P0-2 门禁」。