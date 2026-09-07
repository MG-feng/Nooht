# P0-2 执行证据
生成时间: 2026-09-07T11:23:48.845784+00:00

## 0. deploy_vendor
```
deployed: __init__.py -> C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\__init__.py
deployed: configuration_jamba.py -> C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\configuration_jamba.py
deployed: modeling_jamba.py -> C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\modeling_jamba.py
deployed: modular_jamba.py -> C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\modular_jamba.py
cleared: C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\__pycache__
[OK] vendored Jamba 部署完成
```
(exit code: 0)

## 1. 环境版本
- python: 3.14.4
- torch: 2.14.0+cpu
- transformers: 5.16.1

## 2. inspect_jamba.py 实文输出
```
# modeling_jamba scan 调用链确认
source: C:\Users\fengm\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\transformers\models\jamba\modeling_jamba.py

## 模块级 kernel 定义
- causal_conv1d_update @L270
- causal_conv1d_fn @L294
- mamba_inner_fn @L321
- mamba_selective_state_update @L349
- mamba_selective_scan @L392

## kernel 相关 import
- is_mambapy_available ← utils.import_utils

## 调用链表
- JambaMambaDecoderLayer > __init__()  →  JambaMambaMixer()  [kind=name receiver=None]  [local/unknown]  @L803
- JambaMambaDecoderLayer > forward()  →  mamba()  [kind=self_method receiver=self]  [self/cls 方法调用 ← 模块级 patch 不可达，需类方法级 patch]  @L820
- JambaMambaMixer > __init__()  →  Conv1d()  [kind=attribute_call receiver=nn]  [receiver='nn' 属性调用 ← 需核实 receiver 是否为模块对象；若是，可在其所属模块做 patch]  @L518
- JambaMambaMixer > forward()  →  causal_conv1d_fn()  [kind=name receiver=None]  [module-level def @L294]  @L593
- JambaMambaMixer > forward()  →  causal_conv1d_update()  [kind=name receiver=None]  [module-level def @L270]  @L579
- JambaMambaMixer > forward()  →  mamba_selective_scan()  [kind=name receiver=None]  [module-level def @L392]  @L641
- JambaMambaMixer > forward()  →  mamba_selective_state_update()  [kind=name receiver=None]  [module-level def @L349]  @L626
- mamba_selective_scan()  →  is_mambapy_available()  [kind=name receiver=None]  [import from utils.import_utils]  @L412
- mamba_selective_scan()  →  associative_scan()  [kind=name receiver=None]  [local/unknown]  @L451
- mamba_selective_scan()  →  pscan()  [kind=name receiver=None]  [local/unknown]  @L435
- causal_conv1d_fn()  →  conv1d()  [kind=attribute_call receiver=F]  [receiver='F' 属性调用 ← 需核实 receiver 是否为模块对象；若是，可在其所属模块做 patch]  @L303
- causal_conv1d_update()  →  conv1d()  [kind=attribute_call receiver=F]  [receiver='F' 属性调用 ← 需核实 receiver 是否为模块对象；若是，可在其所属模块做 patch]  @L281

## 结论
[WARN] 存在 receiver 属性调用：需确认 receiver 是否为模块对象，并在其所属模块上做 patch（静态分析无法自证）
```
(exit code: 0)

## 3. verify_dispatch() 实文输出
```
VerifyResult(dispatched=True, kernel_takeover=True, numerics_match=True, hook_queries={'selective_scan': 2, 'selective_state_update': 1, 'causal_conv1d_fn': 2, 'causal_conv1d_update': 1}, takeover_counts={'selective_scan': 2, 'selective_state_update': 1, 'causal_conv1d_fn': 2, 'causal_conv1d_update': 1}, missing_kernels=[], error=None)
```

- dispatched=True
- kernel_takeover=True
- numerics_match=True
- missing_kernels=[]
- takeover_counts={'selective_scan': 2, 'selective_state_update': 1, 'causal_conv1d_fn': 2, 'causal_conv1d_update': 1}
- hook_queries={'selective_scan': 2, 'selective_state_update': 1, 'causal_conv1d_fn': 2, 'causal_conv1d_update': 1}
- **GATE: PASS**

## 4. pytest tests -v 结果（尾部摘要）
```
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-False-dtype1-32] PASSED [ 75%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype0-1] PASSED [ 76%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype0-7] PASSED [ 76%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype0-32] PASSED [ 77%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype1-1] PASSED [ 78%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype1-7] PASSED [ 78%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-False-True-dtype1-32] PASSED [ 79%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype0-1] PASSED [ 80%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype0-7] PASSED [ 80%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype0-32] PASSED [ 81%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype1-1] PASSED [ 82%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype1-7] PASSED [ 82%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-False-dtype1-32] PASSED [ 83%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype0-1] PASSED [ 84%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype0-7] PASSED [ 84%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype0-32] PASSED [ 85%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype1-1] PASSED [ 86%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype1-7] PASSED [ 86%]
tests/test_reference_scan_contract.py::test_forward_backward_contract[True-True-True-True-dtype1-32] PASSED [ 87%]
tests/test_reference_scan_contract.py::test_last_state_semantics PASSED  [ 88%]
tests/test_reference_scan_contract.py::test_bfloat16_forward_only[1] PASSED [ 88%]
tests/test_reference_scan_contract.py::test_bfloat16_forward_only[32] PASSED [ 89%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-True-1] PASSED [ 90%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-True-2] PASSED [ 90%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-True-3] PASSED [ 91%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-False-1] PASSED [ 92%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-False-2] PASSED [ 92%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[False-False-3] PASSED [ 93%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-True-1] PASSED [ 94%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-True-2] PASSED [ 94%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-True-3] PASSED [ 95%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-False-1] PASSED [ 96%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-False-2] PASSED [ 96%]
tests/test_reference_scan_contract.py::test_scalar_recurrence_oracle[True-False-3] PASSED [ 97%]
tests/test_reference_scan_contract.py::test_reference_conv_kernels_contract PASSED [ 98%]
tests/test_reference_scan_contract.py::test_reference_conv_update_mutates_state_in_place PASSED [ 98%]
tests/test_reference_scan_contract.py::test_reference_conv_update_state_content_and_activation PASSED [ 99%]
tests/test_vendor_sync.py::test_modeling_and_modular_hook_in_sync PASSED [100%]

============================ 152 passed in 38.37s =============================
```
(exit code: 0)
